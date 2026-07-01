//! # the worker-thread
//!
//! really great name, no? it isn't actually.
//!
//! basically, we downsample so we can do even faster
//! processing later on. by reducing the sample rate experienced
//! by the spectral components, we can have closer to the wanted
//! real-time visualization
//!

use core::sync::atomic::AtomicBool;
use core::sync::atomic::Ordering::*;

use std::sync::Arc;
use std::thread;

use crossbeam::channel::Sender;
use heapless::spsc::Consumer;
use rubato::Resampler;
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Async, FixedAsync};
use rubato::{SincInterpolationParameters, SincInterpolationType, WindowFunction};

use crate::ResamplerCfg;

pub struct WorkerThread {
    /// used to stop the thread on `Drop`
    stop_flag: Arc<AtomicBool>,
    /// we wrap in an `Option` so we can use [`take`](Option::take) in `Drop`
    inner_thread_handle: Option<thread::JoinHandle<anyhow::Result<()>>>,
}

impl WorkerThread {
    pub fn new(
        resampler_cfg: ResamplerCfg,
        // input for raw samples
        sample_rx: Consumer<'static, f32>,
        // output for resampled frames
        resampled_tx: Sender<Vec<f32>>,
    ) -> anyhow::Result<WorkerThread> {
        let default_resampler_parameters = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.88,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };
        WorkerThread::new_with_params(
            resampler_cfg,
            sample_rx,
            resampled_tx,
            default_resampler_parameters,
        )
    }
    pub fn new_with_params(
        ResamplerCfg {
            input_rate,
            target_rate,
            chunk_size,
        }: ResamplerCfg,
        mut sample_rx: Consumer<'static, f32>,
        resampled_tx: Sender<Vec<f32>>,
        resampler_parameters: SincInterpolationParameters,
    ) -> anyhow::Result<WorkerThread> {
        let stop_flag = Arc::new(AtomicBool::new(false));

        let mut resampler = Async::new_sinc(
            target_rate as f64 / input_rate as f64,
            2.0,
            &resampler_parameters,
            chunk_size,
            1,
            FixedAsync::Input,
        )?;

        let (in_frames, out_frames) = (resampler.input_frames_max(), resampler.output_frames_max());

        let mut accumulator: Vec<Vec<f32>> = vec![Vec::with_capacity(in_frames); 1];
        let mut out_buf: Vec<Vec<f32>> = vec![vec![0_f32; out_frames]; 1];

        // some small helpers so we always modify accumulator[0]
        macro_rules! push {
            ($t:expr) => {
                accumulator[0].push($t)
            };
        }
        macro_rules! len {
            () => {
                accumulator[0].len()
            };
        }
        macro_rules! clear {
            () => {
                accumulator[0].clear()
            };
        }

        let thread_handle = thread::spawn({
            let stop = Arc::clone(&stop_flag);
            move || {
                loop {
                    // first check if we should continue at all
                    if stop.load(Acquire) {
                        tracing::debug!("stop flag! breaking...");
                        break Ok(());
                    }

                    // next up consume the values
                    while let Some(sample_point) = sample_rx.dequeue() {
                        push!(sample_point);

                        // have we hit the send-off limit?
                        if len!() >= chunk_size {
                            let in_adapter =
                                SequentialSliceOfVecs::new(&accumulator, 1, in_frames)?;
                            let mut out_adapter =
                                SequentialSliceOfVecs::new_mut(&mut out_buf, 1, out_frames)?;
                            let (_in_used, out_written) = resampler
                                .process_into_buffer(&in_adapter, &mut out_adapter, None)
                                .inspect_err(|e| tracing::error!("failed to resample: {e}"))?;

                            if resampled_tx
                                .try_send(out_buf[0][..out_written].to_vec())
                                .is_err()
                            {
                                tracing::warn!("ui consumer falling behind, dropping chunk");
                            }

                            // make sure to wipe
                            clear!();
                        };
                    }
                }
            }
        });

        Ok(WorkerThread {
            stop_flag,
            inner_thread_handle: Some(thread_handle),
        })
    }
}

impl Drop for WorkerThread {
    fn drop(&mut self) {
        self.stop_flag.store(true, Release);
        if let Some(handle) = self.inner_thread_handle.take() {
            if let Err(e) = handle.join() {
                tracing::error!("{e:?}")
            };
        } else {
            tracing::warn!("worker thread handle missing...");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use heapless::spsc::Queue;
    use rustfft::{FftPlanner, num_complex::Complex};

    fn leak_queue<const N: usize>() -> &'static mut Queue<f32, N> {
        Box::leak(Box::new(Queue::new()))
    }

    struct AliasTestResult {
        alias_energy: f32,
        passband_avg: f32,
        ratio: f32,
    }

    fn run_alias_test(
        f_cutoff: f32,
        window: WindowFunction,
        interpolation: SincInterpolationType,
        input_rate: u32,
    ) -> anyhow::Result<AliasTestResult> {
        const TARGET_RATE: u32 = 16_000;
        const CHUNK_SIZE: usize = 1024;

        let queue: &'static mut Queue<f32, 144_000> = leak_queue::<144_000>();
        let (mut producer, consumer) = queue.split();
        let (resampled_tx, resampled_rx) = crossbeam::channel::unbounded();

        let resampler_parameters = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff,
            interpolation,
            oversampling_factor: 128,
            window,
        };

        let cfg = ResamplerCfg {
            input_rate,
            target_rate: TARGET_RATE,
            chunk_size: CHUNK_SIZE,
        };

        let _worker =
            WorkerThread::new_with_params(cfg, consumer, resampled_tx, resampler_parameters)?;

        // ─────────────────────────────────────────────
        // TEST SIGNAL (single tone above nyquist)
        // ─────────────────────────────────────────────
        let test_freq = 10_000.0_f32;
        for i in 0..input_rate as usize {
            let t = i as f32 / input_rate as f32;
            producer
                .enqueue((2.0 * std::f32::consts::PI * test_freq * t).sin())
                .ok();
        }

        let mut output = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);

        while std::time::Instant::now() < deadline {
            match resampled_rx.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(chunk) => output.extend(chunk),
                Err(_) => break,
            }
        }

        anyhow::ensure!(!output.is_empty(), "resampler produced nothing");

        // remove only startup transient (not analysis bias)
        output.drain(0..256.min(output.len()));

        // ─────────────────────────────────────────────
        // FFT (NO zero padding)
        // ─────────────────────────────────────────────
        let fft_len = output.len();

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_len);

        let mut buffer: Vec<Complex<f32>> =
            output.iter().map(|&s| Complex { re: s, im: 0.0 }).collect();

        fft.process(&mut buffer);

        let bin_hz = TARGET_RATE as f32 / fft_len as f32;

        // power spectrum
        let power: Vec<f32> = buffer.iter().map(|c| c.norm_sqr()).collect();

        // ─────────────────────────────────────────────
        // SIGNAL vs ALIAS (tight bins only)
        // ─────────────────────────────────────────────
        let signal_freq = (test_freq % TARGET_RATE as f32).abs();
        let alias_freq = TARGET_RATE as f32 - signal_freq;

        fn band_power(power: &[f32], center: usize, width: usize) -> f32 {
            let lo = center.saturating_sub(width);
            let hi = (center + width).min(power.len() - 1);
            power[lo..=hi].iter().sum()
        }

        let alias_bin = (alias_freq / bin_hz).round() as usize;
        let signal_bin = (signal_freq / bin_hz).round() as usize;

        let alias_power = band_power(&power, alias_bin, 2);
        #[expect(unused)]
        let signal_power = band_power(&power, signal_bin, 2);

        // true stopband estimate (everything above nyquist of target system)
        let nyquist_bin = (TARGET_RATE as f32 / 2.0 / bin_hz) as usize;
        let stopband_power: f32 = power[nyquist_bin..].iter().sum();

        let passband_power: f32 = power[..nyquist_bin].iter().sum();

        #[expect(unused)]
        let stopband_avg = stopband_power / nyquist_bin as f32;

        Ok(AliasTestResult {
            alias_energy: alias_power,
            passband_avg: passband_power / nyquist_bin as f32,
            ratio: alias_power / (passband_power + 1e-9),
        })
    }
    #[test]
    fn sweep_window_functions() -> anyhow::Result<()> {
        let windows = [
            ("Hann", WindowFunction::Hann),
            ("Hann2", WindowFunction::Hann2),
            ("Blackman", WindowFunction::Blackman),
            ("Blackman2", WindowFunction::Blackman2),
            ("BlackmanHarris", WindowFunction::BlackmanHarris),
            ("BlackmanHarris2", WindowFunction::BlackmanHarris2),
        ];
        println!();
        for (name, w) in windows {
            let r = run_alias_test(0.95, w, SincInterpolationType::Nearest, 44_100)?;
            println!(
                "window={name} alias={:.4} passband_avg={:.4} ratio={:.4}",
                r.alias_energy, r.passband_avg, r.ratio
            );
        }
        Ok(())
    }
}
