use core::sync::atomic::Ordering::*;
use core::sync::atomic::{AtomicBool, fence};
use std::sync::Arc;
use std::thread;

use crossbeam::channel::Sender;
use heapless::spsc::Consumer;
use rubato::Resampler;
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;

use crate::ResamplerCfg;

pub struct WorkerThread {
    stop_flag: Arc<AtomicBool>,
    inner_thread_handle: Option<thread::JoinHandle<anyhow::Result<()>>>,
}

impl WorkerThread {
    pub fn new(
        ResamplerCfg {
            input_rate,
            target_rate,
            chunk_size,
        }: ResamplerCfg,
        mut sample_rx: Consumer<'static, f32>,
        resampled_tx: Sender<Vec<f32>>,
    ) -> anyhow::Result<WorkerThread> {
        let stop_flag = Arc::new(AtomicBool::new(false));

        let mut resampler = rubato::Async::<f32>::new_sinc(
            target_rate as f64 / input_rate as f64,
            2.0,
            &rubato::SincInterpolationParameters {
                sinc_len: 128,
                f_cutoff: 0.95,
                interpolation: rubato::SincInterpolationType::Linear,
                oversampling_factor: 128,
                window: rubato::WindowFunction::BlackmanHarris2,
            },
            chunk_size,
            1,
            rubato::FixedAsync::Input,
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
                    fence(Release);

                    // next up consume the values
                    // (IDK HOW THO TIHI)
                    while let Some(sample_point) = sample_rx.dequeue() {
                        push!(sample_point);

                        // have we hit the send-off limit?
                        if len!() == chunk_size {
                            let in_adapter = SequentialSliceOfVecs::new(&accumulator, 1, in_frames)
                                .inspect_err(|e| {
                                    tracing::error!("failed to create in-adapter: {e}")
                                })?;
                            let mut out_adapter =
                                SequentialSliceOfVecs::new_mut(&mut out_buf, 1, out_frames)
                                    .inspect_err(|e| {
                                        tracing::error!("failed to create out-adapter: {e}")
                                    })?;
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

        let worker_thread = WorkerThread {
            stop_flag,
            inner_thread_handle: Some(thread_handle),
        };

        Ok(worker_thread)
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
