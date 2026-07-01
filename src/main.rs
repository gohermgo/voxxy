use core::sync::atomic::Ordering::*;
use core::sync::atomic::{AtomicBool, fence};
use std::sync::Arc;
use std::thread;

use cpal::traits::HostTrait;
use heapless::spsc::Queue;
use tracing::level_filters::STATIC_MAX_LEVEL;

struct ResamplerCfg {
    input_rate: cpal::SampleRate,
    target_rate: cpal::SampleRate,
    chunk_size: usize,
}

mod audio_thread;
mod worker_thread;

#[expect(dead_code)]
struct WriterThread {
    stop_flag: Arc<AtomicBool>,
    inner_thread_handle: Option<thread::JoinHandle<anyhow::Result<()>>>,
}

impl WriterThread {
    #[expect(dead_code)]
    pub fn new(
        resampled_rx: crossbeam::channel::Receiver<Vec<f32>>,
        spec: hound::WavSpec,
    ) -> anyhow::Result<WriterThread> {
        let stop_flag = Arc::new(AtomicBool::new(false));

        let validation_thread = thread::spawn({
            tracing::info!("making validation thread...");
            let mut w = hound::WavWriter::create("resampled_16k.wav", spec)?;
            let stop = Arc::clone(&stop_flag);
            move || -> anyhow::Result<()> {
                // fake that we are consuming from this thread
                loop {
                    if stop.load(Acquire) {
                        tracing::info!("leaving writer thread!");
                        w.finalize()?;
                        break Ok(());
                    }
                    fence(Release);

                    for sample in resampled_rx.try_iter().flatten() {
                        w.write_sample(sample)?;
                    }
                }
            }
        });

        Ok(WriterThread {
            stop_flag,
            inner_thread_handle: Some(validation_thread),
        })
    }
}

impl Drop for WriterThread {
    fn drop(&mut self) {
        tracing::trace!("writer thread dropping!");
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

fn main() -> anyhow::Result<()> {
    use audio_thread::AudioThread;
    use worker_thread::WorkerThread;

    tracing_subscriber::fmt()
        .with_max_level(STATIC_MAX_LEVEL)
        .with_thread_ids(true)
        .init();
    let host = cpal::default_host();
    let input_dev = host
        .default_input_device()
        .expect("default input device missing");

    // realtime concerns, we need a heapless channel here
    let q = Box::leak(Box::new(Queue::<f32, 2048>::new()));

    let (sample_tx, sample_rx) = q.split();
    let (resampled_tx, resampled_rx) = crossbeam::channel::bounded(8);

    let (audio_thread, resampler_cfg) = AudioThread::new(input_dev, sample_tx)?;

    // #[expect(dead_code)]
    // let spec = hound::WavSpec {
    //     channels: 1,
    //     sample_rate: resampler_cfg.target_rate,
    //     bits_per_sample: 32,
    //     sample_format: hound::SampleFormat::Float,
    // };

    let sample_rate = resampler_cfg.target_rate;
    let worker_thread = WorkerThread::new(resampler_cfg, sample_rx, resampled_tx)?;

    let mut lpc_framer = LpcFramer {
        accumulator: Vec::default(),
        window_size: 400,
        hop_size: 67,
    };

    let mut lpc_buffers = LpcBuffers::new(lpc_framer.window_size, LPC_ORDER);

    let stop_flag = Arc::new(AtomicBool::new(false));

    let (formant_tx, formant_rx) = crossbeam::channel::bounded::<[f32; 4]>(64);

    let _validation_thread = thread::spawn({
        tracing::info!("making lpc thread...");
        let stop = Arc::clone(&stop_flag);
        move || -> anyhow::Result<()> {
            // fake that we are consuming from this thread
            loop {
                if stop.load(Acquire) {
                    tracing::info!("leaving lpc thread!");
                    break Ok(());
                }
                // blocks until data arrives OR timeout, no spin
                match resampled_rx.recv_timeout(std::time::Duration::from_millis(5)) {
                    Ok(new_samples) => {
                        lpc_framer.push_and_maybe_analyze(&new_samples, |in_buf| {
                            lpc_buffers.run_pipeline(in_buf, LPC_ORDER, sample_rate, 400.0, 0.15);
                            let mut f = [0f32; 4];
                            f[..lpc_buffers.formant.len().min(4)].copy_from_slice(
                                &lpc_buffers.formant[..lpc_buffers.formant.len().min(4)],
                            );
                            let _ = formant_tx.try_send(f);
                        });
                    }
                    Err(crossbeam::channel::RecvTimeoutError::Timeout) => continue,
                    Err(crossbeam::channel::RecvTimeoutError::Disconnected) => break Ok(()),
                }
            }
        }
    });

    // here is our start point!

    audio_thread.play()?;

    eframe::run_native(
        "voxxy",
        eframe::NativeOptions::default(),
        Box::new(|_cc| {
            Ok(Box::new(FormantApp {
                formant_rx,
                history: std::collections::VecDeque::with_capacity(300),
                last_valid: [0.; 4],
                smoothed: [0.; 4],
                is_voiced: false,
            }))
        }),
    )?;

    // let t = std::time::Duration::from_secs(1);

    // thread::sleep(t);
    tracing::debug!("done!");
    drop(audio_thread);
    drop(worker_thread);

    Ok(())
}

use egui_plot::{Plot, PlotPoints, Points};

struct FormantApp {
    formant_rx: crossbeam::channel::Receiver<[f32; 4]>,
    history: std::collections::VecDeque<[f32; 4]>,
    last_valid: [f32; 4],
    // for EMA, makes data smoother
    smoothed: [f32; 4],
    is_voiced: bool,
}

impl eframe::App for FormantApp {
    fn logic(&mut self, _: &egui::Context, _: &mut eframe::Frame) {
        while let Ok(f) = self.formant_rx.try_recv() {
            // only update if at least F1 is real, otherwise keep last known
            if f[0] > 0.0 {
                self.is_voiced = true;
                const ALPHA: f32 = 0.3;
                const SNAP_THRESHOLD_HZ: f32 = 300.0; // jump bigger than this = snap

                #[allow(clippy::needless_range_loop)]
                for i in 0..4 {
                    if f[i] > 0.0 {
                        if self.last_valid[i] == 0.0
                            || (f[i] - self.smoothed[i]).abs() > SNAP_THRESHOLD_HZ
                        {
                            // big jump or coming from silence = snap immediately
                            self.smoothed[i] = f[i];
                        } else {
                            // small variation = smooth it
                            self.smoothed[i] = ALPHA * f[i] + (1.0 - ALPHA) * self.smoothed[i];
                        }
                    }
                }
                self.last_valid = self.smoothed;
                if self.history.len() == 300 {
                    self.history.pop_front();
                }
                self.history.push_back(self.smoothed);
            } else {
                // voiced -> silence: zero out last_valid so next onset snaps
                self.is_voiced = false;
                self.last_valid = [0.0; 4];
            }
        }
    }
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        const COLORS: [egui::Color32; 4] = [
            egui::Color32::from_rgb(220, 80, 80),  // F1 red
            egui::Color32::from_rgb(80, 140, 220), // F2 blue
            egui::Color32::from_rgb(140, 200, 80), // F3 green
            egui::Color32::from_rgb(200, 80, 200), // F4 pink
        ];
        egui::CentralPanel::default().show(ui, |ui| {
            // bottom legend FIRST so plot takes remaining space
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.horizontal(|ui| {
                    #[allow(clippy::needless_range_loop)]
                    for fi in 0..4 {
                        let hz = self.last_valid[fi];
                        let text = if self.is_voiced && self.smoothed[fi] > 0.0 {
                            format!("F{}  {:.0} Hz", fi + 1, hz)
                        } else {
                            format!("F{}  ---", fi + 1)
                        };
                        ui.label(
                            egui::RichText::new(text)
                                .color(COLORS[fi])
                                .strong()
                                .size(18.0),
                        );
                        ui.add_space(24.0);
                    }
                });

                // plot gets everything above the legend
                Plot::new("formants")
                    .include_y(0.0)
                    .include_y(8000.0)
                    .show(ui, |plot_ui| {
                        for fi in 0..4 {
                            let pts: PlotPoints = self
                                .history
                                .iter()
                                .enumerate()
                                .filter(|(_, f)| f[fi] > 0.0)
                                .map(|(i, f)| [i as f64, f[fi] as f64])
                                .collect();

                            plot_ui.points(
                                Points::new(format!("F{}", fi + 1), pts)
                                    .radius(3.0)
                                    .color(COLORS[fi]),
                            );

                            // bold label floating on the line itself
                            if self.last_valid[fi] > 0.0 {
                                plot_ui.text(egui_plot::Text::new(
                                    format!("label-F{}", fi + 1),
                                    egui_plot::PlotPoint::new(295.0, self.last_valid[fi] as f64),
                                    egui::RichText::new(format!("F{}", fi + 1))
                                        .color(COLORS[fi])
                                        .strong()
                                        .size(15.0),
                                ));
                            }
                        }
                    });
            });
        });

        ui.request_repaint();
    }
}

// LPC algorithm sketch
//
// 1. windowing + pre-emphasis
//
//     y[n] = x[n] - 0.97*x[n-1]
//
// 2. autocorrelation
//
//     compute autocorrelation of the windowed frame out to lag = LPC order.
//     straightforward sum-of-products loop, O(n*order), cheap.
//
// 3. levinson-durbin
//
//     turns the autocorrelation sequence into LPC coefficients in the all-pole filter.
//
// 4. formants from coefficients
//
//     we gotta use nalgebra here, basically formant live at roots of the LPC polynomial
//     A(z) = 1 - a_1*z^-1 - a_2*z^-2 - ...
//
//     specifically the complex-conjugate root pairs that land near the unit circle.
//     find roots via the companion matrix's eigenvalues (nalgebra::DMatrix + its eigenvalue solver
//     handles this clean), then for each complex root (r = a + bi):
//     angle theta = atan2(b, a),
//     frequency f = theta * sample_rate / 2pi,
//     bandwidth comes from magnitude |r| (closer to 1.0 = narrower/sharper resonance, real formant,
//     further from 1.0 = probably noise/spurious pole, to be filtered out)
//     sort surviving roots by frequency asc., first 3-4 are ur F1-F4

/// applies pre-emphasis filter then a Hamming window, in that order.
/// out_buf[i] corresponds to in_buf[i], same length required.
fn lpc_window_and_pre_emphasize(in_buf: &[f32], out_buf: &mut [f32]) {
    debug_assert_eq!(
        out_buf.len(),
        in_buf.len(),
        "input and output buffer sizes need to match"
    );

    let n = in_buf.len();

    const PRE_EMPHASIS: f32 = 0.97;

    for i in 0..n {
        // pre-emphasis: y[n] = x[n] - 0.97*x[n-1], x[-1] treated as 0
        let prev = if i == 0 { 0.0 } else { in_buf[i - 1] };
        let emphasized = in_buf[i] - PRE_EMPHASIS * prev;

        // hamming window coefficient for this index
        let hamming_window_coeff =
            0.54 - 0.46 * (2.0 * core::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos();

        out_buf[i] = emphasized * hamming_window_coeff;
    }
}

fn lpc_autocorrelate(in_buf: &[f32], out_buf: &mut [f32], lpc_order: usize) {
    let n = in_buf.len();
    debug_assert!(lpc_order < n && out_buf.len() > lpc_order);

    for lag in 0..=lpc_order {
        let mut sum = 0.0;
        for t in lag..n {
            sum += in_buf[t] * in_buf[t - lag];
        }
        out_buf[lag] = sum;
    }
}

/// `autocorrelation_buf` is input autocorrelation vector of size `p + 1`
/// `output_coeff_buf` is output coefficient buffer of size `p`
/// `reflection_coeff_buf` is reflection coefficient buffer of size `p`
fn lpc_levinson_durbin(
    autocorrelation_buf: &[f32],
    output_coeff_buf: &mut [f32],
    reflection_coeff_buf: &mut [f32],
) {
    let n = output_coeff_buf.len();

    // initial error energy
    let mut e = autocorrelation_buf[0];

    for i in 0..n {
        // 1. compute reflection coefficient (PARCOR)
        let mut sum = 0.0;
        for j in 0..i {
            sum += output_coeff_buf[j] * autocorrelation_buf[i - j];
        }
        let reflection_coeff = -(autocorrelation_buf[i + 1] + sum) / e;
        reflection_coeff_buf[i] = reflection_coeff;

        // 2. update existing coefficients from the edges inwards
        let midpoint = i.div_ceil(2);
        for j in 0..midpoint {
            let back_idx = i - 1 - j;
            let output_coeff = output_coeff_buf[j];

            if j == back_idx {
                // center element, iff i is even
                output_coeff_buf[j] = output_coeff + reflection_coeff * output_coeff;
            } else {
                let output_coeff_back = output_coeff_buf[back_idx];
                output_coeff_buf[j] = output_coeff + reflection_coeff * output_coeff_back;
                output_coeff_buf[back_idx] = output_coeff_back + reflection_coeff * output_coeff;
            }
        }

        // write to i for the first time i suppose...
        output_coeff_buf[i] = reflection_coeff;

        // 3. update the error energy
        e *= 1.0 - reflection_coeff * reflection_coeff;
    }
}

/// `in_buf` should be the `output_coeff_buf` from the levinson-durbin pass
/// apparently it should be without `a_0` = 1.0
fn lpc_extract_formants(
    in_buf: &[f32],
    out_buf: &mut [f32],
    sample_rate: u32,
    bandwidth_threshold_max: f32,
) {
    let n = in_buf.len();
    out_buf.fill(0.0);

    // 1. construct a n x n companion matrix

    // in-buf len should correspond to lpc order atp
    //
    // this is the companion matrix
    let mut m = nalgebra::DMatrix::zeros(n, n);

    // fill the top row with negative lpc-coefficients
    for j in 0..n {
        m[(0, j)] = -in_buf[j];
    }

    // add 1.0 on the subdiagonal (directly below the main diagonal)
    for i in 1..n {
        m[(i, i - 1)] = 1.0;
    }

    // 2. calculate eigenvalues using QR algorithm
    //
    // nalgebra uses their own eigen_qr primarily returning real matrixes,
    // and uses thus schur-decomp where complex values act as 2x2 blocks on the diagonal
    let schur = m.schur();
    let complex_eigenvalues = schur.complex_eigenvalues();

    // 3. convert the complex poles to physical frequencies (formants)
    let mut count = 0;
    for eig in complex_eigenvalues.iter() {
        if eig.im > 0.0 {
            let theta = eig.im.atan2(eig.re);
            let frequency = (theta * sample_rate as f32) / (2.0 * core::f32::consts::PI);

            let magnitude = (eig.re * eig.re + eig.im * eig.im).sqrt();
            let bandwidth = -magnitude.ln() * sample_rate as f32 / core::f32::consts::PI;

            // filter noise and avoid out of bounds
            #[allow(clippy::collapsible_if)]
            if frequency > 250.0
                && frequency < (sample_rate as f32 / 2.0) - 200.0
                && bandwidth < bandwidth_threshold_max
            {
                if count < out_buf.len() {
                    out_buf[count] = frequency;
                    count += 1;
                }
            }
        }
    }

    // 4. sort only elements we added
    out_buf[0..count].sort_by(|a, b| a.partial_cmp(b).unwrap());
}

// at least 16, at most 18 for our 16k
//
// this should be configurable or like
// we should do math or smth, but idk
// those formulas yet
const LPC_ORDER: usize = 18;

struct LpcBuffers {
    cleaned_data: Vec<f32>,
    autocorrelation: Vec<f32>,
    output_coeff: Vec<f32>,
    reflection_coeff: Vec<f32>,
    formant: Vec<f32>,
}

fn is_voiced(in_buf: &[f32], threshold: f32) -> bool {
    let rms = (in_buf.iter().map(|s| s * s).sum::<f32>() / in_buf.len() as f32).sqrt();
    rms > threshold
}

impl LpcBuffers {
    pub fn new(window_size: usize, lpc_order: usize) -> LpcBuffers {
        LpcBuffers {
            cleaned_data: vec![0.0; window_size],
            autocorrelation: vec![0.0; lpc_order + 1],
            output_coeff: vec![0.0; lpc_order],
            reflection_coeff: vec![0.0; lpc_order],
            formant: vec![0.0; lpc_order / 2],
        }
    }

    pub fn run_pipeline(
        &mut self,
        in_buf: &[f32],
        lpc_order: usize,
        sample_rate: u32,
        bandwidth_threshold_max: f32,
        unvoiced_threshold: f32,
    ) {
        if !is_voiced(in_buf, unvoiced_threshold) {
            self.formant.fill(0.0);
            return;
        }
        lpc_window_and_pre_emphasize(in_buf, &mut self.cleaned_data);
        lpc_autocorrelate(&self.cleaned_data, &mut self.autocorrelation, lpc_order);
        lpc_levinson_durbin(
            &self.autocorrelation,
            &mut self.output_coeff,
            &mut self.reflection_coeff,
        );
        lpc_extract_formants(
            &self.output_coeff,
            &mut self.formant,
            sample_rate,
            bandwidth_threshold_max,
        );
    }
}

struct LpcFramer {
    accumulator: Vec<f32>,
    window_size: usize,
    // < window_size for overlapping frames, smoother formant tracking
    hop_size: usize,
}

impl LpcFramer {
    fn push_and_maybe_analyze(&mut self, new_samples: &[f32], mut on_frame: impl FnMut(&[f32])) {
        self.accumulator.extend_from_slice(new_samples);
        while self.accumulator.len() >= self.window_size {
            on_frame(&self.accumulator[..self.window_size]);
            self.accumulator.drain(..self.hop_size);
        }
    }
}

mod signal {
    pub struct SignalBuf(Vec<f32>);
}
