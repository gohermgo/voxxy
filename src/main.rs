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

// at least 16, at most 18 for our 16k
//
// this should be configurable or like
// we should do math or smth, but idk
// those formulas yet
const LPC_ORDER: usize = 18;

/// napkin math gave me 400 for 44100 hz downsampled to 16000 hz,
/// provided we want the typical frame-duration of 25ms
const LPC_WINDOW_LEN: usize = 400;

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

    let sample_rate = resampler_cfg.target_rate;
    let worker_thread = WorkerThread::new(resampler_cfg, sample_rx, resampled_tx)?;

    let mut lpc_framer = LpcFramer {
        accumulator: Vec::default(),
        window_size: 400,
        hop_size: 67,
    };

    let stop_flag = Arc::new(AtomicBool::new(false));

    let (formant_tx, formant_rx) = crossbeam::channel::bounded::<[f32; 4]>(64);

    let mut pipeline = LpcPipeline::<LPC_ORDER, LPC_WINDOW_LEN, { LPC_ORDER + 1 }>::default();
    let mut formants = vec![0.; LPC_ORDER / 2].into_boxed_slice();

    // forwards to the framer
    let _lpc_thread = thread::spawn({
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
                        lpc_framer.push_and_maybe_analyze_arr(&new_samples, |x| {
                            if is_voiced(x, 0.15) {
                                let (b, _a) = pipeline.run_once(x);
                                lpc_extract_formants(b, &mut formants, sample_rate, 400.0);
                                // self.formant.fill(0.0);
                            }
                            // lpc_buffers.run_pipeline(in_buf, LPC_ORDER, sample_rate, 400.0, 0.15);
                            let mut f = [0f32; 4];
                            f[..formants.len().min(4)]
                                .copy_from_slice(&formants[..formants.len().min(4)]);
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
                    .include_y(6000.0)
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

fn hamming_window_coefficient(i: usize, n: usize) -> f32 {
    use core::f32::consts::PI;

    let angle = 2. * PI * i as f32 / (n as f32 - 1.);
    0.54 - (0.46 * f32::cos(angle))
}

/// applies pre-emphasis filter then a Hamming window, in that order.
/// out_buf[i] corresponds to in_buf[i], same length required.
fn lpc_window_and_pre_emphasize_arr<const N: usize>(x_in: &[f32; N], x_out: &mut [f32; N]) {
    // only for the 0-th case
    x_out[0] = x_in[0] * hamming_window_coefficient(0, N);

    const PRE_EMPHASIS: f32 = 0.97;

    // loop from 1 to the end
    for i in 1..N {
        // pre-emphasis: y[n] = x[n] - 0.97*x[n-1], x[-1] treated as 0, but we handle the 0 case outside
        let e = x_in[i] - PRE_EMPHASIS * x_in[i - 1];
        x_out[i] = e * hamming_window_coefficient(i, N)
    }
}

const fn lpc_autocorrelate_arr<const ORDER: usize, const N: usize, const X_A_LEN: usize>(
    x: &[f32; N],
    x_a: &mut [f32; X_A_LEN],
) {
    // ugly compile-time assertion of bounds
    struct Assert<const LHS: usize, const RHS: usize>;
    #[allow(dead_code)]
    impl<const LHS: usize, const RHS: usize> Assert<LHS, RHS> {
        const OK: () = assert!(
            LHS - 1 == RHS,
            "Autocorrelation buffer must be exactly 1 greater than LPC order!"
        );
    }
    let _: () = Assert::<X_A_LEN, ORDER>::OK;

    let mut i = 0;
    while i <= ORDER {
        let mut sum = 0.;
        let mut n = i;

        while n < N {
            sum += x[n] * x[n - i];

            n += 1;
        }

        x_a[i] = sum;

        i += 1;
    }
}

const fn lpc_levinson_durbin_arr<const ORDER: usize, const X_A_LEN: usize>(
    x_a: &[f32; X_A_LEN],
    b: &mut [f32; ORDER],
    a: &mut [f32; ORDER],
) {
    const fn feedback_coefficient<const ORDER: usize, const X_A_LEN: usize>(
        i: usize,
        x_a: &[f32; X_A_LEN],
        b: &[f32; ORDER],
        e: f32,
    ) -> f32 {
        let mut sum = 0.;

        let mut j = 0;
        while j < i {
            sum += b[j] * x_a[i - j];

            j += 1;
        }

        -(x_a[i + 1] + sum) / e
    }

    const fn update_forward_coefficients<const ORDER: usize>(
        i: usize,
        b: &mut [f32; ORDER],
        a_i: f32,
    ) {
        let midpoint = i.div_ceil(2);

        let mut j = 0;
        while j < midpoint {
            let a_j = b[j];
            if j == i - 1 - j {
                // center element, iff i is even (meaning we probably optimize out the branch due
                // to how common the indexing pattern is)
                b[j] = a_j + a_i * a_j;
            } else {
                let a_back = b[i - 1 - j];
                b[j] = a_j + a_i * a_back;
                b[i - 1 - j] = a_back + a_i * a_j;
            }

            j += 1;
        }
    }

    // initial error energy
    let mut e = x_a[0];

    let mut i = 0;
    while i < ORDER {
        // 1. feedback coefficient (PARCOR)
        let a_i = feedback_coefficient(i, x_a, b, e);
        a[i] = a_i;

        // 2. update existing coefficients from edges inwards
        update_forward_coefficients(i, b, a_i);
        b[i] = a_i;

        e *= 1.0 - a_i * a_i;

        i += 1;
    }
}

struct FilterCoefficients<const ORDER: usize> {
    /// the `forward` coefficients for
    /// the `auto-regressive` filter
    b: [f32; ORDER],
    /// the `feedback` coefficients for
    /// the `auto-regressive` filter
    a: [f32; ORDER],
}

impl<const ORDER: usize> Default for FilterCoefficients<ORDER> {
    fn default() -> Self {
        FilterCoefficients {
            b: [0.; ORDER],
            a: [0.; ORDER],
        }
    }
}

impl<const ORDER: usize> FilterCoefficients<ORDER> {
    /// estimate the filter-coefficients in-place, from the provided
    /// signal `x`
    ///
    /// it is assumed that `x` has already been windowed and pre-emphasized as wanted
    ///
    /// the autocorrelation buffer is request such that the caller
    /// can choose where the intermediate is stored
    const fn estimate<const X_LEN: usize, const X_A_LEN: usize>(
        &mut self,
        x: &[f32; X_LEN],
        x_a: &mut [f32; X_A_LEN],
    ) -> (&[f32; ORDER], &[f32; ORDER]) {
        lpc_autocorrelate_arr::<ORDER, X_LEN, X_A_LEN>(x, x_a);
        lpc_levinson_durbin_arr(x_a, &mut self.b, &mut self.a);
        (&self.b, &self.a)
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

fn is_voiced(in_buf: &[f32], threshold: f32) -> bool {
    let rms = (in_buf.iter().map(|s| s * s).sum::<f32>() / in_buf.len() as f32).sqrt();
    rms > threshold
}

/// basically we need to ensure the window size matches exactly
/// what parameters we precalculated.
///
/// for 16kHz => 400 samples (trust me ish, u can do the math but its not fun)
struct LpcFramer {
    /// we store samples as they arrive from the [audio thread](audio_thread::AudioThread)
    accumulator: Vec<f32>,
    #[expect(dead_code)]
    window_size: usize,
    // < window_size for overlapping frames, smoother formant tracking
    hop_size: usize,
}

impl LpcFramer {
    #[expect(dead_code)]
    fn push_and_maybe_analyze(&mut self, new_samples: &[f32], mut on_frame: impl FnMut(&[f32])) {
        self.accumulator.extend_from_slice(new_samples);
        while self.accumulator.len() >= self.window_size {
            on_frame(&self.accumulator[..self.window_size]);
            self.accumulator.drain(..self.hop_size);
        }
    }
    fn push_and_maybe_analyze_arr<const N: usize>(
        &mut self,
        new_samples: &[f32],
        mut on_frame: impl FnMut(&[f32; N]),
    ) {
        self.accumulator.extend_from_slice(new_samples);
        while self.accumulator.len() >= N {
            on_frame(self.accumulator[..N].as_array().unwrap());
            self.accumulator.drain(..self.hop_size);
        }
    }
}

struct LpcPipeline<
    const ORDER: usize = LPC_ORDER,
    const WINDOW_LEN: usize = LPC_WINDOW_LEN,
    const AUTOCORR_LEN: usize = { LPC_ORDER + 1 },
> {
    filter_coeffs: FilterCoefficients<ORDER>,
    window: [f32; WINDOW_LEN],
    autocorrelation: [f32; AUTOCORR_LEN],
}

impl<const ORDER: usize, const WINDOW_LEN: usize, const AUTOCORR_LEN: usize> Default
    for LpcPipeline<ORDER, WINDOW_LEN, AUTOCORR_LEN>
{
    fn default() -> Self {
        LpcPipeline {
            filter_coeffs: FilterCoefficients::default(),
            window: [0.; WINDOW_LEN],
            autocorrelation: [0.; AUTOCORR_LEN],
        }
    }
}

impl<const ORDER: usize, const WINDOW_LEN: usize, const AUTOCORR_LEN: usize>
    LpcPipeline<ORDER, WINDOW_LEN, AUTOCORR_LEN>
{
    /// runs the pipeline until the end of the levinson-durbin step, leaving the caller
    /// to extract formants (this part is now squeaky clean in other words, albeit contrived)
    ///
    /// returns a tuple with `(forward_coefficients, feedback_coefficients)`
    fn run_once(&mut self, x: &[f32; WINDOW_LEN]) -> (&[f32; ORDER], &[f32; ORDER]) {
        lpc_window_and_pre_emphasize_arr(x, &mut self.window);
        self.filter_coeffs
            .estimate(&self.window, &mut self.autocorrelation)
    }
}
