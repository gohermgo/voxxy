use core::sync::atomic::AtomicBool;
use core::sync::atomic::Ordering::*;

use std::sync::Arc;
use std::thread;

use crossbeam::channel::{Receiver, Sender};

use crate::{LPC_ORDER, LPC_WINDOW_LEN};

pub struct LpcThread {
    stop_flag: Arc<AtomicBool>,
    inner_thread_handle: Option<thread::JoinHandle<anyhow::Result<()>>>,
}

impl Drop for LpcThread {
    fn drop(&mut self) {
        self.stop_flag.store(true, Release);
        if let Some(handle) = self.inner_thread_handle.take() {
            if let Err(e) = handle.join() {
                tracing::error!("{e:?}")
            };
        } else {
            tracing::warn!("lpc thread handle missing...");
        }
    }
}

impl LpcThread {
    pub fn new(
        sample_rate: u32,
        resampled_rx: Receiver<Vec<f32>>,
        formant_tx: Sender<[f32; 4]>,
    ) -> LpcThread {
        let stop_flag = Arc::new(AtomicBool::new(false));

        let mut pipeline = LpcPipeline::<LPC_ORDER, LPC_WINDOW_LEN, { LPC_ORDER + 1 }>::default();
        let mut formants = vec![0.; LPC_ORDER / 2].into_boxed_slice();

        let mut lpc_framer = LpcFramer {
            accumulator: Vec::default(),
            window_size: LPC_WINDOW_LEN,
            hop_size: 67,
        };

        let thread_handle = thread::spawn({
            let stop = Arc::clone(&stop_flag);
            move || -> anyhow::Result<()> {
                loop {
                    if stop.load(Acquire) {
                        tracing::debug!("lpc thread stopping");
                        break Ok(());
                    }

                    // blocks until data arrives OR timeout, no spin
                    match resampled_rx.recv_timeout(std::time::Duration::from_millis(5)) {
                        Ok(new_samples) => {
                            lpc_framer.push_and_maybe_analyze_arr(&new_samples, |x| {
                                if is_voiced(x, 0.15) {
                                    let (b, _a) = pipeline.run_once(x);
                                    lpc_extract_formants(b, &mut formants, sample_rate, 400.0);
                                }
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

        LpcThread {
            stop_flag,
            inner_thread_handle: Some(thread_handle),
        }
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
