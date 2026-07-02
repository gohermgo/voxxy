use std::collections::VecDeque;

use cpal::traits::HostTrait;
use crossbeam::channel::Receiver;
use heapless::spsc::Queue;
use tracing::level_filters::STATIC_MAX_LEVEL;

struct ResamplerCfg {
    input_rate: cpal::SampleRate,
    target_rate: cpal::SampleRate,
    chunk_size: usize,
}

mod audio_thread;
mod worker_thread;

mod lpc;

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

    let (formant_tx, formant_rx) = crossbeam::channel::bounded::<[f32; 4]>(64);

    let lpc_thread = lpc::LpcThread::new(sample_rate, resampled_rx, formant_tx);

    // here is our start point!

    audio_thread.play()?;

    let history_size = 600;
    eframe::run_native(
        "voxxy",
        eframe::NativeOptions::default(),
        Box::new(|_cc| {
            Ok(Box::new(Voxxy {
                formant_plot: FormantPlot {
                    rx: formant_rx,
                    history: VecDeque::with_capacity(history_size),
                    history_threshold: history_size,
                    last_valid: [0.; 4],
                },
            }))
        }),
    )?;

    tracing::debug!("done!");

    drop(lpc_thread);
    drop(audio_thread);
    drop(worker_thread);

    Ok(())
}

use egui_plot::{Plot, PlotPoints, Points};

// typical formant ranges in Hz, used ONLY as a cold-start / post-silence seed —
// once continuity tracking has a confirmed anchor these bounds get ignored.
// leanin toward femme-typical ranges since that's the target use case
const FORMANT_SEED_RANGES: [(f32, f32); 4] = [
    (250.0, 950.0),   // F1
    (950.0, 2500.0),  // F2
    (2500.0, 3500.0), // F3
    (3500.0, 4500.0), // F4
];

/// re-maps a raw sorted-ascending, zero-padded formant array (as sent by the
/// lpc thread) onto stable F1..F4 slots. if `last_confirmed` has an established
/// value for a slot, we greedily match the closest raw candidate to it (continuity).
/// any candidate that isn't claimed by continuity matching (typically only relevant
/// right after silence, when last_confirmed is all zero) gets seeded via static
/// range membership instead.
fn assign_formants(raw: &[f32; 4], last_confirmed: &[f32; 4]) -> [f32; 4] {
    let candidates: Vec<f32> = raw.iter().copied().filter(|&v| v > 0.0).collect();
    let mut assigned = [0.0f32; 4];
    let mut claimed = [false; 4]; // candidates already placed, by index into `candidates`

    // --- pass 1: continuity, greedy nearest-neighbor ---
    // build (slot, candidate_idx, distance) for every established slot x unclaimed candidate,
    // sort by distance ascending, claim greedily. small N (<=4x4), brute force is plenty.
    let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
    for (slot, &conf) in last_confirmed.iter().enumerate() {
        if conf <= 0.0 {
            continue; // no established anchor for this slot yet
        }
        for (ci, &cand) in candidates.iter().enumerate() {
            pairs.push((slot, ci, (cand - conf).abs()));
        }
    }
    pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    let mut slot_taken = [false; 4];
    for (slot, ci, dist) in pairs {
        if slot_taken[slot] || claimed[ci] {
            continue;
        }
        // sanity ceiling - a jump this big ain't continuity, it's a different formant
        // entirely, let pass 2 handle it via static seeding instead
        const MAX_CONTINUITY_JUMP_HZ: f32 = 400.0;
        if dist > MAX_CONTINUITY_JUMP_HZ {
            continue;
        }
        assigned[slot] = candidates[ci];
        slot_taken[slot] = true;
        claimed[ci] = true;
    }

    // --- pass 2: static-range seeding for anything continuity didn't claim ---
    for (ci, &cand) in candidates.iter().enumerate() {
        if claimed[ci] {
            continue;
        }
        for (slot, &(lo, hi)) in FORMANT_SEED_RANGES.iter().enumerate() {
            if !slot_taken[slot] && cand >= lo && cand < hi {
                assigned[slot] = cand;
                slot_taken[slot] = true;
                claimed[ci] = true;
                break;
            }
        }
    }

    assigned
}

struct Voxxy {
    formant_plot: FormantPlot,
}

impl eframe::App for Voxxy {
    fn logic(&mut self, _: &egui::Context, _: &mut eframe::Frame) {
        self.formant_plot.update();
    }
    fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ui, |ui| {
            // bottom legend FIRST so plot takes remaining space
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.horizontal(|ui| {
                    #[allow(clippy::needless_range_loop)]
                    for fi in 0..4 {
                        let hz = self.formant_plot.last_valid[fi];

                        let text = format!("F{}  {:.0} Hz", fi + 1, hz);

                        ui.label(
                            egui::RichText::new(text)
                                .color(FORMANT_COLORS[fi])
                                .strong()
                                .size(18.0),
                        );

                        ui.add_space(24.0);
                    }
                });

                // plot gets everything above the legend
                self.formant_plot.show(ui);
            });
        });

        ui.request_repaint();
    }
}

struct FormantPlot {
    rx: Receiver<[f32; 4]>,
    history: VecDeque<[f32; 4]>,
    history_threshold: usize,
    last_valid: [f32; 4],
}

const FORMANT_COLORS: [egui::Color32; 4] = [
    egui::Color32::from_rgb(220, 80, 80),  // F1 red
    egui::Color32::from_rgb(80, 140, 220), // F2 blue
    egui::Color32::from_rgb(140, 200, 80), // F3 green
    egui::Color32::from_rgb(200, 80, 200), // F4 pink
];

impl FormantPlot {
    /// drains all new values from the stored receiver
    fn update(&mut self) {
        let mut received_and_valid = 0;
        while let Ok(raw_formant_frame) = self.rx.try_recv() {
            let formants = assign_formants(&raw_formant_frame, &self.last_valid);

            // F1 must be real, otherwise we ignore this frame
            if formants[0] > 0.0 {
                received_and_valid += 1;
                self.last_valid = formants;

                if self.history.len() >= self.history_threshold {
                    self.history.pop_front();
                }

                self.history.push_back(self.last_valid);
            }
        }
        if received_and_valid != 0 {
            println!("received {received_and_valid} formant-frames this tick");
        }
    }
    fn show(&self, ui: &mut egui::Ui) -> egui_plot::PlotResponse<()> {
        Plot::new("formants")
            .include_y(0.0)
            .include_y(6000.0)
            .default_y_bounds(0.0, 6000.0)
            .show(ui, |plot_ui| {
                // here we iterate over each formant per index... strange...
                #[expect(clippy::needless_range_loop)]
                for fi in 0..4 {
                    render_formant_by_index(plot_ui, &self.history, fi, 3.0, FORMANT_COLORS[fi]);

                    // bold label floating on the line itself
                    if self.last_valid[fi] > 0.0 {
                        plot_ui.text(egui_plot::Text::new(
                            format!("label-F{}", fi + 1),
                            egui_plot::PlotPoint::new(295.0, self.last_valid[fi] as f64),
                            egui::RichText::new(format!("F{}", fi + 1))
                                .color(FORMANT_COLORS[fi])
                                .strong()
                                .size(15.0),
                        ));
                    }
                }
            })
    }
}

fn render_formant_by_index(
    plot_ui: &mut egui_plot::PlotUi,
    formant_history: &std::collections::VecDeque<[f32; 4]>,
    formant_index: usize,
    point_radius: f32,
    point_color: egui::Color32,
) {
    // this represents all points for this specific formant (i.e. F0, F1, F2, or F3)
    // present in the history.
    let points: PlotPoints = formant_history
        .iter()
        .enumerate()
        .filter_map(|(frame_index, formants)| {
            let current_formant = formants[formant_index];
            if current_formant > 0.0 {
                // we use the frame-index as the x-coordinate for the formant
                Some([frame_index as f64, current_formant as f64])
            } else {
                None
            }
        })
        .collect();

    plot_ui.points(
        Points::new(format!("F{}", formant_index + 1), points)
            .radius(point_radius)
            .color(point_color),
    )
}

// fn render_plot(ui: &mut eframe::egui::Ui) {
//     const COLORS: [egui::Color32; 4] = [
//         egui::Color32::from_rgb(220, 80, 80),  // F1 red
//         egui::Color32::from_rgb(80, 140, 220), // F2 blue
//         egui::Color32::from_rgb(140, 200, 80), // F3 green
//         egui::Color32::from_rgb(200, 80, 200), // F4 pink
//     ];
//     egui::CentralPanel::default().show(ui, |ui| {
//         // bottom legend FIRST so plot takes remaining space
//         ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
//             ui.horizontal(|ui| {
//                 #[allow(clippy::needless_range_loop)]
//                 for fi in 0..4 {
//                     let hz = self.last_valid[fi];
//                     let text = if self.is_voiced && self.smoothed[fi] > 0.0 {
//                         format!("F{}  {:.0} Hz", fi + 1, hz)
//                     } else {
//                         format!("F{}  ---", fi + 1)
//                     };
//                     ui.label(
//                         egui::RichText::new(text)
//                             .color(COLORS[fi])
//                             .strong()
//                             .size(18.0),
//                     );
//                     ui.add_space(24.0);
//                 }
//             });

//             // plot gets everything above the legend
//             Plot::new("formants")
//                 .include_y(0.0)
//                 .include_y(6000.0)
//                 .default_y_bounds(0.0, 6000.0)
//                 .show(ui, |plot_ui| {
//                     for fi in 0..4 {
//                         let pts: PlotPoints = self
//                             .history
//                             .iter()
//                             .enumerate()
//                             .filter(|(_, f)| f[fi] > 0.0)
//                             .map(|(i, f)| [i as f64, f[fi] as f64])
//                             .collect();

//                         plot_ui.points(
//                             Points::new(format!("F{}", fi + 1), pts)
//                                 .radius(3.0)
//                                 .color(COLORS[fi]),
//                         );

//                         // bold label floating on the line itself
//                         if self.last_valid[fi] > 0.0 {
//                             plot_ui.text(egui_plot::Text::new(
//                                 format!("label-F{}", fi + 1),
//                                 egui_plot::PlotPoint::new(295.0, self.last_valid[fi] as f64),
//                                 egui::RichText::new(format!("F{}", fi + 1))
//                                     .color(COLORS[fi])
//                                     .strong()
//                                     .size(15.0),
//                             ));
//                         }
//                     }
//                 });
//         });
//     });
//     ui.request_repaint();
// }
