//! Phoneme-targeted practice overlay.
//! Lives entirely at the app layer — lpc_extract_formants and assign_formants
//! stay untouched. This module only reads formant history/state, never writes it.

use egui::Color32;
use egui_plot::{HLine, Line, PlotPoints, PlotUi, Polygon};

// ---------------------------------------------------------------------------
// fork one: const table (femme-typical defaults, adjust freely — these are
// ballpark Peterson/Barney-adjacent numbers nudged upward, NOT gospel)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phoneme {
    I,  // beet
    E,  // bait
    Ae, // bat
    A,  // bot
    O,  // boat
    U,  // boot
}

impl Phoneme {
    pub const ALL: [Phoneme; 6] = [
        Phoneme::I,
        Phoneme::E,
        Phoneme::Ae,
        Phoneme::A,
        Phoneme::O,
        Phoneme::U,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Phoneme::I => "/i/  (beet)",
            Phoneme::E => "/e/  (bait)",
            Phoneme::Ae => "/æ/  (bat)",
            Phoneme::A => "/ɑ/  (bot)",
            Phoneme::O => "/o/  (boat)",
            Phoneme::U => "/u/  (boot)",
        }
    }
}

/// (mean_hz, std_hz) per formant slot.
#[derive(Debug, Clone, Copy)]
pub struct FormantTarget {
    pub f1: (f32, f32),
    pub f2: (f32, f32),
    pub f3: (f32, f32),
    pub f4: (f32, f32),
}

pub const PHONEME_TARGETS: &[(Phoneme, FormantTarget)] = &[
    (
        Phoneme::I,
        FormantTarget {
            f1: (310.0, 30.0),
            f2: (2790.0, 150.0),
            f3: (3310.0, 150.0),
            f4: (4200.0, 200.0),
        },
    ),
    (
        Phoneme::E,
        FormantTarget {
            f1: (430.0, 35.0),
            f2: (2480.0, 140.0),
            f3: (3200.0, 150.0),
            f4: (4100.0, 200.0),
        },
    ),
    (
        Phoneme::Ae,
        FormantTarget {
            f1: (750.0, 45.0),
            f2: (2100.0, 140.0),
            f3: (3000.0, 150.0),
            f4: (3900.0, 200.0),
        },
    ),
    (
        Phoneme::A,
        FormantTarget {
            f1: (780.0, 45.0),
            f2: (1350.0, 120.0),
            f3: (2900.0, 150.0),
            f4: (3800.0, 200.0),
        },
    ),
    (
        Phoneme::O,
        FormantTarget {
            f1: (450.0, 35.0),
            f2: (900.0, 100.0),
            f3: (2800.0, 150.0),
            f4: (3700.0, 200.0),
        },
    ),
    (
        Phoneme::U,
        FormantTarget {
            f1: (330.0, 30.0),
            f2: (850.0, 100.0),
            f3: (2700.0, 150.0),
            f4: (3600.0, 200.0),
        },
    ),
];

pub fn target_for(phoneme: Phoneme) -> FormantTarget {
    // linear scan over ~6 entries, cheaper than a hash for this N
    PHONEME_TARGETS
        .iter()
        .find(|(p, _)| *p == phoneme)
        .map(|(_, t)| *t)
        .expect("Phoneme::ALL and PHONEME_TARGETS must stay in sync")
}

// ---------------------------------------------------------------------------
// fork two: horizontal band rendering (x-axis is time/frame index, so the
// target is a lane, not a region-in-formant-space)
// ---------------------------------------------------------------------------

/// Draws one shaded lane + mean line + dashed sigma lines for a single
/// formant's (mean, std). Spans the plot's *current* visible x-range so it
/// always reaches edge-to-edge regardless of history length / zoom.
pub fn draw_formant_band(
    formant_band_name: &str,
    plot_ui: &mut PlotUi,
    mean_std: (f32, f32),
    base_color: Color32,
    (x_min, x_max): (f64, f64),
) {
    let (mean, std) = mean_std;

    let fill = base_color.gamma_multiply(0.12);
    let line_color = base_color.gamma_multiply(0.9);

    // shaded band between mean - std and mean + std
    let band_points: PlotPoints = vec![
        [x_min, (mean - std) as f64],
        [x_max, (mean - std) as f64],
        [x_max, (mean + std) as f64],
        [x_min, (mean + std) as f64],
    ]
    .into();
    plot_ui.polygon(
        Polygon::new(format!("{formant_band_name}_polygon"), band_points)
            .fill_color(fill)
            .stroke(egui::Stroke::NONE),
    );

    // solid mean line
    plot_ui.hline(
        HLine::new(format!("{formant_band_name}_line_mean"), mean as f64)
            .color(line_color)
            .width(2.0),
    );

    // dashed ±σ lines

    plot_ui.line(dashed_hline(
        format!("{formant_band_name}_line_min"),
        x_min,
        x_max,
        (mean - std) as f64,
        line_color,
    ));

    plot_ui.line(dashed_hline(
        format!("{formant_band_name}_line_max"),
        x_min,
        x_max,
        (mean + std) as f64,
        line_color,
    ));
}

/// egui_plot has no built-in dashed HLine, so we synthesize one as a
/// segmented Line — cheap, recomputed only on selection/zoom change.
fn dashed_hline(
    name: impl Into<String>,
    x_min: f64,
    x_max: f64,
    y: f64,
    color: Color32,
) -> Line<'static> {
    const DASH_LEN: f64 = 8.0;
    const GAP_LEN: f64 = 6.0;
    let mut pts = Vec::new();
    let mut x = x_min;
    while x < x_max {
        let seg_end = (x + DASH_LEN).min(x_max);
        pts.push([x, y]);
        pts.push([seg_end, y]);
        pts.push([f64::NAN, f64::NAN]); // break the line between dashes
        x = seg_end + GAP_LEN;
    }
    Line::new(name, PlotPoints::from(pts))
        .color(color)
        .width(1.5)
}

/// Convenience: draw all four formant bands for a target in one go.
/// Colors are just suggestions — wire up to your existing F1..F4 palette.
pub fn draw_target_overlay(
    plot_ui: &mut PlotUi,
    target: FormantTarget,
    base_colors: &[Color32; 4],
    x_range: (f64, f64),
) {
    // draw_formant_band("f1", plot_ui, target.f1, Color32::from_rgb(255, 20, 147)); // pink
    // draw_formant_band("f2", plot_ui, target.f2, Color32::from_rgb(0, 191, 255)); // sky blue
    // draw_formant_band("f3", plot_ui, target.f3, Color32::from_rgb(50, 205, 50)); // green
    // draw_formant_band("f4", plot_ui, target.f4, Color32::from_rgb(218, 112, 214)); // orchid
    draw_formant_band("f1", plot_ui, target.f1, base_colors[0], x_range);
    draw_formant_band("f2", plot_ui, target.f2, base_colors[1], x_range);
    draw_formant_band("f3", plot_ui, target.f3, base_colors[2], x_range);
    draw_formant_band("f4", plot_ui, target.f4, base_colors[3], x_range);
}

// ---------------------------------------------------------------------------
// fork three: selector, decoupled from the widget that drives it
// ---------------------------------------------------------------------------

/// Owns nothing but the currently-selected phoneme (or none). Any widget —
/// combobox, tabs, chips, whatever you swap in later — just calls `set()`.
/// Render code only ever reads `active`.
#[derive(Default)]
pub struct PhonemeSelection {
    pub active: Option<Phoneme>,
}

impl PhonemeSelection {
    pub fn set(&mut self, phoneme: Option<Phoneme>) {
        self.active = phoneme;
    }
}

/// v1 picker widget: plain combobox. Swap this function out later without
/// touching PhonemeSelection or the render path at all.
pub fn phoneme_combobox(ui: &mut egui::Ui, selection: &mut PhonemeSelection) {
    let current_label = selection.active.map(|ph| ph.label()).unwrap_or("None");

    egui::ComboBox::from_label("Practice target")
        .selected_text(current_label)
        .show_ui(ui, |ui| {
            if ui
                .selectable_label(selection.active.is_none(), "None")
                .clicked()
            {
                selection.set(None);
            }
            for phoneme in Phoneme::ALL {
                let selected = selection.active == Some(phoneme);
                if ui.selectable_label(selected, phoneme.label()).clicked() {
                    selection.set(Some(phoneme));
                }
            }
        });
}

// ---------------------------------------------------------------------------
// wiring sketch — call from wherever FormantPlot currently builds its Plot::show
// ---------------------------------------------------------------------------
//
// phoneme_combobox(ui, &mut self.phoneme_selection);
//
// egui_plot::Plot::new("formant_plot").show(ui, |plot_ui| {
//     // ... existing scatter/history rendering ...
//     if let Some(phoneme) = self.phoneme_selection.active {
//         draw_target_overlay(plot_ui, target_for(phoneme));
//     }
// });
