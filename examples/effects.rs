//! The effects of Chu and Tai's Figures 2 and 5, each painted as a little scene and left to dry,
//! written to `out/effects/<name>.png` and as one sheet, `out/effects.png`.
//!
//!     cargo run --release --features parallel --example effects

use std::fs::{self, File};
use std::io::BufWriter;

use moxi_ink::{Ink, Load, Palette, Paper, Params, Trim, footprint, render_trimmed, to_srgb8};

const N: usize = 160;
const SCALE: usize = 2;

struct Scene {
    name: &'static str,
    paint: fn() -> Ink,
}

fn main() -> std::io::Result<()> {
    let scenes = [
        Scene { name: "feathery", paint: feathery },
        Scene { name: "light-fringe", paint: light_fringe },
        Scene { name: "branching", paint: branching },
        Scene { name: "toes", paint: toes },
        Scene { name: "dark-rim", paint: dark_rim },
        Scene { name: "glue-wall", paint: glue_wall },
        Scene { name: "ink-breaking", paint: ink_breaking },
        Scene { name: "stroke", paint: stroke },
    ];
    fs::create_dir_all("out/effects")?;
    let side = N * SCALE;
    let cols = 4;
    let rows = scenes.len().div_ceil(cols);
    let mut sheet = vec![[255u8; 3]; side * cols * side * rows];
    for (k, scene) in scenes.iter().enumerate() {
        let started = std::time::Instant::now();
        let mut ink = (scene.paint)();
        let steps = ink.dry(20_000);
        let pixels: Vec<[u8; 3]> = render_trimmed(&ink, &Palette::default(), Trim { scale: SCALE, ..Trim::default() })
            .iter()
            .map(|c| c.map(to_srgb8))
            .collect();
        write_png(&format!("out/effects/{}.png", scene.name), side, side, &pixels)?;
        let (ox, oy) = ((k % cols) * side, (k / cols) * side);
        for y in 0..side {
            let row = (oy + y) * side * cols + ox;
            sheet[row..row + side].copy_from_slice(&pixels[y * side..(y + 1) * side]);
        }
        println!("{:>13}: dry after {steps} steps ({:.1} s)", scene.name, started.elapsed().as_secs_f32());
    }
    write_png("out/effects.png", side * cols, side * rows, &sheet)
}

fn sheet(seed: u64) -> Ink {
    Ink::new(Paper::generate(N, N, seed), Params::default())
}

/// Steps `n` times: lets ink spread before the next stroke.
fn wait(ink: &mut Ink, n: usize) {
    for _ in 0..n {
        ink.step();
    }
}

/// Diluted ink, plenty of it: streaks follow the water outward (Figure 2a).
fn feathery() -> Ink {
    let mut ink = sheet(1);
    ink.deposit(&footprint::disc(N, N, (80.0, 80.0), 16.0, 3.0), Load::ink(5.0, 0.18));
    ink
}

/// A stroke, then a second across it while the first is still wet: the wet paper takes less of
/// the second, which leaves it a pale fringe (Figure 2b).
fn light_fringe() -> Ink {
    let mut ink = sheet(2);
    ink.deposit(&footprint::stroke(N, N, &[(30.0, 60.0, 7.0), (130.0, 70.0, 7.0)], 2.0), Load::ink(4.0, 0.25));
    wait(&mut ink, 200);
    ink.deposit(&footprint::stroke(N, N, &[(80.0, 25.0, 8.0), (85.0, 135.0, 8.0)], 2.0), Load::ink(4.0, 0.5));
    ink
}

/// Alum sprinkled on the paper blocks the water, which branches around it (Figure 2c).
fn branching() -> Ink {
    let mut paper = Paper::generate(N, N, 3);
    paper.sprinkle_alum(420, 1.5, 1.0, 4);
    let mut ink = Ink::new(paper, Params::default());
    ink.deposit(&footprint::disc(N, N, (80.0, 80.0), 18.0, 3.0), Load::ink(5.0, 0.3));
    ink
}

/// Concentrated, gluey ink: its front is pinned along the paper's disorder and grows toes
/// (Figure 2d).
fn toes() -> Ink {
    let mut ink = sheet(5);
    let load = Load { glue: 0.25, ..Load::ink(5.0, 0.9) };
    ink.deposit(&footprint::stroke(N, N, &[(35.0, 80.0, 9.0), (125.0, 80.0, 9.0)], 2.0), load);
    ink
}

/// Water lost at the pinned rim draws the ink outward as it dries (Figure 2e).
fn dark_rim() -> Ink {
    let mut ink = sheet(6);
    ink.params.edge_evaporation = 3e-3;
    ink.deposit(&footprint::disc(N, N, (80.0, 80.0), 20.0, 2.0), Load::ink(3.0, 0.3));
    ink
}

/// A ring of glue water first; ink painted inside spreads up to it and is held (Figure 1).
fn glue_wall() -> Ink {
    let mut ink = sheet(7);
    let ring: Vec<f32> = footprint::disc(N, N, (80.0, 80.0), 34.0, 2.0)
        .iter()
        .zip(footprint::disc(N, N, (80.0, 80.0), 27.0, 2.0))
        .map(|(outer, inner)| (outer - inner).max(0.0))
        .collect();
    ink.deposit(&ring, Load { water: 3.0, pigment: [0.0; 3], glue: 1.0 });
    wait(&mut ink, 100);
    ink.deposit(&footprint::disc(N, N, (80.0, 80.0), 13.0, 3.0), Load::ink(5.0, 0.4));
    ink
}

/// Ink-breaking: a dark stroke and, while it is wet, a pale wash over it; the two break into
/// each other.
fn ink_breaking() -> Ink {
    let mut ink = sheet(8);
    ink.deposit(&footprint::stroke(N, N, &[(40.0, 50.0, 8.0), (120.0, 110.0, 8.0)], 2.0), Load::ink(3.0, 0.9));
    wait(&mut ink, 60);
    ink.deposit(&footprint::stroke(N, N, &[(40.0, 110.0, 14.0), (120.0, 50.0, 14.0)], 3.0), Load::ink(5.0, 0.08));
    ink
}

/// A brush stroke that presses down and lifts off, in indigo.
fn stroke() -> Ink {
    let mut ink = sheet(9);
    let points: Vec<(f32, f32, f32)> = (0..=40)
        .map(|k| {
            let t = k as f32 / 40.0;
            (25.0 + 110.0 * t, 115.0 - 70.0 * t + 25.0 * (t * 3.0).sin(), 2.0 + 9.0 * (t * std::f32::consts::PI).sin())
        })
        .collect();
    ink.deposit(&footprint::stroke(N, N, &points, 1.5), Load { pigment: [0.0, 0.6, 0.0], ..Load::ink(4.0, 0.0) });
    ink
}

fn write_png(path: &str, w: usize, h: usize, pixels: &[[u8; 3]]) -> std::io::Result<()> {
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path)?), w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder
        .write_header()
        .map_err(std::io::Error::other)?
        .write_image_data(&pixels.concat())
        .map_err(std::io::Error::other)
}
