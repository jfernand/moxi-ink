//! Paints the same dab of ink while varying one setting at a time, and writes one strip per
//! setting to `out/parameters/<name>.png`. Each strip includes the default value.
//!
//!     cargo run --release --features wgpu --example parameters      # on the GPU
//!     cargo run --release --features parallel --example parameters  # on the CPU (slower)

use std::fs::{self, File};
use std::io::BufWriter;

use moxi_ink::{Ink, Load, Palette, Paper, Params, Trim, footprint, render_trimmed, to_srgb8};

const N: usize = 128;
const STEPS: usize = 6000;
const SCALE: usize = 2;
const GAP: usize = 12;

type Vary = fn(&mut Params, &mut Load, f32);

fn main() -> std::io::Result<()> {
    let d = Params::default();
    let load = Load::ink(5.0, 0.25);
    let strips: [(&str, [f32; 5], Vary); 11] = [
        ("omega", [0.2, 0.35, d.omega, 1.0, 1.5], |p, _, x| p.omega = x),
        ("advection", [0.05, 0.15, d.advection, 0.5, 0.8], |p, _, x| p.advection = x),
        ("evaporation", [0.00005, 0.0001, d.evaporation, 0.0005, 0.001], |p, _, x| p.evaporation = x),
        ("edge_evaporation", [0.0, 0.0001, d.edge_evaporation, 0.001, 0.003], |p, _, x| p.edge_evaporation = x),
        ("block_grain", [0.2, 0.5, d.block_grain, 0.95, 1.0], |p, _, x| p.block_grain = x),
        ("pin_base", [0.03, 0.06, d.pin_base, 0.15, 0.25], |p, _, x| p.pin_base = x),
        ("pin_disorder", [0.0, 0.2, d.pin_disorder, 0.6, 0.8], |p, _, x| p.pin_disorder = x),
        ("hindrance", [0.0, 0.15, d.hindrance, 0.5, 0.7], |p, _, x| p.hindrance = x),
        ("hindrance_still", [0.5, 0.7, d.hindrance_still, 0.95, 1.0], |p, _, x| p.hindrance_still = x),
        ("glue", [0.0, 0.0125, load.glue, 0.15, 0.4], |_, l, x| l.glue = x),
        ("strength", [0.05, 0.12, load.pigment[0], 0.5, 1.0], |_, l, x| l.pigment[0] = x),
    ];
    #[cfg(feature = "wgpu")]
    let device = moxi_ink::gpu::default_device();
    fs::create_dir_all("out/parameters")?;
    for (name, values, vary) in strips {
        let swatches: Vec<Vec<[f32; 3]>> = values
            .iter()
            .map(|&x| {
                let (mut params, mut load) = (d, load);
                vary(&mut params, &mut load, x);
                let mut ink = Ink::new(Paper::generate(N, N, 7), params);
                let dab = footprint::disc(N, N, (64.0, 64.0), 12.0, 2.0);
                #[cfg(feature = "wgpu")]
                if let Some((device, queue)) = &device {
                    let mut gpu = moxi_ink::gpu::GpuInk::new(device, queue, &ink);
                    gpu.deposit(&dab, load);
                    gpu.step(STEPS);
                    return render(&gpu.read());
                }
                ink.deposit(&dab, load);
                ink.dry(STEPS);
                render(&ink)
            })
            .collect();
        let path = format!("out/parameters/{name}.png");
        write_strip(&path, &swatches)?;
        println!("{path}: {}", values.map(|v| format!("{v}")).join(" · "));
    }
    Ok(())
}

fn render(ink: &Ink) -> Vec<[f32; 3]> {
    render_trimmed(ink, &Palette::default(), Trim { scale: SCALE, ..Trim::default() })
}

fn write_strip(path: &str, swatches: &[Vec<[f32; 3]>]) -> std::io::Result<()> {
    let side = N * SCALE;
    let (w, h) = (swatches.len() * side + (swatches.len() + 1) * GAP, side + 2 * GAP);
    let paper = Palette::default().paper.map(|c| to_srgb8(c * 0.975));
    let mut image = vec![paper; w * h];
    for (k, s) in swatches.iter().enumerate() {
        let x0 = GAP + k * (side + GAP);
        for y in 0..side {
            for x in 0..side {
                image[(GAP + y) * w + x0 + x] = s[y * side + x].map(to_srgb8);
            }
        }
    }
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path)?), w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder
        .write_header()
        .map_err(std::io::Error::other)?
        .write_image_data(&image.concat())
        .map_err(std::io::Error::other)
}
