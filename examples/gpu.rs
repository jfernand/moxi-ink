//! The same painting on the CPU and the GPU, timed and compared, then a large one on the GPU
//! alone: `out/gpu-compare.png` (CPU left, GPU right) and `out/gpu-large.png`.
//!
//!     cargo run --release --features wgpu,parallel --example gpu

use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

use moxi_ink::{Ink, Load, Palette, Paper, Params, Trim, footprint, gpu, render_trimmed, to_srgb8};

fn main() -> std::io::Result<()> {
    let (device, queue) = gpu::default_device().expect("no GPU adapter");

    // The same scene both ways.
    let n = 160;
    let steps = 4000;
    let mut cpu = Ink::new(Paper::generate(n, n, 1), Params::default());
    let mut on_gpu = gpu::GpuInk::new(&device, &queue, &cpu);
    let dab = footprint::disc(n, n, (80.0, 80.0), 16.0, 3.0);
    let load = Load::ink(5.0, 0.18);
    cpu.deposit(&dab, load);
    on_gpu.deposit(&dab, load);
    let started = Instant::now();
    for _ in 0..steps {
        cpu.step();
    }
    let cpu_time = started.elapsed();
    let started = Instant::now();
    on_gpu.step(steps);
    let from_gpu = on_gpu.read();
    let gpu_time = started.elapsed();
    let worst = cpu
        .visible_ink()
        .iter()
        .zip(from_gpu.visible_ink())
        .flat_map(|(a, b)| (0..4).map(move |k| (a[k] - b[k]).abs()))
        .fold(0.0f32, f32::max);
    println!(
        "{n}×{n}, {steps} steps: CPU {cpu_time:.2?}, GPU {gpu_time:.2?}; largest difference in visible ink {worst:e}"
    );
    let trim = Trim { scale: 2, ..Trim::default() };
    let (a, b) =
        (render_trimmed(&cpu, &Palette::default(), trim), render_trimmed(&from_gpu, &Palette::default(), trim));
    let side = n * trim.scale;
    let pair: Vec<[u8; 3]> = (0..side)
        .flat_map(|y| {
            a[y * side..(y + 1) * side].iter().chain(&b[y * side..(y + 1) * side]).copied().collect::<Vec<_>>()
        })
        .map(|c| c.map(to_srgb8))
        .collect();
    write_png("out/gpu-compare.png", 2 * side, side, &pair)?;

    // A large sheet with a few strokes, on the GPU only.
    let n = 512;
    let mut large = gpu::GpuInk::new(&device, &queue, &Ink::new(Paper::generate(n, n, 2), Params::default()));
    let started = Instant::now();
    for (k, (y, strength)) in [(140.0, 0.9), (260.0, 0.35), (380.0, 0.12)].into_iter().enumerate() {
        let points: Vec<(f32, f32, f32)> = (0..=30)
            .map(|i| {
                let t = i as f32 / 30.0;
                (60.0 + 390.0 * t, y + 30.0 * (t * 5.0 + k as f32).sin(), 4.0 + 12.0 * (t * std::f32::consts::PI).sin())
            })
            .collect();
        large.deposit(&footprint::stroke(n, n, &points, 2.0), Load::ink(5.0, strength));
        large.step(150);
    }
    large.step(6000);
    let ink = large.read();
    println!("{n}×{n}: three strokes and 6450 steps on the GPU in {:.2?}", started.elapsed());
    let colours = render_trimmed(&ink, &Palette::default(), Trim::default());
    let side = n * Trim::default().scale;
    write_png("out/gpu-large.png", side, side, &colours.iter().map(|c| c.map(to_srgb8)).collect::<Vec<_>>())
}

fn write_png(path: &str, w: usize, h: usize, pixels: &[[u8; 3]]) -> std::io::Result<()> {
    std::fs::create_dir_all("out")?;
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path)?), w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder
        .write_header()
        .map_err(std::io::Error::other)?
        .write_image_data(&pixels.concat())
        .map_err(std::io::Error::other)
}
