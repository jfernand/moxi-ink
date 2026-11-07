//! The GPU passes agree with the CPU ones. Skipped where no GPU adapter is available.
#![cfg(feature = "wgpu")]

use moxi_ink::{Ink, Load, Paper, Params, footprint, gpu};

#[test]
fn gpu_matches_cpu() {
    let Some((device, queue)) = gpu::default_device() else {
        eprintln!("no GPU adapter: skipping");
        return;
    };
    let n = 64;
    let mut paper = Paper::generate(n, n, 11);
    paper.sprinkle_alum(20, 1.5, 0.8, 2);
    let mut cpu = Ink::new(paper, Params::default());
    let mut gpu = gpu::GpuInk::new(&device, &queue, &cpu);
    let dab = footprint::disc(n, n, (30.0, 34.0), 7.0, 2.0);
    let load = Load { pigment: [0.4, 0.1, 0.0], ..Load::ink(4.0, 0.0) };
    cpu.deposit(&dab, load);
    gpu.deposit(&dab, load);
    let steps = 60;
    for _ in 0..steps {
        cpu.step();
    }
    gpu.step(steps);
    let g = gpu.read();
    let worst = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let flat = |v: &[[f32; 4]]| v.iter().flatten().copied().collect::<Vec<_>>();
    let water = worst(&cpu.water, &g.water);
    let surface = worst(&cpu.surface_water, &g.surface_water);
    let ink = worst(&flat(&cpu.flow_ink), &flat(&g.flow_ink));
    let fixed = worst(&flat(&cpu.fixed_ink), &flat(&g.fixed_ink));
    let pinned_differ = cpu.block.iter().zip(&g.block).filter(|(a, b)| a.is_infinite() != b.is_infinite()).count();
    println!(
        "after {steps} steps: water {water:e}, surface {surface:e}, ink {ink:e}, fixed {fixed:e}, pinning differs at {pinned_differ} sites"
    );
    assert!(water < 1e-4 && surface < 1e-4 && ink < 1e-4 && fixed < 1e-4, "GPU and CPU disagree");
    assert!(pinned_differ <= 2, "pinning differs at {pinned_differ} sites");
}
