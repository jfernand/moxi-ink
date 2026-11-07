//! Turning ink into colours (Chu and Tai §6).
//!
//! Ink absorbs light: each pigment has an absorption per colour channel, and paper under an amount
//! `x` of it reflects `paper · e^(−absorption · x)`. [`render`] shows the ink at simulation
//! resolution; [`render_trimmed`] upscales it with the paper's boundary trimming (§6.1), which
//! keeps edges sharp and slightly rough instead of blurred and blocky.

use crate::ink::{Ink, lerp};

/// How each of the three pigments absorbs light, per linear RGB channel, per unit of ink.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Palette {
    /// Absorption of pigments 1, 2 and 3.
    pub absorption: [[f32; 3]; 3],
    /// The paper's colour, linear RGB.
    pub paper: [f32; 3],
}

impl Default for Palette {
    /// Sumi (a warm black), indigo and vermilion on warm white paper.
    fn default() -> Self {
        Self { absorption: [[2.6, 2.7, 3.0], [2.4, 1.4, 0.5], [0.3, 2.2, 2.6]], paper: [0.93, 0.9, 0.82] }
    }
}

impl Palette {
    /// The colour of paper of `grain` under `ink` (`[pigment 1, 2, 3, glue]` amounts).
    pub fn colour(&self, ink: [f32; 4], grain: f32) -> [f32; 3] {
        // Denser fibres are a touch darker, as Xuan paper is against the light.
        let paper = self.paper.map(|c| c * (0.95 + 0.05 * (1.0 - grain)));
        std::array::from_fn(|ch| {
            let absorbed: f32 = (0..3).map(|p| self.absorption[p][ch] * ink[p]).sum();
            paper[ch] * libm::expf(-absorbed)
        })
    }
}

/// One linear RGB colour per site.
pub fn render(ink: &Ink, palette: &Palette) -> Vec<[f32; 3]> {
    ink.visible_ink().iter().zip(&ink.paper.grain).map(|(&v, &g)| palette.colour(v, g)).collect()
}

/// Options for [`render_trimmed`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trim {
    /// Output pixels per site (the paper's d-scale).
    pub scale: usize,
    /// r_scale: how much the rougher texture roughens the edge (0 for a smooth edge).
    pub roughness: f32,
    /// r_bias, added to the rougher texture before scaling.
    pub bias: f32,
}

impl Default for Trim {
    fn default() -> Self {
        Self { scale: 3, roughness: 0.25, bias: -0.5 }
    }
}

/// The ink upscaled by `trim.scale` (`width · scale × height · scale` colours, row-major), with
/// interpolated pixels that fall outside the stained area trimmed back to paper (equation 11).
pub fn render_trimmed(ink: &Ink, palette: &Palette, trim: Trim) -> Vec<[f32; 3]> {
    let (w, h, s) = (ink.width, ink.height, trim.scale.max(1));
    let visible = ink.visible_ink();
    // φ per site: the most water it held, shrunk to zero at the stain's own edge so the trimmed
    // shape sits just inside the stained sites; negative where ink never reached.
    let stained = |x: usize, y: usize| ink.stain[y * w + x] > 0.0;
    let phi: Vec<f32> = (0..w * h)
        .map(|c| {
            let (x, y) = (c % w, c / w);
            if !stained(x, y) {
                return -0.2;
            }
            let at_edge = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)].iter().any(|&(dx, dy)| {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h && !stained(nx as usize, ny as usize)
            });
            if at_edge { 0.0 } else { ink.stain[c].min(1.0) }
        })
        .collect();
    let sample = |field: &dyn Fn(usize) -> f32, x: f32, y: f32| {
        let (x, y) = ((x - 0.5).clamp(0.0, (w - 1) as f32), (y - 0.5).clamp(0.0, (h - 1) as f32));
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let top = lerp(field(y0 * w + x0), field(y0 * w + x1), fx);
        let bottom = lerp(field(y1 * w + x0), field(y1 * w + x1), fx);
        lerp(top, bottom, fy)
    };
    let mut out = Vec::with_capacity(w * h * s * s);
    for py in 0..h * s {
        for px in 0..w * s {
            let (x, y) = ((px as f32 + 0.5) / s as f32, (py as f32 + 0.5) / s as f32);
            let grain = sample(&|c| ink.paper.grain[c], x, y);
            let r = sample(&|c| ink.paper.rougher[c], x, y);
            let edge = sample(&|c| phi[c], x, y) + trim.roughness * (r + trim.bias);
            let v = if edge < 0.0 { [0.0; 4] } else { std::array::from_fn(|k| sample(&|c| visible[c][k], x, y)) };
            out.push(palette.colour(v, grain));
        }
    }
    out
}

/// Linear reflectance to an 8-bit sRGB channel.
pub fn to_srgb8(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let v = if c <= 0.003_130_8 { 12.92 * c } else { 1.055 * libm::powf(c, 1.0 / 2.4) - 0.055 };
    (v * 255.0).round() as u8
}
