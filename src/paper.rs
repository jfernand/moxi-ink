//! Procedural Xuan paper: the textures that steer the ink (Chu and Tai §5.3).
//!
//! - **Grain** G: how dense the fibres are. Dense fibres block the water (equation 6) and vary
//!   where its front gets pinned (equation 9).
//! - **Alum** A: sizing that blocks the water. Plain paper has none; [`Paper::sprinkle_alum`]
//!   adds dots of it, as artists do to make ink branch.
//! - **Pinning** P: light line segments on dark. Under gluey (concentrated) ink it replaces the
//!   grain in the pinning threshold, so the front escapes along the segments and grows toes.
//! - **Rougher** r: blurred random spots, used to roughen the edge of an upscaled rendering
//!   (§6.1).
//!
//! All four are `width × height`, row-major, in `[0, 1]`.

use crate::rng::Rng;

/// The paper's textures, one value per lattice site.
#[derive(Clone, Debug, PartialEq)]
pub struct Paper {
    /// Sites per row.
    pub width: usize,
    /// Rows.
    pub height: usize,
    /// G, fibre density.
    pub grain: Vec<f32>,
    /// A, alum (sizing).
    pub alum: Vec<f32>,
    /// P, the pinning texture.
    pub pinning: Vec<f32>,
    /// r, the edge roughener.
    pub rougher: Vec<f32>,
}

impl Paper {
    /// Paper of `width × height` sites, without alum. The same seed gives the same paper.
    pub fn generate(width: usize, height: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let n = width * height;

        // Grain: soft value noise in a few octaves, with long fibres laid over it.
        let mut grain = vec![0.0f32; n];
        for (cell, amp) in [(24.0f32, 0.15f32), (6.0, 0.15), (2.0, 0.2), (1.0, 0.2)] {
            let noise = ValueNoise::new(&mut rng, width, height, cell);
            for (c, g) in grain.iter_mut().enumerate() {
                *g += amp * noise.at((c % width) as f32, (c / width) as f32);
            }
        }
        let fibres = n / 3;
        for _ in 0..fibres {
            let (x0, y0) = (rng.unit() * width as f32, rng.unit() * height as f32);
            let angle = rng.unit() * std::f32::consts::TAU;
            let len = 8.0 + 30.0 * rng.unit();
            let lift = 0.15 + 0.25 * rng.unit();
            line(&mut grain, width, height, (x0, y0), angle, len, lift, true);
        }
        normalise(&mut grain);

        // Pinning: light, short line segments on black.
        let mut pinning = vec![0.0f32; n];
        for _ in 0..n / 25 {
            let (x0, y0) = (rng.unit() * width as f32, rng.unit() * height as f32);
            let angle = rng.unit() * std::f32::consts::TAU;
            let len = 2.0 + 7.0 * rng.unit();
            let value = 0.5 + 0.5 * rng.unit();
            line(&mut pinning, width, height, (x0, y0), angle, len, value, false);
        }

        // Rougher: random spots, blurred.
        let mut rougher: Vec<f32> = (0..n).map(|_| if rng.chance(1, 6) { rng.unit() } else { 0.0 }).collect();
        for _ in 0..2 {
            rougher = box_blur(&rougher, width, height, 1);
        }
        normalise(&mut rougher);

        Self { width, height, grain, alum: vec![0.0; n], pinning, rougher }
    }

    /// Paper from your own textures (each `width × height`, in `[0, 1]`), such as a scanned sheet
    /// for the grain.
    ///
    /// # Panics
    ///
    /// If a texture does not hold `width * height` values.
    pub fn from_textures(
        width: usize,
        height: usize,
        grain: Vec<f32>,
        alum: Vec<f32>,
        pinning: Vec<f32>,
        rougher: Vec<f32>,
    ) -> Self {
        for t in [&grain, &alum, &pinning, &rougher] {
            assert_eq!(t.len(), width * height, "textures must be width × height");
        }
        Self { width, height, grain, alum, pinning, rougher }
    }

    /// Adds `count` dots of alum of about `radius` sites and the given `strength`, so the water
    /// branches around them.
    pub fn sprinkle_alum(&mut self, count: usize, radius: f32, strength: f32, seed: u64) {
        let mut rng = Rng::new(seed);
        let (w, h) = (self.width, self.height);
        for _ in 0..count {
            let (cx, cy) = (rng.unit() * w as f32, rng.unit() * h as f32);
            let r = radius * (0.5 + rng.unit());
            let (x0, x1) = ((cx - r).floor().max(0.0) as usize, ((cx + r).ceil() as usize).min(w));
            let (y0, y1) = ((cy - r).floor().max(0.0) as usize, ((cy + r).ceil() as usize).min(h));
            for y in y0..y1 {
                for x in x0..x1 {
                    let d = (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy) / r;
                    if d < 1.0 {
                        let a = &mut self.alum[y * w + x];
                        *a = (*a + strength * (1.0 - d * d)).min(1.0);
                    }
                }
            }
        }
    }
}

/// Smooth value noise with knots every `cell` sites.
struct ValueNoise {
    cols: usize,
    knots: Vec<f32>,
    cell: f32,
}

impl ValueNoise {
    fn new(rng: &mut Rng, width: usize, height: usize, cell: f32) -> Self {
        let cols = (width as f32 / cell).ceil() as usize + 2;
        let rows = (height as f32 / cell).ceil() as usize + 2;
        Self { cols, knots: (0..cols * rows).map(|_| rng.unit()).collect(), cell }
    }

    fn at(&self, x: f32, y: f32) -> f32 {
        let (x, y) = (x / self.cell, y / self.cell);
        let (i, j) = (x.floor() as usize, y.floor() as usize);
        let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
        let (fx, fy) = (smooth(x.fract()), smooth(y.fract()));
        let k = |i: usize, j: usize| self.knots[j * self.cols + i];
        let top = k(i, j) * (1.0 - fx) + k(i + 1, j) * fx;
        let bottom = k(i, j + 1) * (1.0 - fx) + k(i + 1, j + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// Draws a line of `len` sites from `start` at `angle`: adding `value` (with a soft one-site
/// shoulder when `soft`), or raising the texture to it.
#[allow(clippy::too_many_arguments)]
fn line(t: &mut [f32], w: usize, h: usize, start: (f32, f32), angle: f32, len: f32, value: f32, soft: bool) {
    let steps = (len * 2.0) as usize;
    let (dx, dy) = (angle.cos() / 2.0, angle.sin() / 2.0);
    for s in 0..steps {
        let (x, y) = (start.0 + dx * s as f32, start.1 + dy * s as f32);
        let (xi, yi) = (x as isize, y as isize);
        let taps: &[(isize, isize, f32)] = if soft {
            &[(0, 0, 0.5), (1, 0, 0.15), (-1, 0, 0.15), (0, 1, 0.15), (0, -1, 0.15)]
        } else {
            &[(0, 0, 1.0)]
        };
        for &(ox, oy, wgt) in taps {
            let (px, py) = (xi + ox, yi + oy);
            if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                let v = &mut t[py as usize * w + px as usize];
                *v = if soft { *v + value * wgt } else { v.max(value) };
            }
        }
    }
}

fn box_blur(src: &[f32], w: usize, h: usize, r: isize) -> Vec<f32> {
    let mut out = vec![0.0; w * h];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let (mut sum, mut count) = (0.0, 0.0);
            for oy in -r..=r {
                for ox in -r..=r {
                    let (px, py) = (x + ox, y + oy);
                    if px >= 0 && py >= 0 && px < w as isize && py < h as isize {
                        sum += src[py as usize * w + px as usize];
                        count += 1.0;
                    }
                }
            }
            out[y as usize * w + x as usize] = sum / count;
        }
    }
    out
}

fn normalise(t: &mut [f32]) {
    let (lo, hi) = t.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let span = (hi - lo).max(1e-6);
    t.iter_mut().for_each(|v| *v = (*v - lo) / span);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paper_is_deterministic_and_in_range() {
        let a = Paper::generate(40, 30, 9);
        assert_eq!(a, Paper::generate(40, 30, 9));
        assert_ne!(a.grain, Paper::generate(40, 30, 10).grain);
        for t in [&a.grain, &a.alum, &a.pinning, &a.rougher] {
            assert_eq!(t.len(), 40 * 30);
            assert!(t.iter().all(|v| (0.0..=1.0).contains(v)));
        }
        assert!(a.alum.iter().all(|&v| v == 0.0));
        assert!(a.pinning.iter().any(|&v| v > 0.4));
    }

    #[test]
    fn alum_lands_as_dots() {
        let mut p = Paper::generate(32, 32, 1);
        p.sprinkle_alum(5, 2.0, 0.8, 3);
        let covered = p.alum.iter().filter(|&&a| a > 0.0).count();
        assert!(covered > 5 && covered < 32 * 32 / 2, "{covered}");
    }
}
