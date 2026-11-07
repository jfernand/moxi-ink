//! The simulation: water percolating through the paper by a lattice-Boltzmann model, carrying
//! ink (Chu and Tai §5).
//!
//! The paper has three layers:
//!
//! - the **surface**, where the brush leaves water and ink: a reservoir that seeps into the paper
//!   as fast as the fibres take it ([`Ink::surface_water`], [`Ink::surface_ink`]);
//! - the **flow** layer, where the water moves and carries ink ([`Ink::water`],
//!   [`Ink::flow_ink`]);
//! - the **fixture** layer, where ink settles for good as the water dries ([`Ink::fixed_ink`]).
//!
//! The water is a D2Q9 lattice-Boltzmann fluid (He and Luo's incompressible model) modified for
//! percolation:
//!
//! - **permeability**: each site partly blocks streaming by bouncing particles back
//!   ([`Ink::block`], equations 5 and 6);
//! - **weak advection in thin water**, so density never goes negative (equations 7 and 8);
//! - **pinning**: a dry site stays dry until a neighbour's water passes a threshold that the
//!   paper's disorder varies, which roughens the front (equation 9);
//! - **evaporation**, faster at the pinned rim, which draws ink there and darkens the edge.
//!
//! [`Ink::step`] runs the passes of the authors' implementation in order, each reading the
//! previous state and writing fresh buffers, so the result does not depend on the order sites are
//! visited (and the `parallel` feature can split them across threads).
//!
//! Departures from the paper and its code: water seeping in from the surface, and water
//! evaporating, are added to or taken from the particle distributions themselves (the authors'
//! code changed only the derived density, so part of it was lost or regained at the next
//! collision); a dry site's pinning threshold uses the glue of its wettest neighbour, since a dry
//! site has none of its own; and the surface layer holds water as an amount, not a fraction of
//! its capacity.

use crate::paper::Paper;
use crate::params::Params;

/// The D2Q9 lattice directions, `[dx, dy]` with y down: rest, N, E, W, S, NE, SE, NW, SW.
pub const DIRECTIONS: [[i32; 2]; 9] = [[0, 0], [0, -1], [1, 0], [-1, 0], [0, 1], [1, -1], [1, 1], [-1, -1], [-1, 1]];
/// The direction opposite each of [`DIRECTIONS`].
pub const OPPOSITE: [usize; 9] = [0, 4, 3, 2, 1, 8, 7, 6, 5];
/// The D2Q9 lattice weights.
pub const WEIGHTS: [f32; 9] =
    [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0];

/// Below this much water a site counts as dry, and its water is dropped.
pub const DRY: f32 = 1e-5;

/// What a brush puts down per unit of footprint: water, three pigment concentrations and glue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Load {
    /// Water.
    pub water: f32,
    /// Concentration of each of the three pigments in that water.
    pub pigment: [f32; 3],
    /// Concentration of glue: more glue spreads less and fixes sooner.
    pub glue: f32,
}

impl Load {
    /// Ink of one pigment (the first) at `strength`, in `water`, with a little glue.
    pub fn ink(water: f32, strength: f32) -> Self {
        Self { water, pigment: [strength, 0.0, 0.0], glue: 0.1 * strength }
    }

    fn channels(&self) -> [f32; 4] {
        [self.pigment[0], self.pigment[1], self.pigment[2], self.glue]
    }
}

/// A sheet of paper with ink on it, and everything needed to let the ink spread.
///
/// Ink values are `[pigment 1, pigment 2, pigment 3, glue]`. In the surface and flow layers they
/// are concentrations in the water there; in the fixture layer they are amounts.
#[derive(Clone, Debug)]
pub struct Ink {
    /// Sites per row.
    pub width: usize,
    /// Rows.
    pub height: usize,
    /// The constants. They may be changed between steps ("physics tinkering" in the paper).
    pub params: Params,
    /// The paper. Its textures may be changed between steps too (adding alum, say).
    pub paper: Paper,
    /// The particle distributions fᵢ along [`DIRECTIONS`].
    pub f: Vec<[f32; 9]>,
    /// ρ, the water in the flow layer.
    pub water: Vec<f32>,
    /// The water before the current step.
    pub last_water: Vec<f32>,
    /// u, the water velocity, in sites per step.
    pub velocity: Vec<[f32; 2]>,
    /// Water on the surface, waiting to seep in.
    pub surface_water: Vec<f32>,
    /// Water that seeped from the surface into the flow layer this step.
    pub seep: Vec<f32>,
    /// κ, how much each site blocks streaming: 0 lets everything through, 1 bounces everything
    /// back, and infinity marks a pinned dry site.
    pub block: Vec<f32>,
    /// The most water each site has held: positive wherever ink has been.
    pub stain: Vec<f32>,
    /// Ink concentrations in the surface water.
    pub surface_ink: Vec<[f32; 4]>,
    /// Ink concentrations in the flow layer's water.
    pub flow_ink: Vec<[f32; 4]>,
    /// Ink fixed in the paper.
    pub fixed_ink: Vec<[f32; 4]>,
    scratch_f: Vec<[f32; 9]>,
    scratch_ink: Vec<[f32; 4]>,
}

impl Ink {
    /// Dry, clean paper.
    pub fn new(paper: Paper, params: Params) -> Self {
        let (width, height) = (paper.width, paper.height);
        let n = width * height;
        Self {
            width,
            height,
            params,
            paper,
            f: vec![[0.0; 9]; n],
            water: vec![0.0; n],
            last_water: vec![0.0; n],
            velocity: vec![[0.0; 2]; n],
            surface_water: vec![0.0; n],
            seep: vec![0.0; n],
            block: vec![0.0; n],
            stain: vec![0.0; n],
            surface_ink: vec![[0.0; 4]; n],
            flow_ink: vec![[0.0; 4]; n],
            fixed_ink: vec![[0.0; 4]; n],
            scratch_f: vec![[0.0; 9]; n],
            scratch_ink: vec![[0.0; 4]; n],
        }
    }

    /// Puts `load` on the surface under `footprint` (one weight in `[0, 1]` per site, as made by
    /// [`crate::footprint`]). Wet paper takes less (§5.2): each site's share is scaled by
    /// `max(1 − ρ/λ, m)`, which is what leaves a light fringe where a stroke crosses a wet one.
    ///
    /// # Panics
    ///
    /// If `footprint` does not hold one weight per site.
    pub fn deposit(&mut self, footprint: &[f32], load: Load) {
        assert_eq!(footprint.len(), self.water.len(), "one footprint weight per site");
        let p = &self.params;
        let ink = load.channels();
        for (c, &weight) in footprint.iter().enumerate() {
            if weight <= 0.0 {
                continue;
            }
            let mask = (1.0 - self.water[c] / p.receptivity).max(p.base_mask);
            let before = self.surface_water[c];
            let after = (before + weight * mask * load.water).min(p.surface_capacity);
            let added = after - before;
            if added <= 0.0 {
                continue;
            }
            let s = &mut self.surface_ink[c];
            for k in 0..4 {
                s[k] = (s[k] * before + ink[k] * added) / after;
            }
            self.surface_water[c] = after;
        }
    }

    /// One time step: block, collide, stream, gather the water, fix ink, move ink.
    pub fn step(&mut self) {
        self.block_and_pin();
        self.collide();
        self.stream();
        self.gather_water();
        self.fix_ink();
        self.move_ink();
    }

    /// Steps until no water is left, or `max_steps` have run; returns the steps taken.
    pub fn dry(&mut self, max_steps: usize) -> usize {
        for s in 0..max_steps {
            if self.is_dry() {
                return s;
            }
            self.step();
        }
        max_steps
    }

    /// Whether no water is left on or in the paper.
    pub fn is_dry(&self) -> bool {
        self.water.iter().chain(&self.surface_water).all(|&w| w <= 0.0)
    }

    /// All water, on the surface and in the paper.
    pub fn total_water(&self) -> f64 {
        self.water.iter().chain(&self.surface_water).map(|&w| w as f64).sum()
    }

    /// All of each ink channel, in the three layers.
    pub fn total_ink(&self) -> [f64; 4] {
        let mut t = [0.0f64; 4];
        for c in 0..self.water.len() {
            for (k, total) in t.iter_mut().enumerate() {
                *total += (self.surface_ink[c][k] * self.surface_water[c]) as f64
                    + (self.flow_ink[c][k] * self.water[c]) as f64
                    + self.fixed_ink[c][k] as f64;
            }
        }
        t
    }

    /// The ink showing at each site: fixed, plus what the water in and on the paper carries.
    pub fn visible_ink(&self) -> Vec<[f32; 4]> {
        (0..self.water.len())
            .map(|c| {
                let mut v = self.fixed_ink[c];
                for (k, x) in v.iter_mut().enumerate() {
                    *x += self.flow_ink[c][k] * self.water[c] + self.surface_ink[c][k] * self.surface_water[c];
                }
                v
            })
            .collect()
    }

    fn neighbour(&self, c: usize, d: [i32; 2]) -> Option<usize> {
        let (x, y) = ((c % self.width) as i32 + d[0], (c / self.width) as i32 + d[1]);
        (x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height)
            .then(|| y as usize * self.width + x as usize)
    }

    /// Pass 1 (§5.3.1, §5.3.3): each site's blocking κ (equation 6), infinite at dry sites whose
    /// neighbours all hold less water than the pinning threshold σ (equation 9).
    fn block_and_pin(&mut self) {
        let mut block = std::mem::take(&mut self.block);
        fill(&mut block, |c| {
            let p = &self.params;
            let (grain, glue) = (self.paper.grain[c], self.flow_ink[c][3]);
            let fixed: f32 = self.fixed_ink[c].iter().sum();
            let kappa = (p.block_base
                + p.block_grain * grain
                + p.block_alum * self.paper.alum[c]
                + p.block_glue * glue
                + p.block_fixed * fixed)
                .min(1.0);
            if self.water[c] > 0.0 {
                return kappa;
            }
            // A dry site: it opens only once a neighbour's water passes the threshold.
            let wettest_glue = DIRECTIONS[1..]
                .iter()
                .filter_map(|&d| self.neighbour(c, d))
                .map(|n| self.flow_ink[n][3])
                .fold(0.0f32, f32::max);
            let toes = smoothstep(0.0, p.toe_glue, wettest_glue);
            let disorder = lerp(grain, self.paper.pinning[c], toes);
            let sigma = p.pin_base + p.pin_fixed * fixed + p.pin_disorder * disorder;
            let pinned = DIRECTIONS[1..].iter().enumerate().all(|(i, &d)| {
                let threshold = if i < 4 { sigma } else { sigma * p.pin_diagonal };
                self.neighbour(c, d).is_none_or(|n| self.water[n] < threshold)
            });
            if pinned { f32::INFINITY } else { kappa }
        });
        self.block = block;
        self.last_water.copy_from_slice(&self.water);
    }

    /// Pass 2: relax each site's distributions toward equilibrium (equations 1, 2, 7 and 8).
    fn collide(&mut self) {
        let (omega, alpha) = (self.params.omega, self.params.advection);
        let (water, velocity) = (&self.water, &self.velocity);
        fill_in_place(&mut self.f, |c, f| {
            let rho = water[c];
            if rho <= 0.0 {
                return;
            }
            let u = velocity[c];
            let psi = smoothstep(0.0, alpha, rho);
            let uu = u[0] * u[0] + u[1] * u[1];
            for i in 0..9 {
                let eu = DIRECTIONS[i][0] as f32 * u[0] + DIRECTIONS[i][1] as f32 * u[1];
                let eq = WEIGHTS[i] * (rho + psi * (3.0 * eu + 4.5 * eu * eu - 1.5 * uu));
                f[i] += omega * (eq - f[i]);
            }
        });
    }

    /// Pass 3: stream to the neighbours, each link partly bouncing back by the average κ of its
    /// two sites (equation 5). Links to a pinned site bounce back fully and lose εb (§5.3.4).
    /// The canvas edge is a wall.
    fn stream(&mut self) {
        let mut out = std::mem::take(&mut self.scratch_f);
        let edge_loss = self.params.edge_evaporation;
        fill(&mut out, |c| {
            let here = &self.f[c];
            let mut f = [0.0f32; 9];
            f[0] = here[0];
            for i in 1..9 {
                let back = here[OPPOSITE[i]];
                // The particle arriving along i comes from the site behind, against i.
                let behind = [-DIRECTIONS[i][0], -DIRECTIONS[i][1]];
                f[i] = match self.neighbour(c, behind) {
                    None => back,
                    Some(s) => {
                        let k = (self.block[c] + self.block[s]) / 2.0;
                        if k > 1.0 { (back - edge_loss).max(0.0) } else { lerp(self.f[s][i], back, k) }
                    }
                };
            }
            f
        });
        self.scratch_f = std::mem::replace(&mut self.f, out);
    }

    /// Pass 4: the water and its velocity (equations 3 and 4) after evaporation (εs), then the
    /// surface water seeping in as far as the fibres' capacity π allows (§5.2).
    fn gather_water(&mut self) {
        let p = self.params;
        for c in 0..self.water.len() {
            let f = &mut self.f[c];
            let mut rho: f32 = f.iter().sum();
            if rho > 0.0 && p.evaporation > 0.0 {
                let kept = (rho - p.evaporation).max(0.0) / rho;
                f.iter_mut().for_each(|x| *x *= kept);
                rho *= kept;
            }
            if rho < DRY {
                *f = [0.0; 9];
                rho = 0.0;
            }
            let mut u = [0.0f32; 2];
            for i in 1..9 {
                u[0] += DIRECTIONS[i][0] as f32 * f[i];
                u[1] += DIRECTIONS[i][1] as f32 * f[i];
            }
            let seep = self.surface_water[c].clamp(0.0, (p.capacity - rho).max(0.0));
            self.surface_water[c] -= seep;
            if self.surface_water[c] <= 0.0 {
                self.surface_water[c] = 0.0;
                self.surface_ink[c] = [0.0; 4];
            }
            f[0] += seep;
            rho += seep;
            self.seep[c] = seep;
            self.water[c] = rho;
            self.velocity[c] = u;
            self.stain[c] = self.stain[c].max(rho);
        }
    }

    /// Pass 5 (§5.4.3): ink settles into the fixture layer, one way, faster as the water dries
    /// and where there is more glue; all of it once the site is dry.
    fn fix_ink(&mut self) {
        let p = self.params;
        for c in 0..self.water.len() {
            let before = self.last_water[c];
            if before <= 0.0 {
                continue;
            }
            let now = self.water[c] - self.seep[c];
            let lost = (before - now).max(0.0) / before;
            let dryness = (p.fix_dryness + p.fix_glue * self.flow_ink[c][3]).clamp(0.0, 1.0);
            let fix =
                if now <= 0.0 { 1.0 } else { (lost * (1.0 - smoothstep(0.0, dryness, now))).max(p.fix_base).min(1.0) };
            for k in 0..4 {
                let amount = self.flow_ink[c][k] * before;
                let moved = fix * amount;
                self.fixed_ink[c][k] += moved;
                self.flow_ink[c][k] = if now > 0.0 { (amount - moved) / now } else { 0.0 };
            }
        }
    }

    /// Pass 6 (§5.4.1, §5.4.2): ink moves with the water. Sites wetted this step take the ink
    /// their incoming particles carried (equation 10); wet sites trace the velocity back. The
    /// paper holds some of it back (hindrance), and ink seeping from the surface mixes in.
    fn move_ink(&mut self) {
        let mut out = std::mem::take(&mut self.scratch_ink);
        let p = self.params;
        fill(&mut out, |c| {
            let water = self.water[c];
            if water <= 0.0 {
                return [0.0; 4];
            }
            let seep = self.seep[c];
            let flowing = water - seep;
            let old = self.flow_ink[c];
            let mut ink = [0.0f32; 4];
            if flowing > 0.0 {
                let carried =
                    if self.last_water[c] <= 0.0 { self.fill_from_neighbours(c, flowing) } else { self.trace_back(c) };
                let u = self.velocity[c];
                let speed = (u[0] * u[0] + u[1] * u[1]).sqrt();
                let held = lerp(1.0, p.hindrance, smoothstep(0.0, p.hindrance_speed, speed));
                for k in 0..4 {
                    ink[k] = lerp(carried[k], old[k], held);
                }
            }
            let surface = self.surface_ink[c];
            if seep > 0.0 {
                for k in 0..4 {
                    ink[k] = (ink[k] * flowing.max(0.0) + surface[k] * seep) / water;
                }
            }
            ink
        });
        self.scratch_ink = std::mem::replace(&mut self.flow_ink, out);
    }

    /// Equation 10: the ink each arriving particle brought from the site behind it.
    #[allow(clippy::needless_range_loop)]
    fn fill_from_neighbours(&self, c: usize, water: f32) -> [f32; 4] {
        let mut ink = [0.0f32; 4];
        for i in 1..9 {
            if let Some(s) = self.neighbour(c, [-DIRECTIONS[i][0], -DIRECTIONS[i][1]]) {
                for (k, x) in ink.iter_mut().enumerate() {
                    *x += self.f[c][i] * self.flow_ink[s][k];
                }
            }
        }
        ink.map(|x| x / water)
    }

    /// The method of characteristics: the ink at `x − u`, interpolated, unless the interpolation
    /// would reach across the wet area's boundary, in which case the ink stays as it is.
    fn trace_back(&self, c: usize) -> [f32; 4] {
        let u = self.velocity[c];
        let (x, y) = ((c % self.width) as f32 - u[0], (c / self.width) as f32 - u[1]);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        let mut corners = [0usize; 4];
        for (k, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
            let (px, py) = (x0 + dx, y0 + dy);
            if px < 0 || py < 0 || px as usize >= self.width || py as usize >= self.height {
                return self.flow_ink[c];
            }
            let s = py as usize * self.width + px as usize;
            if self.last_water[s] <= 0.0 || self.block[s].is_infinite() {
                return self.flow_ink[c];
            }
            corners[k] = s;
        }
        let at = |k: usize, ch: usize| self.flow_ink[corners[k]][ch];
        std::array::from_fn(|ch| {
            let top = lerp(at(0, ch), at(1, ch), fx);
            let bottom = lerp(at(2, ch), at(3, ch), fx);
            lerp(top, bottom, fy)
        })
    }
}

/// Hermite interpolation from 0 at `a` to 1 at `b`, as in shading languages.
pub fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    if b <= a {
        return if x < a { 0.0 } else { 1.0 };
    }
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `a` at `t = 0` to `b` at `t = 1`.
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Sets every element of `out` to `f(index)`, across threads with the `parallel` feature.
fn fill<T: Send>(out: &mut [T], f: impl Fn(usize) -> T + Sync + Send) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(c, x)| *x = f(c));
    }
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().enumerate().for_each(|(c, x)| *x = f(c));
}

/// Calls `f(index, element)` on every element, across threads with the `parallel` feature.
fn fill_in_place<T: Send>(out: &mut [T], f: impl Fn(usize, &mut T) + Sync + Send) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(c, x)| f(c, x));
    }
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().enumerate().for_each(|(c, x)| f(c, x));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::footprint;

    fn flat_paper(n: usize) -> Paper {
        let z = vec![0.0; n * n];
        Paper::from_textures(n, n, z.clone(), z.clone(), z.clone(), z)
    }

    fn no_evaporation() -> Params {
        Params { evaporation: 0.0, edge_evaporation: 0.0, ..Params::default() }
    }

    #[test]
    fn still_water_stays_still() {
        let n = 16;
        let mut ink = Ink::new(flat_paper(n), no_evaporation());
        ink.deposit(&vec![1.0; n * n], Load::ink(0.6, 0.0));
        for _ in 0..50 {
            ink.step();
        }
        assert!(ink.water.iter().all(|&w| (w - 0.6).abs() < 1e-4), "{:?}", &ink.water[..4]);
        assert!(ink.velocity.iter().all(|u| u[0].abs() < 1e-5 && u[1].abs() < 1e-5));
    }

    #[test]
    fn water_is_conserved_without_evaporation() {
        let n = 32;
        let mut ink = Ink::new(Paper::generate(n, n, 4), no_evaporation());
        ink.deposit(&footprint::disc(n, n, (16.0, 16.0), 5.0, 1.0), Load::ink(3.0, 0.5));
        let before = ink.total_water();
        for _ in 0..300 {
            ink.step();
            assert!(ink.f.iter().flatten().all(|&f| f >= 0.0), "negative distribution");
        }
        let after = ink.total_water();
        assert!((after - before).abs() / before < 1e-4, "{before} → {after}");
    }

    #[test]
    fn pinning_stops_the_front() {
        let n = 48;
        let spread = |pin_base: f32| {
            let mut ink = Ink::new(flat_paper(n), Params { pin_base, ..no_evaporation() });
            ink.deposit(&footprint::disc(n, n, (24.0, 24.0), 4.0, 0.0), Load::ink(1.5, 0.3));
            for _ in 0..400 {
                ink.step();
            }
            ink.stain.iter().filter(|&&s| s > 0.0).count()
        };
        let (easy, hard) = (spread(0.05), spread(0.4));
        assert!(hard < easy, "a higher threshold should spread less: {hard} vs {easy}");
        assert!(easy < n * n, "the front must stop somewhere: {easy}");
    }

    #[test]
    fn glue_holds_the_water() {
        let n = 40;
        let spread = |glue: f32| {
            let mut ink = Ink::new(flat_paper(n), no_evaporation());
            ink.deposit(&footprint::disc(n, n, (20.0, 20.0), 4.0, 0.0), Load { glue, ..Load::ink(1.5, 0.3) });
            for _ in 0..300 {
                ink.step();
            }
            ink.stain.iter().filter(|&&s| s > 0.0).count()
        };
        assert!(spread(1.0) < spread(0.0), "glue should hold the water back");
    }

    #[test]
    fn ink_dries_into_the_paper_and_is_kept() {
        let n = 32;
        let mut ink = Ink::new(Paper::generate(n, n, 2), Params { evaporation: 0.004, ..Params::default() });
        ink.deposit(&footprint::disc(n, n, (16.0, 16.0), 5.0, 1.0), Load::ink(1.5, 0.5));
        let before = ink.total_ink();
        let steps = ink.dry(5000);
        assert!(ink.is_dry(), "still wet after {steps} steps");
        assert!(ink.flow_ink.iter().all(|i| *i == [0.0; 4]));
        let after = ink.total_ink();
        // Tracing the velocity back is not exactly conservative; it should be close.
        assert!((after[0] - before[0]).abs() / before[0] < 0.1, "{before:?} → {after:?}");
    }

    #[test]
    fn same_seed_same_painting() {
        let run = || {
            let mut ink = Ink::new(Paper::generate(24, 24, 8), Params::default());
            ink.deposit(&footprint::disc(24, 24, (12.0, 12.0), 3.0, 1.0), Load::ink(2.0, 0.4));
            for _ in 0..80 {
                ink.step();
            }
            ink.visible_ink()
        };
        assert_eq!(run(), run());
    }
}
