//! The simulation's constants.
//!
//! Each field is named after what it does and documented with its symbol in Chu and Tai (2005).
//! The paper gives values or ranges for some of them; the weights that combine the paper's
//! textures (its equations 6 and 9) and the pigment rates are not published, and their defaults
//! here were tuned against the paper's figures. Those are marked *tuned*.

/// Constants of the water flow, the paper and the pigments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// ω, the lattice-Boltzmann relaxation rate. Lower is more viscous. The paper uses 0.5, and
    /// up to 1.5 for more fluent flow; keep it in `(0, 2)`.
    pub omega: f32,
    /// α: below this much water, advection weakens smoothly (equation 8), so thin water never
    /// carries density away into negative values. The paper uses 0.2–0.5.
    pub advection: f32,
    /// εs: water lost from every wet site each step. Paper: 0–0.005.
    pub evaporation: f32,
    /// εb: extra water lost through each link to a pinned dry site, so pigment migrates to the
    /// rim and darkens it. Paper: 5 × 10⁻⁵.
    pub edge_evaporation: f32,

    /// k₁, the blocking every site has (equation 6). *Tuned.*
    pub block_base: f32,
    /// k₂ per unit of paper grain: fibre-dense paper resists flow. *Tuned.*
    pub block_grain: f32,
    /// k₃ per unit of alum. *Tuned.*
    pub block_alum: f32,
    /// k₄ per unit of glue in the water: at 1, glue holds the water still. *Tuned.*
    pub block_glue: f32,
    /// k₅ per unit of ink already fixed in the paper. *Tuned.*
    pub block_fixed: f32,

    /// q₁, the water a dry site's neighbours must reach before it can be wetted (equation 9):
    /// higher values stop the front sooner. *Tuned.*
    pub pin_base: f32,
    /// q₂: fixed ink raises the threshold. *Tuned.*
    pub pin_fixed: f32,
    /// q₃: how much the paper's disorder (grain, or the pinning texture under gluey ink) varies
    /// the threshold, which roughens the front. *Tuned.*
    pub pin_disorder: f32,
    /// ϑ: glue concentration at which the pinning texture fully replaces the grain, giving the
    /// "toes" of concentrated ink. *Tuned.*
    pub toe_glue: f32,
    /// Multiplier of the threshold for the four diagonal neighbours. The paper writes 2; its
    /// authors' code defaults to 1.
    pub pin_diagonal: f32,

    /// π: how much water the paper's fibres hold.
    pub capacity: f32,
    /// Most water the surface can hold, waiting to seep in.
    pub surface_capacity: f32,
    /// λ: receptivity. Wet paper takes less new ink: a deposit is scaled by
    /// `max(1 − ρ/λ, m)`. Paper: 0.3–1.
    pub receptivity: f32,
    /// m: the least a deposit is scaled by, however wet the paper. Paper: 0.1.
    pub base_mask: f32,

    /// γ: the share of its ink that fast-moving water leaves behind in the paper (0 lets the ink
    /// travel with the water, 1 holds it in place). Slower water leaves more, up to
    /// [`Params::hindrance_still`]; this is what draws the ink into streaks. *Tuned.*
    pub hindrance: f32,
    /// ς: water speed above which the paper holds back only γ. *Tuned.*
    pub hindrance_speed: f32,
    /// The share held back in still water (the paper's formula uses 1). Below 1, ink is not
    /// trapped where the water drains without a net flow, such as under the middle of a dab.
    /// *Tuned.*
    pub hindrance_still: f32,
    /// η: the least fraction of pigment fixed into the paper each step. *Tuned.*
    pub fix_base: f32,
    /// µ: water level below which drying fixes pigment fully. *Tuned.*
    pub fix_dryness: f32,
    /// ξ: glue raises the drying level µ, so gluey ink fixes sooner. *Tuned.*
    pub fix_glue: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            omega: 0.5,
            advection: 0.3,
            evaporation: 0.0002,
            edge_evaporation: 2e-4,
            block_base: 0.05,
            block_grain: 0.85,
            block_alum: 0.6,
            block_glue: 1.0,
            block_fixed: 0.1,
            pin_base: 0.1,
            pin_fixed: 0.1,
            pin_disorder: 0.4,
            toe_glue: 0.3,
            pin_diagonal: 1.0,
            capacity: 1.0,
            surface_capacity: 5.0,
            receptivity: 0.6,
            base_mask: 0.1,
            hindrance: 0.3,
            hindrance_speed: 0.002,
            hindrance_still: 0.85,
            fix_base: 0.0001,
            fix_dryness: 0.15,
            fix_glue: 0.5,
        }
    }
}
