//! Eastern ink dispersing through absorbent paper: a Rust implementation of Nelson S.-H. Chu and
//! Chiew-Lan Tai, *MoXi: Real-Time Ink Dispersion in Absorbent Paper* (SIGGRAPH 2005).
//!
//! Where watercolor flows over the paper's surface, ink on thin Xuan paper soaks *through* the
//! fibres. MoXi models that water as a lattice-Boltzmann fluid whose streaming the paper partly
//! blocks, with a wet front that is pinned until enough water pushes it on, and ink that rides
//! the water and settles as it dries ([`ink`]). That gives the effects of real ink painting:
//! spreading blots with feathery streaks, light fringes, branching around sizing, rough toes along
//! the edge of concentrated ink, and dark rims.
//!
//! Put ink down with [`Ink::deposit`] and a [`footprint`], run [`Ink::step`] until it dries, and
//! turn it into colours with [`render`] or the sharper, upscaled [`render_trimmed`].
//!
//! ```
//! use moxi_ink::{Ink, Load, Palette, Paper, Params, footprint, render};
//!
//! let paper = Paper::generate(48, 48, 1);
//! let mut ink = Ink::new(paper, Params::default());
//! ink.deposit(&footprint::disc(48, 48, (24.0, 24.0), 4.0, 1.0), Load::ink(2.0, 0.6));
//! for _ in 0..100 {
//!     ink.step();
//! }
//! // The blot has spread beyond the brush.
//! let wet = ink.stain.iter().filter(|&&s| s > 0.0).count();
//! assert!(wet > 60, "{wet}");
//! let colours = render(&ink, &Palette::default());
//! assert!(colours[24 * 48 + 24][0] < colours[0][0]);
//! ```

#![warn(missing_docs)]

pub mod footprint;
pub mod ink;
pub mod paper;
pub mod params;
pub mod render;
pub mod rng;

pub use ink::{Ink, Load};
pub use paper::Paper;
pub use params::Params;
pub use render::{Palette, Trim, render, render_trimmed, to_srgb8};

/// The crate version, for caches keyed on the simulator's behaviour.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
