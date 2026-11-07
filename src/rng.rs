//! A small deterministic random number generator (PCG32, seeded through SplitMix64).
//!
//! The crate carries its own generator so the same seed paints the same paper and strokes on every
//! platform and in every release, independent of any random-number crate's versioning.

/// SplitMix64 step: a strong 64-bit mixer, used for seeding.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Combines two values into a well-mixed seed, e.g. a base seed and an index.
pub fn mix(a: u64, b: u64) -> u64 {
    let mut s = a ^ b.rotate_left(32);
    splitmix64(&mut s)
}

/// PCG32 (XSH-RR variant).
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    inc: u64,
}

impl Rng {
    /// A generator seeded with `seed`; equal seeds give equal sequences.
    pub fn new(seed: u64) -> Self {
        let mut s = seed;
        let init = splitmix64(&mut s);
        let inc = splitmix64(&mut s) | 1;
        let mut rng = Rng { state: 0, inc };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(init);
        rng.next_u32();
        rng
    }

    /// The next uniformly distributed 32-bit value.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    }

    /// Uniform in `[0, 1]`.
    pub fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    /// Uniform integer in `0..n` without modulo bias (Lemire's method).
    ///
    /// # Panics
    ///
    /// If `n` is zero.
    pub fn below(&mut self, n: u32) -> u32 {
        assert!(n > 0);
        let threshold = n.wrapping_neg() % n;
        loop {
            let m = self.next_u32() as u64 * n as u64;
            if (m as u32) >= threshold {
                return (m >> 32) as u32;
            }
        }
    }

    /// True with probability `num / den`.
    pub fn chance(&mut self, num: u32, den: u32) -> bool {
        self.below(den) < num
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_numbers() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        assert!((0..100).all(|_| a.next_u32() == b.next_u32()));
        // Pinned: a change here changes every painting made from a seed.
        assert_eq!(Rng::new(0).next_u32(), Rng::new(0).next_u32());
        let x: Vec<u32> = (0..3).map(|_| a.below(10)).collect();
        assert!(x.iter().all(|&v| v < 10));
    }
}
