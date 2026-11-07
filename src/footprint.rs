//! Brush footprints for [`crate::Ink::deposit`]: one weight in `[0, 1]` per site.
//!
//! These stand in for the 3D brush of Chu and Tai's system, which is out of scope: a disc for a
//! dab, and a stroke of discs swept along a path.

/// A dab: weight 1 inside radius `r` of `(cx, cy)`, falling to 0 over `soft` sites at the rim.
pub fn disc(width: usize, height: usize, (cx, cy): (f32, f32), r: f32, soft: f32) -> Vec<f32> {
    let mut out = vec![0.0; width * height];
    add_disc(&mut out, width, height, (cx, cy), r, soft);
    out
}

/// A stroke through `points`, each `(x, y, radius)`, swept with discs spaced half a site apart.
/// Weights do not add up where the stroke overlaps itself.
pub fn stroke(width: usize, height: usize, points: &[(f32, f32, f32)], soft: f32) -> Vec<f32> {
    let mut out = vec![0.0; width * height];
    for pair in points.windows(2) {
        let ((x0, y0, r0), (x1, y1, r1)) = (pair[0], pair[1]);
        let steps = ((x1 - x0).hypot(y1 - y0) * 2.0).ceil().max(1.0) as usize;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            add_disc(&mut out, width, height, (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t), r0 + (r1 - r0) * t, soft);
        }
    }
    if let [(x, y, r)] = points {
        add_disc(&mut out, width, height, (*x, *y), *r, soft);
    }
    out
}

fn add_disc(out: &mut [f32], width: usize, height: usize, (cx, cy): (f32, f32), r: f32, soft: f32) {
    let reach = r + soft;
    let (x0, x1) = ((cx - reach).floor().max(0.0) as usize, ((cx + reach).ceil().max(0.0) as usize).min(width));
    let (y0, y1) = ((cy - reach).floor().max(0.0) as usize, ((cy + reach).ceil().max(0.0) as usize).min(height));
    for y in y0..y1 {
        for x in x0..x1 {
            let d = (x as f32 - cx).hypot(y as f32 - cy);
            let w = if d <= r {
                1.0
            } else if soft > 0.0 && d < reach {
                1.0 - (d - r) / soft
            } else {
                0.0
            };
            let o = &mut out[y * width + x];
            *o = o.max(w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn footprints_cover_what_they_should() {
        let d = disc(20, 20, (10.0, 10.0), 3.0, 1.0);
        assert_eq!(d[10 * 20 + 10], 1.0);
        assert_eq!(d[0], 0.0);
        assert!(d.iter().all(|w| (0.0..=1.0).contains(w)));
        let s = stroke(30, 10, &[(3.0, 5.0, 1.5), (26.0, 5.0, 1.5)], 0.0);
        assert!((3..=26).all(|x| s[5 * 30 + x] == 1.0));
        assert_eq!(s[0], 0.0);
    }
}
