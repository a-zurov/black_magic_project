// haar_wavelet/src/haar.rs
// ────────────────────────────────────────────────────────────────────────────
// haar.rs  –  1-D and 2-D Haar wavelet transforms
//
// Coefficient layout after L levels on a W×H image (row-major, stride = W):
//
//   ┌───────┬───────┬───────────────────┐
//   │  LL_L │  LH_L │                   │
//   ├───────┼───────┤       LH_1        │
//   │  HL_L │  HH_L │                   │
//   ├───────────────┼───────────────────┤
//   │               │                   │
//   │     HL_1      │       HH_1        │
//   │               │                   │
//   └───────────────┴───────────────────┘
//
// Subband origins (row_offset, col_offset) for level l in [1..L]:
//   LL  (only at coarsest L)  →  (0,          0        )
//   LH_l                      →  (0,          W >> l   )
//   HL_l                      →  (H >> l,     0        )
//   HH_l                      →  (H >> l,     W >> l   )
//
// Each subband at level l has size (H >> l) × (W >> l).
//
// Naming convention  (first letter = row direction, second = column direction):
//   L = low-pass (averages),  H = high-pass (differences)
// ────────────────────────────────────────────────────────────────────────────

/// Forward 1-D Haar wavelet transform (in-place, normalized by ½).
///
/// Input length must be a power of two and ≥ 2.
/// Output: first half = scaled averages, second half = scaled differences.
pub fn haar_1d_fwd(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(
        n >= 2 && n.is_power_of_two(),
        "haar_1d_fwd: length must be a power of 2 ≥ 2"
    );
    let h = n / 2;
    let mut buf = vec![0.0_f32; n];
    for i in 0..h {
        let a = data[2 * i];
        let b = data[2 * i + 1];
        buf[i] = (a + b) * 0.5; // average
        buf[h + i] = (a - b) * 0.5; // difference
    }
    data.copy_from_slice(&buf);
}

/// Inverse 1-D Haar wavelet transform (in-place).
///
/// Exact left-inverse of `haar_1d_fwd`.
pub fn haar_1d_inv(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(
        n >= 2 && n.is_power_of_two(),
        "haar_1d_inv: length must be a power of 2 ≥ 2"
    );
    let h = n / 2;
    let mut buf = vec![0.0_f32; n];
    for i in 0..h {
        let avg = data[i];
        let diff = data[h + i];
        buf[2 * i] = avg + diff;
        buf[2 * i + 1] = avg - diff;
    }
    data.copy_from_slice(&buf);
}

// ── 2-D ────────────────────────────────────────────────────────────────────

/// Forward 2-D Haar DWT for `levels` decomposition levels.
///
/// `pixels` is a row-major f32 slice of shape `height × width`.
/// Both `width` and `height` must be powers of two and ≥ 2^levels.
///
/// Returns a new `Vec<f32>` with the same layout as the input,
/// containing the wavelet coefficients in the tiled pyramid layout.
pub fn haar_2d_fwd(pixels: &[f32], width: usize, height: usize, levels: usize) -> Vec<f32> {
    assert!(
        width.is_power_of_two() && height.is_power_of_two(),
        "Width ({width}) and height ({height}) must both be powers of two"
    );
    assert!(
        width >= (1 << levels) && height >= (1 << levels),
        "Image ({width}×{height}) is too small for {levels} DWT levels"
    );

    let mut data = pixels.to_vec();

    // Each pass operates on the top-left (cw × ch) block.
    let mut cw = width;
    let mut ch = height;

    for _ in 0..levels {
        // 1) Row-wise 1-D Haar on every row of the current LL block.
        for row in 0..ch {
            let off = row * width; // stride is always the full image width
            haar_1d_fwd(&mut data[off..off + cw]);
        }

        // 2) Column-wise 1-D Haar on every column of the current LL block.
        let mut col_buf = vec![0.0_f32; ch];
        for col in 0..cw {
            for r in 0..ch {
                col_buf[r] = data[r * width + col];
            }
            haar_1d_fwd(&mut col_buf);
            for r in 0..ch {
                data[r * width + col] = col_buf[r];
            }
        }

        cw /= 2;
        ch /= 2;
    }

    data
}

/// Inverse 2-D Haar DWT for `levels` decomposition levels.
///
/// Exact left-inverse of `haar_2d_fwd` for the same `width`, `height`, `levels`.
pub fn haar_2d_inv(coeffs: &[f32], width: usize, height: usize, levels: usize) -> Vec<f32> {
    let mut data = coeffs.to_vec();

    // Collect the (width, height) of each forward pass block, then reverse.
    let pass_sizes: Vec<(usize, usize)> = (0..levels).map(|l| (width >> l, height >> l)).collect();

    // Undo passes in reverse order (finest → coarsest applied during forward
    // becomes coarsest → finest during inverse).
    for &(cw, ch) in pass_sizes.iter().rev() {
        // 1) Inverse column-wise transform.
        let mut col_buf = vec![0.0_f32; ch];
        for col in 0..cw {
            for r in 0..ch {
                col_buf[r] = data[r * width + col];
            }
            haar_1d_inv(&mut col_buf);
            for r in 0..ch {
                data[r * width + col] = col_buf[r];
            }
        }

        // 2) Inverse row-wise transform.
        for row in 0..ch {
            let off = row * width;
            haar_1d_inv(&mut data[off..off + cw]);
        }
    }

    data
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn roundtrip_1d() {
        let original = vec![1.0_f32, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        let mut data = original.clone();
        haar_1d_fwd(&mut data);
        haar_1d_inv(&mut data);
        for (a, b) in original.iter().zip(data.iter()) {
            assert!(approx_eq(*a, *b), "roundtrip failed: {a} ≠ {b}");
        }
    }

    #[test]
    fn roundtrip_2d() {
        let w = 8;
        let h = 8;
        let original: Vec<f32> = (0..(w * h)).map(|i| i as f32).collect();
        let coeffs = haar_2d_fwd(&original, w, h, 3);
        let restored = haar_2d_inv(&coeffs, w, h, 3);
        for (a, b) in original.iter().zip(restored.iter()) {
            assert!(approx_eq(*a, *b), "roundtrip failed: {a} ≠ {b}");
        }
    }

    #[test]
    fn energy_preservation_2d() {
        let w = 16;
        let h = 16;
        let pixels: Vec<f32> = (0..(w * h)).map(|i| (i % 50) as f32).collect();
        let coeffs = haar_2d_fwd(&pixels, w, h, 3);
        // Parseval: sum of squares is preserved (the ½ factor is baked in at each level,
        // so the total energy is scaled by (1/2)^levels per dimension).
        let e_in: f32 = pixels.iter().map(|&v| v * v).sum();
        let e_out: f32 = coeffs.iter().map(|&v| v * v).sum();
        // Energy ratio should be (1/4)^levels = 1/64 for 3 levels
        let ratio = e_out / e_in;
        let expected = 1.0_f32 / (4.0_f32.powi(3 as i32));
        assert!(
            (ratio - expected).abs() < 1e-4,
            "energy ratio {ratio} ≠ {expected}"
        );
    }
}
