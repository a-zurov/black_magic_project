/// CDF 5/3 Wavelet Transform - lifting scheme
///
/// Data is interleaved in-place:
///   even indices (0, 2, 4 ...) - scaling coefficients (s)
///   odd  indices (1, 3, 5 ...) - detail  coefficients (d)
///
/// Lifting steps (forward):
///   Step 1 Predict (update d using neighbours of s):
///     d[n] += A * (s[n] + s[n+1])
///   Step 2 Update (update s using neighbours of d):
///     s[n] += B * (d[n-1] + d[n])
///
/// Lifting steps (inverse) - reverse order, negate coefficients.
///
/// Normalisation factor E = sqrt(2):
///   forward : d /= E,  s *= E
///   inverse : s /= E,  d *= E

pub type Dt = f32;

/// A = -1/2  (predict step coefficient)
const A_DWT53: Dt = -0.5;

/// B = +1/4  (update step coefficient)
const B_DWT53: Dt = 0.25;

/// E = sqrt(2)  (energy normalisation)
const E_DWT53: Dt = std::f32::consts::SQRT_2;

// ==== helpers ===================================================

/// Update **odd** (detail) samples:
///   data[i] += coeff * (data[i - step] + data[i + step])
/// Boundary (right edge): mirror - use 2 * data[i - step].
#[inline]
fn lift_odd(data: &mut [Dt], step: usize, coeff: Dt) {
    let step2 = 2 * step;
    let n = data.len();
    let end = n.saturating_sub(step);

    let mut i = step;
    while i < end {
        data[i] += (data[i - step] + data[i + step]) * coeff;
        i += step2;
    }
    if i < n {
        // right boundary: mirror
        data[i] += data[i - step] * 2.0 * coeff;
    }
}

/// Update **even** (scaling) samples:
///   data[i] += coeff * (data[i - step] + data[i + step])
/// Left boundary (i == 0): mirror - use 2 * data[step].
/// Right boundary: mirror - use 2 * data[i - step].
#[inline]
fn lift_even(data: &mut [Dt], step: usize, coeff: Dt) {
    let step2 = 2 * step;
    let n = data.len();
    let end = n.saturating_sub(step);

    // left boundary
    if step < n {
        data[0] += data[step] * 2.0 * coeff;
    }

    let mut i = step2;
    while i < end {
        data[i] += (data[i - step] + data[i + step]) * coeff;
        i += step2;
    }
    if i < n {
        // right boundary
        data[i] += data[i - step] * 2.0 * coeff;
    }
}

// ==== forward transform ===================================================

/// CDF 5/3 forward DWT (analysis).
///
/// `data`   - interleaved buffer, length must be > 1
/// `step`   - stride (1 for the top level, grows with recursion depth)
///
/// After the call:
///   even positions -> low-pass  (scaling) coefficients
///   odd  positions -> high-pass (detail) coefficients
pub fn dwt53_forward(data: &mut [Dt], step: usize) {
    // ==== Step 1: Predict - update detail (odd) samples ====================
    //   d[n] += A * (s[n] + s[n+1])
    lift_odd(data, step, A_DWT53);

    // ==== Step 2: Update - update scaling (even) samples ====================
    //   s[n] += B * (d[n-1] + d[n])
    lift_even(data, step, B_DWT53);

    // ==== Normalisation =====================================================
    let n = data.len();
    let step2 = 2 * step;

    // detail: divide by sqrt(2)
    let mut i = step;
    while i < n {
        data[i] /= E_DWT53;
        i += step2;
    }
    // scaling: multiply by sqrt(2)
    let mut i = 0;
    while i < n {
        data[i] *= E_DWT53;
        i += step2;
    }
}

// ==== inverse transform =====================================================

/// CDF 5/3 inverse DWT (synthesis).
///
/// Exact inverse of `dwt53_forward` - undo normalisation, then
/// apply lifting steps in reverse order with negated coefficients.
pub fn dwt53_inverse(data: &mut [Dt], step: usize) {
    let n = data.len();
    let step2 = 2 * step;

    // ==== Undo normalisation =================================================
    // scaling: divide by sqrt(2)
    let mut i = 0;
    while i < n {
        data[i] /= E_DWT53;
        i += step2;
    }
    // detail: multiply by sqrt(2)
    let mut i = step;
    while i < n {
        data[i] *= E_DWT53;
        i += step2;
    }

    // ==== Step 2 (inverse): undo Update =====================================
    //   s[n] -= B * (d[n-1] + d[n])
    lift_even(data, step, -B_DWT53);

    // ==== Step 1 (inverse): undo Predict =====================================
    //   d[n] -= A * (s[n] + s[n+1])
    lift_odd(data, step, -A_DWT53);
}

// ==== multi-level decomposition ==============================================
/// Apply `levels` forward DWT passes over the scaling subband.
pub fn dwt53_forward_multilevel(data: &mut [Dt], levels: usize) {
    let mut step = 1usize;
    for _ in 0..levels {
        if data.len() / step < 2 {
            break;
        }
        dwt53_forward(data, step);
        step *= 2;
    }
}

/// Undo `levels` forward passes (must match what was passed to forward).
pub fn dwt53_inverse_multilevel(data: &mut [Dt], levels: usize) {
    // work out the final step used in the forward pass
    let mut step = 1usize << levels.saturating_sub(1);
    for _ in 0..levels {
        if step == 0 {
            break;
        }
        dwt53_inverse(data, step);
        step >>= 1;
    }
}

// ==== tests ==================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[Dt], b: &[Dt]) -> Dt {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn round_trip_single_level() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0_f32];
        let mut data = original.clone();

        dwt53_forward(&mut data, 1);
        dwt53_inverse(&mut data, 1);

        let err = max_abs_diff(&data, &original);
        println!("single-level round-trip error: {:.2e}", err);
        assert!(err < 1e-5, "round-trip error too large: {}", err);
    }

    #[test]
    fn round_trip_multilevel() {
        let original: Vec<Dt> = (0..64).map(|i| (i as f32).sin()).collect();
        let mut data = original.clone();

        let levels = 4;
        dwt53_forward_multilevel(&mut data, levels);
        dwt53_inverse_multilevel(&mut data, levels);

        let err = max_abs_diff(&data, &original);
        println!("multi-level round-trip error: {:.2e}", err);
        assert!(err < 1e-4, "round-trip error too large: {}", err);
    }

    #[test]
    fn dc_signal_goes_to_scaling_only() {
        // A constant signal should produce zero detail coefficients
        let mut data = vec![3.0_f32; 8];
        dwt53_forward(&mut data, 1);

        // odd indices are detail - should be near zero
        let detail_energy: Dt = (1..8).step_by(2).map(|i| data[i] * data[i]).sum();
        println!("detail energy for DC input: {:.2e}", detail_energy);
        assert!(detail_energy < 1e-9);
    }

    #[test]
    fn print_coefficients_1() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0_f32];
        println!("input:    {:?}", data);
        dwt53_forward(&mut data, 1);
        println!("forward:  {:?}", data);
        dwt53_inverse(&mut data, 1);
        println!("restored: {:?}", data);
    }

    #[test]
    fn print_coefficients_2() {
        let mut data = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0_f32];
        println!("input:    {:?}", data);
        dwt53_forward(&mut data, 1);
        println!("forward:  {:?}", data);
        dwt53_inverse(&mut data, 1);
        println!("restored: {:?}", data);
    }

    #[test]
    fn print_coefficients_3() {
        let mut data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0_f32];
        println!("input:    {:?}", data);
        dwt53_forward(&mut data, 1);
        println!("forward:  {:?}", data);
        dwt53_inverse(&mut data, 1);
        println!("restored: {:?}", data);
    }

    #[test]
    fn print_coefficients_4() {
        let mut data = vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32];
        println!("input:    {:?}", data);
        dwt53_forward(&mut data, 1);
        println!("forward:  {:?}", data);
        dwt53_inverse(&mut data, 1);
        println!("restored: {:?}", data);
    }

    #[test]
    fn print_coefficients_5() {
        let mut data = vec![-10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0_f32];
        println!("input:    {:?}", data);
        dwt53_forward(&mut data, 1);
        println!("forward:  {:?}", data);
        dwt53_inverse(&mut data, 1);
        println!("restored: {:?}", data);
    }
}
