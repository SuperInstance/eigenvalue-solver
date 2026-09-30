/// Power iteration method for dominant eigenvalue
pub fn power_iteration(matrix: &[Vec<f64>], max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
    let n = matrix.len();
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix[i][j] * v[j];
            }
        }
        let new_eigenvalue = w.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            v = w.iter().map(|x| x / norm).collect();
        }
        if (new_eigenvalue - eigenvalue).abs() < tol {
            eigenvalue = new_eigenvalue;
            break;
        }
        eigenvalue = new_eigenvalue;
    }
    (eigenvalue, v)
}

/// QR algorithm for all eigenvalues (symmetric matrices)
pub fn qr_eigenvalues(mut matrix: Vec<Vec<f64>>, max_iter: usize) -> Vec<f64> {
    let n = matrix.len();
    for _ in 0..max_iter {
        // Gram-Schmidt QR decomposition
        let mut q = vec![vec![0.0; n]; n];
        let mut r = vec![vec![0.0; n]; n];
        let mut u = vec![vec![0.0; n]; n];
        for j in 0..n {
            for i in 0..n {
                u[i][j] = matrix[i][j];
            }
            for k in 0..j {
                let dot: f64 = (0..n).map(|i| u[i][j] * q[i][k]).sum();
                r[k][j] = dot;
                for i in 0..n {
                    u[i][j] -= dot * q[i][k];
                }
            }
            let norm = (0..n).map(|i| u[i][j] * u[i][j]).sum::<f64>().sqrt();
            r[j][j] = norm;
            if norm > 0.0 {
                for i in 0..n {
                    q[i][j] = u[i][j] / norm;
                }
            }
        }
        // A = R * Q
        matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    matrix[i][j] += r[i][k] * q[k][j];
                }
            }
        }
    }
    (0..n).map(|i| matrix[i][i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_power_iteration() {
        let m = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let (val, vec_) = power_iteration(&m, 100, 1e-10);
        assert!(val > 3.0 && val < 4.0);
        assert!((vec_.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-8);
    }
}

/// FNV-1a 64 — the digest every substrate in the SuperInstance fleet agrees on.
pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
pub const FNV_PRIME: u64 = 0x100000001b3;

#[inline]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

/// True if this crate's FNV-1a still agrees with the rest of the fleet.
pub fn canary_holds() -> bool {
    fnv1a64("café Δ 日本語".as_bytes()) == 0x024a555471370b18d
}
