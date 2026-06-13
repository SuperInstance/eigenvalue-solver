# Eigenvalue Solver

**A numerical linear algebra library for computing eigenvalues and eigenvectors** of square matrices using the power iteration method and QR algorithm — the mathematical engines behind PageRank, principal component analysis (PCA), and vibration analysis.

## Why It Matters

Eigenvalues and eigenvectors are among the most important computations in applied mathematics. They reveal the fundamental modes of a linear transformation: eigenvectors are directions that don't rotate under the transformation, and eigenvalues are the scaling factors in those directions.

**Applications:**
- **PageRank** — Google's original ranking algorithm computes the dominant eigenvector of the web link matrix. The `power_iteration` function is literally how PageRank works.
- **PCA (Principal Component Analysis)** — Dimensionality reduction for machine learning computes all eigenvalues of the covariance matrix via the QR algorithm.
- **Quantum mechanics** — Energy states are eigenvalues of the Hamiltonian operator.
- **Vibration analysis** — Natural frequencies of structures are eigenvalues of the stiffness matrix.
- **Stability analysis** — Eigenvalues of the Jacobian determine whether a dynamical system is stable.

**Two methods implemented:**
- **Power iteration** — Finds only the dominant (largest magnitude) eigenvalue, but is simple and memory-efficient. Converges quickly when there's a large gap between the top two eigenvalues.
- **QR algorithm** — Finds all eigenvalues of a symmetric matrix by iterating QR decompositions until the matrix becomes diagonal.

## How It Works

**Power iteration:** Start with a random vector v. Repeatedly compute v = A·v, normalizing after each step. Under most conditions, v converges to the dominant eigenvector, and the Rayleigh quotient (vᵀ·A·v / vᵀ·v) converges to the dominant eigenvalue.

The intuition: any vector can be decomposed as a linear combination of eigenvectors. Each multiplication by A scales each component by its eigenvalue. After k iterations, the dominant component is scaled by λ₁ᵏ while others are scaled by λᵢᵏ (where |λᵢ| < |λ₁|). Eventually the dominant component overwhelms all others — the vector aligns with the dominant eigenvector.

**Convergence rate:** O(|λ₂/λ₁|ᵏ) — fast when λ₁ >> λ₂, slow when they're close. The `tol` parameter stops early when the eigenvalue estimate stabilizes.

**QR algorithm:** Repeatedly decompose A = QR (via Gram-Schmidt), then reform A = RQ. Each iteration moves A closer to upper-triangular (Schur) form, where eigenvalues appear on the diagonal. For symmetric matrices, it converges to diagonal form.

**Gram-Schmidt QR decomposition:** For each column of A, subtract its projections onto all previous Q columns, then normalize. This produces orthogonal Q and upper-triangular R such that A = QR.

## Quick Start

```rust
use eigenvalue_solver::{power_iteration, qr_eigenvalues};

// Power iteration: find dominant eigenvalue
let matrix = vec![
    vec![2.0, 1.0],
    vec![1.0, 3.0],
];
let (eigenvalue, eigenvector) = power_iteration(&matrix, 100, 1e-10);
println!("Dominant eigenvalue: {:.6}", eigenvalue);
// For [[2,1],[1,3]], dominant eigenvalue ≈ 3.618 (golden ratio + 2)
println!("Eigenvector: {:?}", eigenvector);

// QR algorithm: find all eigenvalues
let matrix = vec![
    vec![4.0, 1.0, 0.0],
    vec![1.0, 3.0, 1.0],
    vec![0.0, 1.0, 2.0],
];
let eigenvalues = qr_eigenvalues(matrix, 100);
println!("All eigenvalues: {:?}", eigenvalues);
```

## API

### `power_iteration(matrix: &[Vec<f64>], max_iter: usize, tol: f64) -> (f64, Vec<f64>)`
- Find dominant eigenvalue and normalized eigenvector via iterative multiplication
- O(n² · iterations) time, O(n) space
- `tol`: convergence threshold for eigenvalue change

### `qr_eigenvalues(matrix: Vec<Vec<f64>>, max_iter: usize) -> Vec<f64>`
- Find all eigenvalues of a symmetric matrix via QR iteration
- O(n³ · iterations) time, O(n²) space
- Consumes the matrix (no copy needed)

## Architecture Notes

This library provides numerical linear algebra primitives for SuperInstance's scientific computing and machine learning toolkit. The power iteration method powers PageRank-style ranking algorithms, and the QR decomposition supports PCA and spectral clustering.

See the full architecture: [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md)

## License

MIT
