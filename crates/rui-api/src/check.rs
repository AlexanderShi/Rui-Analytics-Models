//! Independent check of a first-order (KKT) point, computed only from the oracle's callbacks and the
//! original bounds — not from anything the solver computed internally.
//!
//! With `σ = +1` for minimization and `-1` for maximization, the multipliers of a solution satisfy
//!
//! ```text
//!     σ ∇f(x) + J(x)ᵀ λ − z_L + z_U = 0,       z_L >= 0,  z_U >= 0,
//!     λ_i <= 0 where g_L,i is the active side,  λ_i >= 0 where g_U,i is the active side
//! ```
//!
//! (Ipopt's convention for the minimized function `σ f`; equality rows have free sign). All measures
//! are unscaled max-norms. Complementarity is the natural residual `min(|slack|, |multiplier|)`, which
//! is zero exactly when one of the two is. NaN anywhere makes the corresponding measure infinite.

use crate::problem::{constraint_bound_vectors, EvalError, SmoothOracle};
use crate::solution::Duals;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KktReport {
    /// Largest bound or constraint violation.
    pub primal_infeasibility: f64,
    /// `‖σ ∇f + Jᵀλ − z_L + z_U‖∞`.
    pub dual_infeasibility: f64,
    /// Largest `min(|slack|, |multiplier|)` over bounds and inequality constraints.
    pub complementarity: f64,
    /// Largest multiplier of the wrong sign, or on an infinite bound.
    pub sign_violation: f64,
}

impl KktReport {
    pub fn within(&self, tol: f64) -> bool {
        self.max() <= tol
    }

    pub fn max(&self) -> f64 {
        let mut m = 0.0;
        for v in [self.primal_infeasibility, self.dual_infeasibility, self.complementarity, self.sign_violation] {
            worse(&mut m, v);
        }
        m
    }
}

/// `acc = max(acc, v)`, with NaN counting as infinitely bad.
fn worse(acc: &mut f64, v: f64) {
    if v.is_nan() {
        *acc = f64::INFINITY;
    } else if v > *acc {
        *acc = v;
    }
}

/// Checks `x` with `duals` against `problem` (its own sense decides `σ`).
pub fn kkt_check(problem: &dyn SmoothOracle, x: &[f64], duals: &Duals) -> Result<KktReport, EvalError> {
    let vars = problem.variables();
    let n = vars.len();
    let m = problem.num_constraints();
    if x.len() != n || duals.lower.len() != n || duals.upper.len() != n || duals.constraints.len() != m {
        return Err(EvalError::new(format!("kkt_check: expected x, z_L, z_U of length {n} and λ of length {m}")));
    }
    let sigma = problem.sense().sign();
    let mut r = vec![0.0; n];
    problem.gradient(x, &mut r)?;
    for v in &mut r {
        *v *= sigma;
    }
    let mut rep = KktReport::default();
    if m > 0 {
        let (gl, gu) = constraint_bound_vectors(problem);
        let mut g = vec![0.0; m];
        problem.constraints(x, &mut g)?;
        let structure = problem.jacobian_structure();
        let mut values = vec![0.0; structure.len()];
        problem.jacobian_values(x, &mut values)?;
        for (&(i, j), &v) in structure.iter().zip(&values) {
            if i >= m || j >= n {
                return Err(EvalError::new(format!("kkt_check: Jacobian entry ({i}, {j}) is out of range")));
            }
            r[j] += v * duals.constraints[i];
        }
        for i in 0..m {
            worse(&mut rep.primal_infeasibility, (gl[i] - g[i]).max(g[i] - gu[i]));
            if g[i].is_nan() {
                rep.primal_infeasibility = f64::INFINITY;
            }
            let lam = duals.constraints[i];
            if gl[i] == gu[i] {
                continue; // equality: free sign, no complementarity
            }
            if lam < 0.0 {
                if gl[i].is_finite() {
                    worse(&mut rep.complementarity, (g[i] - gl[i]).abs().min(-lam));
                } else {
                    worse(&mut rep.sign_violation, -lam);
                }
            } else if lam > 0.0 {
                if gu[i].is_finite() {
                    worse(&mut rep.complementarity, (gu[i] - g[i]).abs().min(lam));
                } else {
                    worse(&mut rep.sign_violation, lam);
                }
            } else if lam.is_nan() {
                rep.sign_violation = f64::INFINITY;
            }
        }
    }
    for j in 0..n {
        let (l, u) = (vars[j].lower, vars[j].upper);
        worse(&mut rep.primal_infeasibility, (l - x[j]).max(x[j] - u));
        if x[j].is_nan() {
            rep.primal_infeasibility = f64::INFINITY;
        }
        let (zl, zu) = (duals.lower[j], duals.upper[j]);
        r[j] += zu - zl;
        for (z, bound, slack) in [(zl, l, x[j] - l), (zu, u, u - x[j])] {
            if z.is_nan() || z < 0.0 {
                worse(&mut rep.sign_violation, -z);
            } else if z > 0.0 {
                if bound.is_finite() {
                    worse(&mut rep.complementarity, slack.abs().min(z));
                } else {
                    worse(&mut rep.sign_violation, z);
                }
            }
        }
    }
    for v in &r {
        worse(&mut rep.dual_infeasibility, v.abs());
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{Problem, Sense, Variable};

    /// min/max x0 + x1  s.t.  x0² + x1² <= 2 (or >= -2 from below with the sign flipped), x0 <= 0.5.
    struct Disk {
        vars: Vec<Variable>,
        sense: Sense,
    }

    impl Problem for Disk {
        fn variables(&self) -> &[Variable] {
            &self.vars
        }
        fn num_constraints(&self) -> usize {
            1
        }
        fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
            lower[0] = f64::NEG_INFINITY;
            upper[0] = 2.0;
        }
        fn sense(&self) -> Sense {
            self.sense
        }
    }

    impl SmoothOracle for Disk {
        fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
            Ok(x[0] + x[1])
        }
        fn gradient(&self, _x: &[f64], grad: &mut [f64]) -> Result<(), EvalError> {
            grad.copy_from_slice(&[1.0, 1.0]);
            Ok(())
        }
        fn constraints(&self, x: &[f64], g: &mut [f64]) -> Result<(), EvalError> {
            g[0] = x[0] * x[0] + x[1] * x[1];
            Ok(())
        }
        fn jacobian_structure(&self) -> Vec<(usize, usize)> {
            vec![(0, 0), (0, 1)]
        }
        fn jacobian_values(&self, x: &[f64], values: &mut [f64]) -> Result<(), EvalError> {
            values[0] = 2.0 * x[0];
            values[1] = 2.0 * x[1];
            Ok(())
        }
    }

    fn disk(sense: Sense) -> Disk {
        Disk { vars: vec![Variable::continuous("x0", f64::NEG_INFINITY, 0.5), Variable::free("x1")], sense }
    }

    #[test]
    fn maximum_on_the_disk_with_an_active_bound() {
        // max x0 + x1 on the disk with x0 <= 0.5: x = (0.5, sqrt(1.75)).
        // σ = -1: -∇f + λ ∇g + z_U e0 = 0  ⇒  λ = 1 / (2 x1), z_U = 1 - 2 x0 λ.
        let p = disk(Sense::Maximize);
        let x1 = 1.75f64.sqrt();
        let lam = 1.0 / (2.0 * x1);
        let duals = Duals { constraints: vec![lam], lower: vec![0.0, 0.0], upper: vec![1.0 - lam, 0.0] };
        let rep = kkt_check(&p, &[0.5, x1], &duals).unwrap();
        assert!(rep.within(1e-14), "{rep:?}");

        // The same multipliers do not certify the minimization problem.
        let rep = kkt_check(&disk(Sense::Minimize), &[0.5, x1], &duals).unwrap();
        assert!((rep.dual_infeasibility - 2.0).abs() < 1e-14, "{rep:?}");
    }

    #[test]
    fn wrong_signs_and_slack_are_reported() {
        let p = disk(Sense::Minimize);
        // Multiplier on the missing lower side of the constraint and a negative z_U.
        let duals = Duals { constraints: vec![-0.5], lower: vec![0.0, 0.0], upper: vec![-0.25, 0.0] };
        let rep = kkt_check(&p, &[0.0, 0.0], &duals).unwrap();
        assert_eq!(rep.sign_violation, 0.5);
        // Positive z_U on x0 = 0 < 0.5: complementarity min(0.5, 0.75).
        let duals = Duals { constraints: vec![0.0], lower: vec![0.0, 0.0], upper: vec![0.75, 0.0] };
        let rep = kkt_check(&p, &[0.0, 0.0], &duals).unwrap();
        assert_eq!(rep.complementarity, 0.5);
        // Infeasible point.
        let rep =
            kkt_check(&p, &[1.0, 2.0], &Duals { constraints: vec![0.0], lower: vec![0.0; 2], upper: vec![0.0; 2] })
                .unwrap();
        assert_eq!(rep.primal_infeasibility, 3.0);
        assert!(kkt_check(&p, &[0.0], &Duals::default()).is_err());
    }
}
