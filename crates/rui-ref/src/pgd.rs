//! Projected gradient for `min f(x)` subject to `x_L <= x <= x_U`.
//!
//! Each iteration tries `x(α) = P(x − α ∇f(x))`, where `P` projects onto the bounds, starting from the
//! Barzilai–Borwein step `α = sᵀs / sᵀy` and halving `α` until the Armijo condition
//! `f(x(α)) <= f(x) + 10⁻⁴ ∇f(x)ᵀ(x(α) − x)` holds. It stops when the projected gradient
//! `‖P(x − ∇f(x)) − x‖∞` is at most `tol`.
//!
//! The template for derivative-based plugins: it reports multipliers and confirms them with the
//! independent check [`rui_api::kkt_check`] before claiming a KKT point.

use rui_api::{
    common, default_start, kkt_check, Access, Capabilities, Context, Duals, Event, FnClass, OptionSpec, PointKind,
    ProblemRef, ResolvedOptions, Sense, Solution, Solver, SolverInfo, Termination,
};

/// Projected gradient (`method=pgd`).
pub struct ProjectedGradient;

const ARMIJO: f64 = 1e-4;
const MAX_BACKTRACKS: usize = 60;

impl Solver for ProjectedGradient {
    fn info(&self) -> SolverInfo {
        SolverInfo {
            id: "pgd",
            name: "projected gradient",
            version: env!("CARGO_PKG_VERSION"),
            description:
                "projected gradient with Barzilai-Borwein steps for bound-constrained problems (reference solver)",
            capabilities: Capabilities {
                uses: Access::Gradients,
                objective: FnClass::Smooth,
                constraints: FnClass::Smooth,
                integer: false,
                categorical: false,
                finite_bounds: false,
                inequalities: false,
                equalities: false,
                ask_tell: false,
            },
        }
    }

    fn options(&self) -> Vec<OptionSpec> {
        vec![common::max_iterations(1000), common::tol(1e-6)]
    }

    fn solve(&self, problem: ProblemRef<'_>, options: &ResolvedOptions, ctx: &mut Context<'_>) -> Solution {
        let info = self.info();
        let invalid = |m: &str| Solution::new(info, Termination::InvalidProblem, PointKind::NoPoint).with_message(m);
        let ProblemRef::Smooth(p) = problem else { return invalid("pgd needs a smooth oracle") };
        if p.sense() != Sense::Minimize {
            return invalid("solvers receive minimization problems (dispatch through rui-models)");
        }
        if p.num_constraints() > 0 {
            return invalid("pgd handles bound constraints only");
        }
        let tol = options.f64(common::TOL);
        let max_iterations = options.usize(common::MAX_ITERATIONS);
        let vars = p.variables();
        let n = vars.len();
        let lo: Vec<f64> = vars.iter().map(|v| v.lower).collect();
        let hi: Vec<f64> = vars.iter().map(|v| v.upper).collect();
        let project = |i: usize, v: f64| v.max(lo[i]).min(hi[i]);

        let mut x: Vec<f64> = default_start(vars).iter().enumerate().map(|(i, &v)| project(i, v)).collect();
        let mut g = vec![0.0; n];
        let mut evaluations = 1;
        let start_error = |m: &str| {
            let mut s = Solution::new(info, Termination::EvaluationError, PointKind::NoPoint).with_message(m);
            s.stats.evaluations = 1;
            s
        };
        let mut f = match p.objective(&x) {
            Ok(v) if v.is_finite() => v,
            _ => return start_error("the objective cannot be evaluated at the starting point"),
        };
        if p.gradient(&x, &mut g).is_err() || g.iter().any(|v| !v.is_finite()) {
            return start_error("the gradient cannot be evaluated at the starting point");
        }

        ctx.log(1, format!("pgd: n={n}, tol={tol:e}, max_iterations={max_iterations}"));
        if ctx.logs(2) {
            ctx.log(2, format!("{:>5}  {:>23}  {:>9}  {:>9}", "iter", "objective", "proj.grad", "step"));
        }
        let mut alpha = 1.0 / inf_norm(&g).max(1.0);
        let (mut xt, mut gt) = (vec![0.0; n], vec![0.0; n]);
        let mut iteration = 0;
        let termination = loop {
            let pg = (0..n).map(|i| (project(i, x[i] - g[i]) - x[i]).abs()).fold(0.0, f64::max);
            ctx.emit(&Event::Iteration { iteration, objective: f, infeasibility: 0.0, optimality: pg });
            if ctx.logs(2) {
                ctx.log(2, format!("{iteration:>5}  {f:>23.16e}  {pg:>9.2e}  {alpha:>9.2e}"));
            }
            if pg <= tol {
                break Termination::Converged;
            }
            if iteration >= max_iterations {
                break Termination::IterationLimit;
            }
            if let Some(t) = ctx.stop_reason() {
                break t;
            }
            // Armijo backtracking along the projection arc.
            let mut accepted = None;
            let mut a = alpha;
            for _ in 0..MAX_BACKTRACKS {
                for i in 0..n {
                    xt[i] = project(i, x[i] - a * g[i]);
                }
                let slope: f64 = (0..n).map(|i| g[i] * (xt[i] - x[i])).sum();
                if !(slope < 0.0) {
                    break; // the step is numerically zero
                }
                evaluations += 1;
                if let Ok(ft) = p.objective(&xt) {
                    if ft.is_finite() && ft <= f + ARMIJO * slope {
                        accepted = Some((a, ft));
                        break;
                    }
                }
                a *= 0.5;
            }
            let Some((a, ft)) = accepted else { break Termination::SlowProgress };
            if p.gradient(&xt, &mut gt).is_err() || gt.iter().any(|v| !v.is_finite()) {
                break Termination::EvaluationError;
            }
            let (mut ss, mut sy) = (0.0, 0.0);
            for i in 0..n {
                let s = xt[i] - x[i];
                ss += s * s;
                sy += s * (gt[i] - g[i]);
            }
            alpha = if sy > 0.0 { (ss / sy).clamp(1e-12, 1e12) } else { (2.0 * a).min(1e12) };
            std::mem::swap(&mut x, &mut xt);
            std::mem::swap(&mut g, &mut gt);
            f = ft;
            iteration += 1;
        };

        let mut s = Solution::new(info, termination, PointKind::LastIterate);
        if termination == Termination::Converged {
            // A component within `tol` of a bound that the gradient pushes against is active there.
            let mut duals = Duals { constraints: Vec::new(), lower: vec![0.0; n], upper: vec![0.0; n] };
            for i in 0..n {
                if g[i] > 0.0 && x[i] - lo[i] <= tol {
                    duals.lower[i] = g[i];
                } else if g[i] < 0.0 && hi[i] - x[i] <= tol {
                    duals.upper[i] = -g[i];
                }
            }
            match kkt_check(p, &x, &duals) {
                Ok(report) if report.within(tol) => {
                    s.point = PointKind::KktPoint { verified: true };
                    s.duals = Some(duals);
                }
                Ok(report) => {
                    s.termination = Termination::CheckFailed;
                    s.message = format!("{}: {report:?}", Termination::CheckFailed);
                }
                Err(e) => {
                    s.termination = Termination::CheckFailed;
                    s.message = format!("{}: {e}", Termination::CheckFailed);
                }
            }
        }
        s.x = x;
        s.objective = Some(f);
        s.stats.iterations = iteration;
        s.stats.evaluations = evaluations;
        s.stats.elapsed = ctx.elapsed();
        s
    }
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}
