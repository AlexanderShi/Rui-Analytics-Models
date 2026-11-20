//! Random search: every point is drawn uniformly from the box (log-uniformly on logarithmic scales).
//!
//! The template for black-box plugins: [`RandomSearch::solve`] is written on top of the ask/tell
//! session, the way a model-based solver's `minimize` driver is.

use crate::rng::Rng;
use rui_api::{
    common, constraint_bound_vectors, constraint_violation, Access, AskTell, Capabilities, Context, Error, Event,
    FnClass, OptionSpec, Outcome, PointKind, Problem, ProblemRef, ResolvedOptions, Sense, Solution, Solver, SolverInfo,
    Termination, VarKind, Variable,
};
use std::time::Instant;

/// Random search (`method=random`).
pub struct RandomSearch;

impl Solver for RandomSearch {
    fn info(&self) -> SolverInfo {
        SolverInfo {
            id: "random",
            name: "random search",
            version: env!("CARGO_PKG_VERSION"),
            description: "uniform sampling of the search space (reference solver)",
            capabilities: Capabilities {
                uses: Access::Values,
                objective: FnClass::BlackBox,
                constraints: FnClass::BlackBox,
                integer: true,
                categorical: true,
                finite_bounds: true,
                inequalities: true,
                // Uniform samples satisfy an equality constraint with probability zero.
                equalities: false,
                ask_tell: true,
            },
        }
    }

    fn options(&self) -> Vec<OptionSpec> {
        vec![common::max_evaluations(100), common::seed(0)]
    }

    fn solve(&self, problem: ProblemRef<'_>, options: &ResolvedOptions, ctx: &mut Context<'_>) -> Solution {
        let ProblemRef::BlackBox(oracle) = problem else {
            return invalid(self.info(), "random search needs a black-box oracle");
        };
        if oracle.sense() != Sense::Minimize {
            return invalid(self.info(), "solvers receive minimization problems (dispatch through rui-models)");
        }
        let budget = options.usize(common::MAX_EVALUATIONS);
        let mut session = RandomSession::new(self.info(), oracle, options);
        ctx.log(1, format!("random: budget {budget} evaluations, seed {}", options.i64(common::SEED)));
        let termination = loop {
            if session.evaluations >= budget {
                break Termination::EvaluationLimit;
            }
            if let Some(t) = ctx.stop_reason() {
                break t;
            }
            let x = session.sample();
            let outcome = sanitize(oracle.evaluate(&x), session.lower.len());
            session.record(&x, &outcome);
            let best = session.best.as_ref().map(|b| b.objective);
            ctx.emit(&Event::Evaluation { index: session.evaluations - 1, x: &x, outcome: &outcome, best });
            if ctx.logs(2) {
                let value = match &outcome {
                    Outcome::Value { objective, .. } => format!("{objective:>14.6e}"),
                    Outcome::Failed => format!("{:>14}", "failed"),
                };
                let best = best.map_or_else(|| "-".to_string(), |b| format!("{b:.6e}"));
                ctx.log(2, format!("{:>6}  {value}  best {best}", session.evaluations));
            }
        };
        let mut s = session.solution();
        s.termination = termination;
        s.message = termination.description().to_string();
        s.stats.elapsed = ctx.elapsed();
        s
    }

    fn session(&self, space: &dyn Problem, options: &ResolvedOptions) -> Result<Box<dyn AskTell>, Error> {
        Ok(Box::new(RandomSession::new(self.info(), space, options)))
    }
}

fn invalid(info: SolverInfo, message: &str) -> Solution {
    Solution::new(info, Termination::InvalidProblem, PointKind::NoPoint).with_message(message)
}

/// A value that cannot be used (non-finite, wrong number of constraint values) counts as a failure.
fn sanitize(outcome: Outcome, m: usize) -> Outcome {
    match outcome {
        Outcome::Value { objective, constraints }
            if objective.is_finite() && constraints.len() == m && constraints.iter().all(|c| c.is_finite()) =>
        {
            Outcome::Value { objective, constraints }
        }
        _ => Outcome::Failed,
    }
}

struct Record {
    x: Vec<f64>,
    objective: f64,
    constraints: Vec<f64>,
}

struct RandomSession {
    info: SolverInfo,
    variables: Vec<Variable>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    rng: Rng,
    start: Instant,
    evaluations: usize,
    failures: usize,
    pending: Vec<Vec<f64>>,
    /// Best feasible point.
    best: Option<Record>,
    /// Least infeasible point (ties: lower objective), used while no point is feasible.
    least_infeasible: Option<(f64, Record)>,
}

impl RandomSession {
    fn new<P: Problem + ?Sized>(info: SolverInfo, space: &P, options: &ResolvedOptions) -> Self {
        let (lower, upper) = constraint_bound_vectors(space);
        RandomSession {
            info,
            variables: space.variables().to_vec(),
            lower,
            upper,
            rng: Rng::new(options.i64(common::SEED) as u64),
            start: Instant::now(),
            evaluations: 0,
            failures: 0,
            pending: Vec::new(),
            best: None,
            least_infeasible: None,
        }
    }

    fn sample(&mut self) -> Vec<f64> {
        let rng = &mut self.rng;
        self.variables
            .iter()
            .map(|v| {
                let (lo, hi) = (v.lower, v.upper);
                match &v.kind {
                    VarKind::Categorical(levels) => rng.below(levels.len() as u64) as f64,
                    VarKind::Integer => {
                        // Uniform over the integers, or log-uniform over [lo, hi + 1) then rounded down.
                        let x = if v.log_scale {
                            (lo.ln() + rng.uniform() * ((hi + 1.0).ln() - lo.ln())).exp()
                        } else {
                            lo + rng.uniform() * (hi - lo + 1.0)
                        };
                        x.floor().max(lo).min(hi)
                    }
                    VarKind::Continuous => {
                        let x = if v.log_scale {
                            (lo.ln() + rng.uniform() * (hi.ln() - lo.ln())).exp()
                        } else {
                            lo + rng.uniform() * (hi - lo)
                        };
                        x.max(lo).min(hi)
                    }
                }
            })
            .collect()
    }

    fn check_point(&self, x: &[f64]) -> Result<(), Error> {
        if x.len() != self.variables.len() {
            return Err(Error::InvalidPoint(format!("{} values for {} variables", x.len(), self.variables.len())));
        }
        for (v, &xi) in self.variables.iter().zip(x) {
            if !(xi >= v.lower && xi <= v.upper) || (v.is_discrete() && xi.fract() != 0.0) {
                return Err(Error::InvalidPoint(format!("value {xi:?} of `{}` is outside its domain", v.name)));
            }
        }
        Ok(())
    }

    fn record(&mut self, x: &[f64], outcome: &Outcome) {
        self.evaluations += 1;
        if let Some(i) = self.pending.iter().position(|p| p == x) {
            self.pending.swap_remove(i);
        }
        let Outcome::Value { objective, constraints } = outcome else {
            self.failures += 1;
            return;
        };
        let record = || Record { x: x.to_vec(), objective: *objective, constraints: constraints.clone() };
        let violation = constraint_violation(constraints, &self.lower, &self.upper);
        if violation <= 0.0 {
            if self.best.as_ref().is_none_or(|b| *objective < b.objective) {
                self.best = Some(record());
            }
        } else if self
            .least_infeasible
            .as_ref()
            .is_none_or(|(v, b)| violation < *v || (violation == *v && *objective < b.objective))
        {
            self.least_infeasible = Some((violation, record()));
        }
    }
}

impl AskTell for RandomSession {
    fn ask(&mut self, q: usize) -> Result<Vec<Vec<f64>>, Error> {
        let points: Vec<Vec<f64>> = (0..q).map(|_| self.sample()).collect();
        self.pending.extend(points.iter().cloned());
        Ok(points)
    }

    fn tell(&mut self, x: &[f64], outcome: Outcome) -> Result<(), Error> {
        self.check_point(x)?;
        if let Outcome::Value { constraints, .. } = &outcome {
            if constraints.len() != self.lower.len() {
                return Err(Error::InvalidValue(format!(
                    "{} constraint values for {} constraints",
                    constraints.len(),
                    self.lower.len()
                )));
            }
        }
        self.record(x, &outcome);
        Ok(())
    }

    fn mark_pending(&mut self, x: &[f64]) -> Result<(), Error> {
        // Random search ignores pending points when sampling; it only keeps the record.
        self.check_point(x)?;
        self.pending.push(x.to_vec());
        Ok(())
    }

    fn solution(&mut self) -> Solution {
        let (record, feasible) = match (&self.best, &self.least_infeasible) {
            (Some(b), _) => (Some(b), true),
            (None, Some((_, b))) => (Some(b), false),
            (None, None) => (None, false),
        };
        let point = if record.is_some() { PointKind::BestObserved { feasible } } else { PointKind::NoPoint };
        let mut s = Solution::new(self.info, Termination::AskTell, point);
        if let Some(r) = record {
            s.x = r.x.clone();
            s.objective = Some(r.objective);
            s.constraints = r.constraints.clone();
        }
        s.stats.evaluations = self.evaluations;
        s.stats.elapsed = self.start.elapsed().as_secs_f64();
        if self.failures > 0 {
            s.notes.push(format!("{} of {} evaluations failed", self.failures, self.evaluations));
        }
        s
    }

    fn n_evaluations(&self) -> usize {
        self.evaluations
    }
}
