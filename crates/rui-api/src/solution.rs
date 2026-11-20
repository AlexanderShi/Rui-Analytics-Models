//! Results.
//!
//! A [`Solution`] answers two separate questions: why the solver stopped ([`Termination`]) and what
//! the returned point is ([`PointKind`]). They are separate because "success" means different things
//! for different solvers: for a black-box solver, using up the evaluation budget is the normal end and
//! the point is the best one observed; for a derivative-based solver, the normal end is convergence to
//! a first-order point. Neither kind of point is a certified global optimum, and no solution claims one.

use crate::signature::Capabilities;
use std::any::Any;
use std::fmt;

/// Why the solver stopped. Solver plugins map every native status to exactly one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Termination {
    Converged,
    ConvergedAcceptable,
    LocallyInfeasible,
    Diverging,
    IterationLimit,
    EvaluationLimit,
    TimeLimit,
    TargetReached,
    SpaceExhausted,
    /// The solution of an ask/tell session, which the caller drives and stops.
    AskTell,
    SlowProgress,
    SolverFailure,
    NumericalFailure,
    EvaluationError,
    CheckFailed,
    Interrupted,
    InvalidProblem,
}

impl Termination {
    pub fn description(self) -> &'static str {
        match self {
            Termination::Converged => "converged to the requested tolerance",
            Termination::ConvergedAcceptable => "converged to the acceptable (looser) tolerance",
            Termination::LocallyInfeasible => {
                "converged to a local minimizer of the infeasibility; this is not a proof of infeasibility"
            }
            Termination::Diverging => "iterates diverged (the problem may be unbounded)",
            Termination::IterationLimit => "iteration limit reached",
            Termination::EvaluationLimit => "evaluation budget used up",
            Termination::TimeLimit => "time limit reached",
            Termination::TargetReached => "target objective value reached",
            Termination::SpaceExhausted => "every point of the finite search space has been evaluated",
            Termination::AskTell => "ask/tell session: the caller decides when to stop",
            Termination::SlowProgress => "no further progress possible (step too small)",
            Termination::SolverFailure => "the algorithm could not continue",
            Termination::NumericalFailure => "numerical failure",
            Termination::EvaluationError => "a function evaluation failed and could not be avoided",
            Termination::CheckFailed => "the convergence test passed but the independent check did not",
            Termination::Interrupted => "interrupted",
            Termination::InvalidProblem => "the solver rejected the problem as invalid",
        }
    }
}

impl fmt::Display for Termination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.description())
    }
}

/// What the returned point is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointKind {
    /// Satisfies first-order optimality (KKT) conditions within tolerance. For nonconvex problems this is
    /// a local stationary point, not a certified minimum. `verified`: confirmed by a residual check
    /// computed only from the problem's callbacks and original bounds.
    KktPoint { verified: bool },
    /// The best of the evaluated points; no optimality certificate. `feasible`: it satisfies the
    /// constraints (otherwise no evaluated point did, and this is the least infeasible one).
    BestObserved { feasible: bool },
    /// A stationary point of the constraint violation that is not feasible.
    InfeasibleStationary,
    /// The iterate at which the solver stopped, without any guarantee.
    LastIterate,
    /// No usable point.
    NoPoint,
}

impl fmt::Display for PointKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            PointKind::KktPoint { verified: true } => "KKT point (verified independently)",
            PointKind::KktPoint { verified: false } => "KKT point (not verified independently)",
            PointKind::BestObserved { feasible: true } => "best observed point (feasible)",
            PointKind::BestObserved { feasible: false } => {
                "best observed point (infeasible: no feasible point observed)"
            }
            PointKind::InfeasibleStationary => "stationary point of the infeasibility (not feasible)",
            PointKind::LastIterate => "last iterate (no guarantee)",
            PointKind::NoPoint => "no point",
        })
    }
}

/// Multipliers. With `σ = +1` for minimization and `-1` for maximization they satisfy
/// `σ ∇f + Jᵀλ − z_L + z_U = 0`, `z_L, z_U >= 0` (see [`kkt_check`](crate::kkt_check)): the duals of
/// the minimized function `σ f`, in Ipopt's convention. Adapters convert to other tools' conventions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Duals {
    /// `λ`, one per constraint.
    pub constraints: Vec<f64>,
    /// `z_L`, one per variable.
    pub lower: Vec<f64>,
    /// `z_U`, one per variable.
    pub upper: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    pub iterations: usize,
    /// Objective (black box: full) evaluations.
    pub evaluations: usize,
    /// Wall-clock seconds.
    pub elapsed: f64,
}

/// Identity and capabilities of a solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolverInfo {
    /// What users write as `method=`: lowercase, `[a-z0-9_-]`.
    pub id: &'static str,
    pub name: &'static str,
    pub version: &'static str,
    pub description: &'static str,
    pub capabilities: Capabilities,
}

/// Result of a solve.
pub struct Solution {
    pub solver: SolverInfo,
    pub termination: Termination,
    pub point: PointKind,
    pub message: String,
    /// One entry per variable; empty for [`PointKind::NoPoint`].
    pub x: Vec<f64>,
    /// Objective value at `x`, for the problem as stated (maximization problems: the maximized value).
    pub objective: Option<f64>,
    /// `g(x)`; empty when unknown.
    pub constraints: Vec<f64>,
    pub duals: Option<Duals>,
    pub stats: Stats,
    /// Dispatch decision, ignored options and other remarks (Models adds these).
    pub notes: Vec<String>,
    /// The solver's own result (iteration history, all evaluations, ...), for callers that know its type.
    pub native: Option<Box<dyn Any + Send>>,
}

impl Solution {
    pub fn new(solver: SolverInfo, termination: Termination, point: PointKind) -> Self {
        Solution {
            solver,
            termination,
            point,
            message: termination.description().to_string(),
            x: Vec::new(),
            objective: None,
            constraints: Vec::new(),
            duals: None,
            stats: Stats::default(),
            notes: Vec::new(),
            native: None,
        }
    }

    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// A KKT point, or a feasible best observed point.
    pub fn is_success(&self) -> bool {
        matches!(self.point, PointKind::KktPoint { .. } | PointKind::BestObserved { feasible: true })
    }
}

impl fmt::Debug for Solution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Solution")
            .field("solver", &self.solver.id)
            .field("termination", &self.termination)
            .field("point", &self.point)
            .field("message", &self.message)
            .field("x", &self.x)
            .field("objective", &self.objective)
            .field("constraints", &self.constraints)
            .field("duals", &self.duals)
            .field("stats", &self.stats)
            .field("notes", &self.notes)
            .field("native", &self.native.as_ref().map(|_| ".."))
            .finish()
    }
}
