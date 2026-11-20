//! Problems as solvers receive them.
//!
//! A problem is a list of [`Variable`]s, `m` constraints `g_L <= g(x) <= g_U`, an optimization
//! [`Sense`], and one of two oracles:
//!
//! * [`SmoothOracle`]: values and derivatives, with sparsity structure separated from values (the
//!   shape of `rui_nlp::NlpProblem` and of Ipopt's TNLP);
//! * [`BlackBoxOracle`]: one evaluation returns the objective and all constraint values, and may fail.
//!
//! Conventions shared by every crate of the workspace:
//!
//! * indices are 0-based;
//! * an absent bound is `±f64::INFINITY`;
//! * a point is a `Vec<f64>` with one entry per variable; integer variables hold integral values,
//!   categorical variables hold the index of their level;
//! * Jacobian triplets are `(constraint, variable)`; duplicates are summed;
//! * the Hessian of the Lagrangian `obj_factor ∇²f + Σ λ_i ∇²g_i` is given as one triangle (either);
//!   duplicates are summed.

use std::fmt;

/// Optimization direction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Sense {
    #[default]
    Minimize,
    Maximize,
}

impl Sense {
    /// `σ = +1` for minimization, `-1` for maximization: the minimized function is `σ f`.
    pub fn sign(self) -> f64 {
        match self {
            Sense::Minimize => 1.0,
            Sense::Maximize => -1.0,
        }
    }
}

impl fmt::Display for Sense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Sense::Minimize => "minimize",
            Sense::Maximize => "maximize",
        })
    }
}

/// Type of a variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VarKind {
    Continuous,
    Integer,
    /// One of the listed levels; the variable's value is the index of the level (`0..levels.len()`).
    Categorical(Vec<String>),
}

/// A decision variable.
#[derive(Clone, Debug, PartialEq)]
pub struct Variable {
    /// Name for logs and messages (may be empty).
    pub name: String,
    pub kind: VarKind,
    /// Bounds; `±f64::INFINITY` when absent. Integer bounds are integral; categorical bounds are
    /// `0` and `levels - 1`.
    pub lower: f64,
    pub upper: f64,
    /// Starting value or guess; solvers that need one use [`default_start`] when it is `None`.
    pub start: Option<f64>,
    /// Search on a logarithmic scale (used by black-box solvers); requires `lower > 0`.
    pub log_scale: bool,
}

impl Variable {
    pub fn continuous(name: impl Into<String>, lower: f64, upper: f64) -> Self {
        Self { name: name.into(), kind: VarKind::Continuous, lower, upper, start: None, log_scale: false }
    }

    /// A continuous variable without bounds.
    pub fn free(name: impl Into<String>) -> Self {
        Self::continuous(name, f64::NEG_INFINITY, f64::INFINITY)
    }

    pub fn integer(name: impl Into<String>, lower: f64, upper: f64) -> Self {
        Self { name: name.into(), kind: VarKind::Integer, lower, upper, start: None, log_scale: false }
    }

    pub fn categorical<S: AsRef<str>>(name: impl Into<String>, levels: &[S]) -> Self {
        let levels: Vec<String> = levels.iter().map(|s| s.as_ref().to_string()).collect();
        let upper = levels.len() as f64 - 1.0;
        Self { name: name.into(), kind: VarKind::Categorical(levels), lower: 0.0, upper, start: None, log_scale: false }
    }

    pub fn with_start(mut self, start: f64) -> Self {
        self.start = Some(start);
        self
    }

    /// Search this variable on a logarithmic scale.
    pub fn log(mut self) -> Self {
        self.log_scale = true;
        self
    }

    pub fn is_discrete(&self) -> bool {
        !matches!(self.kind, VarKind::Continuous)
    }

    pub fn is_bounded(&self) -> bool {
        self.lower.is_finite() && self.upper.is_finite()
    }
}

/// What is known about a function, from most to least structured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FnClass {
    Linear,
    Quadratic,
    /// Twice continuously differentiable, otherwise unknown.
    Smooth,
    /// Only values can be computed.
    BlackBox,
}

impl fmt::Display for FnClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            FnClass::Linear => "linear",
            FnClass::Quadratic => "quadratic",
            FnClass::Smooth => "smooth nonlinear",
            FnClass::BlackBox => "black-box",
        })
    }
}

/// Information an oracle provides, or a solver needs, ordered from least to most.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Access {
    /// Function values only.
    Values,
    /// Values and first derivatives.
    Gradients,
    /// Values, first derivatives and the Hessian of the Lagrangian.
    Hessians,
}

impl fmt::Display for Access {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Access::Values => "function values only",
            Access::Gradients => "first derivatives",
            Access::Hessians => "first and second derivatives",
        })
    }
}

/// Error returned by an evaluation callback, e.g. because `x` is outside the function's domain.
/// Solvers treat it as recoverable where they can (shorter step, failed evaluation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalError {
    pub message: String,
}

impl EvalError {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "evaluation error: {}", self.message)
    }
}

impl std::error::Error for EvalError {}

/// Result of one black-box evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// The objective value and the values of all constraint functions `g(x)` (empty without constraints).
    Value { objective: f64, constraints: Vec<f64> },
    /// The evaluation failed (crash, simulation did not converge, hidden constraint).
    Failed,
}

impl Outcome {
    /// Objective value of a problem without constraints.
    pub fn value(objective: f64) -> Self {
        Outcome::Value { objective, constraints: Vec::new() }
    }

    pub fn constrained(objective: f64, constraints: Vec<f64>) -> Self {
        Outcome::Value { objective, constraints }
    }

    pub fn objective(&self) -> Option<f64> {
        match self {
            Outcome::Value { objective, .. } => Some(*objective),
            Outcome::Failed => None,
        }
    }
}

/// Classes of the objective and of each constraint, when known (algebraic front ends know them).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionClasses {
    pub objective: FnClass,
    /// One entry per constraint.
    pub constraints: Vec<FnClass>,
}

/// What every problem has: variables, constraint bounds and a sense.
pub trait Problem {
    /// Human-readable name (logs, messages).
    fn name(&self) -> String {
        "unnamed".to_string()
    }
    fn variables(&self) -> &[Variable];
    /// Number of constraints `m`.
    fn num_constraints(&self) -> usize {
        0
    }
    /// Fills `g_L` and `g_U` (length `m`). An equality constraint has `g_L == g_U`.
    fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
        let _ = (lower, upper);
    }
    fn sense(&self) -> Sense {
        Sense::Minimize
    }
}

/// A problem whose functions are smooth and can be differentiated. Structure methods are called once
/// per solve and must not change afterwards; value methods fill arrays in the structure's order.
pub trait SmoothOracle: Problem {
    /// Derivatives this oracle provides: [`Access::Gradients`] (the default) or [`Access::Hessians`]
    /// (then [`hessian_structure`](Self::hessian_structure) and
    /// [`hessian_values`](Self::hessian_values) must be implemented).
    fn derivatives(&self) -> Access {
        Access::Gradients
    }
    /// Classes of the functions, when known; `None` means all are [`FnClass::Smooth`].
    fn function_classes(&self) -> Option<FunctionClasses> {
        None
    }

    fn objective(&self, x: &[f64]) -> Result<f64, EvalError>;
    fn gradient(&self, x: &[f64], grad: &mut [f64]) -> Result<(), EvalError>;

    fn constraints(&self, x: &[f64], g: &mut [f64]) -> Result<(), EvalError> {
        let _ = (x, g);
        Ok(())
    }
    /// Jacobian sparsity: `(constraint, variable)` pairs.
    fn jacobian_structure(&self) -> Vec<(usize, usize)> {
        Vec::new()
    }
    /// Jacobian values in the order of [`jacobian_structure`](Self::jacobian_structure).
    fn jacobian_values(&self, x: &[f64], values: &mut [f64]) -> Result<(), EvalError> {
        let _ = (x, values);
        Ok(())
    }
    /// Sparsity of the Hessian of the Lagrangian (one triangle).
    fn hessian_structure(&self) -> Vec<(usize, usize)> {
        Vec::new()
    }
    /// Values of `obj_factor ∇²f(x) + Σ_i lambda_i ∇²g_i(x)` in the order of
    /// [`hessian_structure`](Self::hessian_structure).
    fn hessian_values(&self, x: &[f64], obj_factor: f64, lambda: &[f64], values: &mut [f64]) -> Result<(), EvalError> {
        let _ = (x, obj_factor, lambda, values);
        Err(EvalError::new("this problem does not provide second derivatives"))
    }
}

/// A problem whose functions can only be evaluated. One evaluation computes the objective and all
/// constraints (one simulation run produces all outputs).
pub trait BlackBoxOracle: Problem {
    fn evaluate(&self, x: &[f64]) -> Outcome;
    /// Whether repeated evaluations at the same point can give different values.
    fn noisy(&self) -> bool {
        false
    }
}

/// A problem handed to [`Solver::solve`](crate::Solver::solve).
#[derive(Clone, Copy)]
pub enum ProblemRef<'a> {
    Smooth(&'a dyn SmoothOracle),
    BlackBox(&'a dyn BlackBoxOracle),
}

impl<'a> ProblemRef<'a> {
    pub fn name(self) -> String {
        match self {
            ProblemRef::Smooth(p) => p.name(),
            ProblemRef::BlackBox(p) => p.name(),
        }
    }

    pub fn variables(self) -> &'a [Variable] {
        match self {
            ProblemRef::Smooth(p) => p.variables(),
            ProblemRef::BlackBox(p) => p.variables(),
        }
    }

    pub fn num_constraints(self) -> usize {
        match self {
            ProblemRef::Smooth(p) => p.num_constraints(),
            ProblemRef::BlackBox(p) => p.num_constraints(),
        }
    }

    /// `(g_L, g_U)`; see [`constraint_bound_vectors`].
    pub fn constraint_bounds(self) -> (Vec<f64>, Vec<f64>) {
        match self {
            ProblemRef::Smooth(p) => constraint_bound_vectors(p),
            ProblemRef::BlackBox(p) => constraint_bound_vectors(p),
        }
    }

    pub fn sense(self) -> Sense {
        match self {
            ProblemRef::Smooth(p) => p.sense(),
            ProblemRef::BlackBox(p) => p.sense(),
        }
    }
}

/// `(g_L, g_U)` of a problem. Entries the problem does not fill stay NaN, which validation rejects.
pub fn constraint_bound_vectors<P: Problem + ?Sized>(problem: &P) -> (Vec<f64>, Vec<f64>) {
    let m = problem.num_constraints();
    let mut lower = vec![f64::NAN; m];
    let mut upper = vec![f64::NAN; m];
    problem.constraint_bounds(&mut lower, &mut upper);
    (lower, upper)
}

/// The variables' start values, or for variables without one the point of their bounds closest to 0.
pub fn default_start(variables: &[Variable]) -> Vec<f64> {
    variables.iter().map(|v| v.start.unwrap_or_else(|| 0.0f64.max(v.lower).min(v.upper))).collect()
}

/// Largest violation `max_i max(g_L,i − g_i, g_i − g_U,i, 0)`; NaN values count as infinitely violated.
pub fn constraint_violation(g: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for ((&gi, &l), &u) in g.iter().zip(lower).zip(upper) {
        let v = (l - gi).max(gi - u);
        if gi.is_nan() {
            return f64::INFINITY;
        }
        if v > worst {
            worst = v;
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_start_projects_zero_onto_the_bounds() {
        let vars = [
            Variable::continuous("a", 1.0, 2.0),
            Variable::continuous("b", -3.0, -1.0),
            Variable::free("c"),
            Variable::integer("d", -2.0, 5.0).with_start(4.0),
            Variable::categorical("e", &["x", "y"]),
        ];
        assert_eq!(default_start(&vars), vec![1.0, -1.0, 0.0, 4.0, 0.0]);
    }

    #[test]
    fn violation_is_a_max_norm_and_nan_is_infinite() {
        let (l, u) = ([0.0, f64::NEG_INFINITY, 1.0], [1.0, 2.0, 1.0]);
        assert_eq!(constraint_violation(&[0.5, 1.0, 1.0], &l, &u), 0.0);
        assert_eq!(constraint_violation(&[-0.25, 3.0, 1.5], &l, &u), 1.0);
        assert_eq!(constraint_violation(&[0.5, f64::NAN, 1.0], &l, &u), f64::INFINITY);
    }

    #[test]
    fn categorical_bounds_follow_the_levels() {
        let v = Variable::categorical("act", &["relu", "tanh", "gelu"]);
        assert_eq!((v.lower, v.upper), (0.0, 2.0));
        assert!(v.is_discrete() && v.is_bounded());
    }
}
