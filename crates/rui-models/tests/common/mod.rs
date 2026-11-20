//! Mock solvers and test problems shared by the dispatch tests.
#![allow(dead_code)]

use rui_api::{
    default_start, Access, AskTell, BlackBoxOracle, Capabilities, Context, Error, EvalError, Event, FnClass,
    FunctionClasses, OptionSpec, Outcome, PointKind, Problem, ProblemRef, ResolvedOptions, Sense, SmoothOracle,
    Solution, Solver, SolverInfo, Termination, Variable,
};
use std::sync::{Arc, Mutex};

/// Capabilities the rui-nlp plugin declares (docs/design.md §2): continuous variables, any bounds,
/// equality and inequality constraints, second derivatives required.
pub fn nlp_caps() -> Capabilities {
    Capabilities {
        uses: Access::Hessians,
        objective: FnClass::Smooth,
        constraints: FnClass::Smooth,
        integer: false,
        categorical: false,
        finite_bounds: false,
        inequalities: true,
        equalities: true,
        ask_tell: false,
    }
}

/// Capabilities the rui-bo plugin declares: values only, mixed variables with finite bounds,
/// inequality constraints (c(x) <= 0), ask/tell.
pub fn bo_caps() -> Capabilities {
    Capabilities {
        uses: Access::Values,
        objective: FnClass::BlackBox,
        constraints: FnClass::BlackBox,
        integer: true,
        categorical: true,
        finite_bounds: true,
        inequalities: true,
        equalities: false,
        ask_tell: true,
    }
}

/// What a mock solver received.
#[derive(Clone, Debug, PartialEq)]
pub struct Seen {
    pub black_box: bool,
    pub sense: Sense,
    /// Objective at the default start, as the solver sees it.
    pub objective: Option<f64>,
    pub options: ResolvedOptions,
}

pub struct Mock {
    pub info: SolverInfo,
    pub options: Vec<OptionSpec>,
    pub seen: Arc<Mutex<Vec<Seen>>>,
}

impl Mock {
    pub fn new(id: &'static str, capabilities: Capabilities) -> (Mock, Arc<Mutex<Vec<Seen>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let info = SolverInfo { id, name: id, version: "9.9.9", description: "mock", capabilities };
        (Mock { info, options: Vec::new(), seen: seen.clone() }, seen)
    }

    pub fn with_options(mut self, options: Vec<OptionSpec>) -> Self {
        self.options = options;
        self
    }
}

impl Solver for Mock {
    fn info(&self) -> SolverInfo {
        self.info
    }
    fn options(&self) -> Vec<OptionSpec> {
        self.options.clone()
    }
    fn solve(&self, problem: ProblemRef<'_>, options: &ResolvedOptions, ctx: &mut Context<'_>) -> Solution {
        let x = default_start(problem.variables());
        let (black_box, objective) = match problem {
            ProblemRef::Smooth(p) => (false, p.objective(&x).ok()),
            ProblemRef::BlackBox(p) => (true, p.evaluate(&x).objective()),
        };
        self.seen.lock().unwrap().push(Seen { black_box, sense: problem.sense(), objective, options: options.clone() });
        ctx.emit(&Event::Iteration {
            iteration: 0,
            objective: objective.unwrap_or(f64::NAN),
            infeasibility: 0.0,
            optimality: 0.0,
        });
        let mut s = Solution::new(self.info, Termination::Converged, PointKind::KktPoint { verified: false });
        s.x = x;
        s.objective = objective;
        s
    }
    fn session(&self, _space: &dyn Problem, _options: &ResolvedOptions) -> Result<Box<dyn AskTell>, Error> {
        Ok(Box::new(MockSession { told: Vec::new() }))
    }
}

/// Remembers what it was told; asks for the origin.
pub struct MockSession {
    pub told: Vec<(Vec<f64>, Outcome)>,
}

impl AskTell for MockSession {
    fn ask(&mut self, q: usize) -> Result<Vec<Vec<f64>>, Error> {
        Ok(vec![vec![0.0]; q])
    }
    fn tell(&mut self, x: &[f64], outcome: Outcome) -> Result<(), Error> {
        self.told.push((x.to_vec(), outcome));
        Ok(())
    }
    fn mark_pending(&mut self, _x: &[f64]) -> Result<(), Error> {
        Ok(())
    }
    fn solution(&mut self) -> Solution {
        let info = SolverInfo { id: "bo", name: "bo", version: "9.9.9", description: "mock", capabilities: bo_caps() };
        let mut s = Solution::new(info, Termination::AskTell, PointKind::BestObserved { feasible: true });
        if let Some((x, Outcome::Value { objective, .. })) = self.told.last() {
            s.x = x.clone();
            s.objective = Some(*objective);
        }
        s
    }
    fn n_evaluations(&self) -> usize {
        self.told.len()
    }
}

/// `f(x) = Σ (x_j − 1)²`; constraint `i` is `g_i(x) = Σ x_j`.
pub struct Smooth {
    pub vars: Vec<Variable>,
    /// `(g_L, g_U)` per constraint.
    pub bounds: Vec<(f64, f64)>,
    pub classes: Option<FunctionClasses>,
    pub derivatives: Access,
    pub sense: Sense,
}

impl Smooth {
    pub fn new(vars: Vec<Variable>) -> Self {
        Smooth { vars, bounds: Vec::new(), classes: None, derivatives: Access::Hessians, sense: Sense::Minimize }
    }
    pub fn bounded(n: usize) -> Self {
        Self::new((0..n).map(|j| Variable::continuous(format!("x{j}"), -2.0, 3.0).with_start(0.5)).collect())
    }
    pub fn constraint(mut self, lower: f64, upper: f64) -> Self {
        self.bounds.push((lower, upper));
        self
    }
}

impl Problem for Smooth {
    fn name(&self) -> String {
        "smooth".to_string()
    }
    fn variables(&self) -> &[Variable] {
        &self.vars
    }
    fn num_constraints(&self) -> usize {
        self.bounds.len()
    }
    fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
        for (i, (l, u)) in self.bounds.iter().enumerate() {
            lower[i] = *l;
            upper[i] = *u;
        }
    }
    fn sense(&self) -> Sense {
        self.sense
    }
}

impl SmoothOracle for Smooth {
    fn derivatives(&self) -> Access {
        self.derivatives
    }
    fn function_classes(&self) -> Option<FunctionClasses> {
        self.classes.clone()
    }
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        Ok(x.iter().map(|v| (v - 1.0) * (v - 1.0)).sum())
    }
    fn gradient(&self, x: &[f64], grad: &mut [f64]) -> Result<(), EvalError> {
        for (g, v) in grad.iter_mut().zip(x) {
            *g = 2.0 * (v - 1.0);
        }
        Ok(())
    }
    fn constraints(&self, x: &[f64], g: &mut [f64]) -> Result<(), EvalError> {
        g.fill(x.iter().sum());
        Ok(())
    }
}

/// `f(x) = Σ x_j`; every constraint value is `x_0`.
pub struct BlackBox {
    pub vars: Vec<Variable>,
    pub bounds: Vec<(f64, f64)>,
    pub sense: Sense,
}

impl BlackBox {
    pub fn new(vars: Vec<Variable>) -> Self {
        BlackBox { vars, bounds: Vec::new(), sense: Sense::Minimize }
    }
    pub fn constraint(mut self, lower: f64, upper: f64) -> Self {
        self.bounds.push((lower, upper));
        self
    }
}

impl Problem for BlackBox {
    fn variables(&self) -> &[Variable] {
        &self.vars
    }
    fn num_constraints(&self) -> usize {
        self.bounds.len()
    }
    fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
        for (i, (l, u)) in self.bounds.iter().enumerate() {
            lower[i] = *l;
            upper[i] = *u;
        }
    }
    fn sense(&self) -> Sense {
        self.sense
    }
}

impl BlackBoxOracle for BlackBox {
    fn evaluate(&self, x: &[f64]) -> Outcome {
        Outcome::constrained(x.iter().sum(), vec![x[0]; self.bounds.len()])
    }
}

pub fn mixed_box() -> Vec<Variable> {
    vec![
        Variable::continuous("lr", 1e-4, 1e-1).log(),
        Variable::integer("layers", 1.0, 8.0),
        Variable::categorical("activation", &["relu", "tanh", "gelu"]),
    ]
}
