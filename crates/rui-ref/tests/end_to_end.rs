//! The reference solvers driven through rui-models: automatic dispatch, bridges, options, sessions,
//! interruption and the independent KKT check, end to end.

use rui_api::{
    kkt_check, BlackBoxOracle, Error, EvalError, Event, Flow, Observer, Options, Outcome, PointKind, Problem,
    ProblemRef, Sense, Silent, SmoothOracle, Termination, Variable,
};
use rui_models::Registry;
use rui_ref::{ProjectedGradient, RandomSearch};
use std::f64::consts::PI;

fn registry() -> Registry {
    Registry::new().with(RandomSearch).with(ProjectedGradient)
}

// --- smooth problems ----------------------------------------------------------------------------

/// `f = c0 (x0 − a0)² + c1 (x1 − a1)² + q x0 x1` on bounds; optional constraint `x0 + x1 <= 1`.
struct Quadratic {
    vars: Vec<Variable>,
    a: [f64; 2],
    c: [f64; 2],
    q: f64,
    sense: Sense,
    constrained: bool,
}

impl Quadratic {
    fn new(vars: Vec<Variable>, a: [f64; 2], c: [f64; 2], q: f64) -> Self {
        Quadratic { vars, a, c, q, sense: Sense::Minimize, constrained: false }
    }
}

impl Problem for Quadratic {
    fn name(&self) -> String {
        "quadratic".into()
    }
    fn variables(&self) -> &[Variable] {
        &self.vars
    }
    fn num_constraints(&self) -> usize {
        usize::from(self.constrained)
    }
    fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
        if self.constrained {
            lower[0] = f64::NEG_INFINITY;
            upper[0] = 1.0;
        }
    }
    fn sense(&self) -> Sense {
        self.sense
    }
}

impl SmoothOracle for Quadratic {
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        let d = [x[0] - self.a[0], x[1] - self.a[1]];
        Ok(self.c[0] * d[0] * d[0] + self.c[1] * d[1] * d[1] + self.q * x[0] * x[1])
    }
    fn gradient(&self, x: &[f64], grad: &mut [f64]) -> Result<(), EvalError> {
        grad[0] = 2.0 * self.c[0] * (x[0] - self.a[0]) + self.q * x[1];
        grad[1] = 2.0 * self.c[1] * (x[1] - self.a[1]) + self.q * x[0];
        Ok(())
    }
    fn constraints(&self, x: &[f64], g: &mut [f64]) -> Result<(), EvalError> {
        if self.constrained {
            g[0] = x[0] + x[1];
        }
        Ok(())
    }
    fn jacobian_structure(&self) -> Vec<(usize, usize)> {
        if self.constrained {
            vec![(0, 0), (0, 1)]
        } else {
            Vec::new()
        }
    }
    fn jacobian_values(&self, _x: &[f64], values: &mut [f64]) -> Result<(), EvalError> {
        values.fill(1.0);
        Ok(())
    }
}

#[test]
fn bound_constrained_quadratic_goes_to_pgd_and_is_verified() {
    // min (x0 − 3)² + (x1 + 1)² + 0.5 x0 x1 on [0, 2]²: convex, minimizer (2, 0) with
    // ∇f = (−2, 3): z_U,0 = 2 and z_L,1 = 3.
    let p = Quadratic::new(
        vec![Variable::continuous("x0", 0.0, 2.0), Variable::continuous("x1", 0.0, 2.0)],
        [3.0, -1.0],
        [1.0, 1.0],
        0.5,
    );
    let s = registry().solve(ProblemRef::Smooth(&p), &Options::new(), &mut Silent).unwrap();
    assert_eq!(s.solver.id, "pgd");
    // The oracle does not declare its function classes, so it is a general NLP.
    assert_eq!(s.notes[0], format!("rui: NLP (n=2, m=0) -> pgd {} (automatic)", rui_ref_version()));
    assert_eq!((s.termination, s.point), (Termination::Converged, PointKind::KktPoint { verified: true }));
    assert_eq!(s.x, vec![2.0, 0.0]);
    assert_eq!(s.objective, Some(2.0));
    let d = s.duals.as_ref().unwrap();
    assert_eq!((d.lower.clone(), d.upper.clone()), (vec![0.0, 3.0], vec![2.0, 0.0]));
    assert!(kkt_check(&p, &s.x, d).unwrap().within(1e-12));
    assert!(s.is_success());
}

fn rui_ref_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[test]
fn maximization_with_pgd_reports_the_maximized_value_and_checkable_duals() {
    // max −(x0 − 1)² − 2 (x1 − 2)² with x1 <= 1.5: maximizer (1, 1.5), value −0.5.
    let mut p = Quadratic::new(
        vec![Variable::free("x0"), Variable::continuous("x1", f64::NEG_INFINITY, 1.5)],
        [1.0, 2.0],
        [-1.0, -2.0],
        0.0,
    );
    p.sense = Sense::Maximize;
    let s = registry().solve(ProblemRef::Smooth(&p), &Options::new().with("tol", 1e-9), &mut Silent).unwrap();
    assert_eq!(s.point, PointKind::KktPoint { verified: true });
    assert!((s.x[0] - 1.0).abs() < 1e-9 && s.x[1] == 1.5, "{:?}", s.x);
    assert!((s.objective.unwrap() + 0.5).abs() < 1e-12, "{:?}", s.objective);
    // The duals belong to the minimized −f and pass the check against the problem as stated.
    let d = s.duals.as_ref().unwrap();
    assert!((d.upper[1] - 2.0).abs() < 1e-8, "{d:?}");
    assert!(kkt_check(&p, &s.x, d).unwrap().within(1e-8));
}

#[test]
fn pgd_reports_its_iteration_limit_without_claims() {
    // Ill-conditioned: c = (1, 1e4).
    let p = Quadratic::new(vec![Variable::free("x0"), Variable::free("x1")], [1.0, 1.0], [1.0, 1e4], 0.0);
    let s = registry().solve(ProblemRef::Smooth(&p), &Options::new().with("max_iterations", 2), &mut Silent).unwrap();
    assert_eq!((s.termination, s.point), (Termination::IterationLimit, PointKind::LastIterate));
    assert_eq!(s.stats.iterations, 2);
    assert!(s.duals.is_none() && !s.is_success());
}

#[test]
fn constrained_smooth_problem_needs_a_named_black_box_solver() {
    let mut p = Quadratic::new(
        vec![Variable::continuous("x0", 0.0, 2.0), Variable::continuous("x1", 0.0, 2.0)],
        [3.0, 3.0],
        [1.0, 1.0],
        0.0,
    );
    p.constrained = true;
    let r = registry();
    let e = r.solve(ProblemRef::Smooth(&p), &Options::new(), &mut Silent).unwrap_err();
    assert_eq!(
        e.to_string(),
        "no registered solver takes this NLP (n=2, m=1) problem automatically\n  pgd: does not handle inequality \
         constraints (1)\n  random: usable only when named (method=random): ignores the derivatives and treats the \
         problem as a black box"
    );
    let options = Options::parse("method=random max_evaluations=400 seed=1").unwrap();
    let s = r.solve(ProblemRef::Smooth(&p), &options, &mut Silent).unwrap();
    assert_eq!(s.point, PointKind::BestObserved { feasible: true });
    assert!(s.constraints[0] <= 1.0);
    // The constrained minimizer is (0.5, 0.5) with f = 12.5; random search gets close.
    assert!(s.objective.unwrap() < 13.0, "{s:?}");
    assert_eq!(s.stats.evaluations, 400);
}

// --- black boxes --------------------------------------------------------------------------------

struct Branin(Vec<Variable>);

impl Branin {
    fn new() -> Self {
        Branin(vec![Variable::continuous("x1", -5.0, 10.0), Variable::continuous("x2", 0.0, 15.0)])
    }
}

impl Problem for Branin {
    fn variables(&self) -> &[Variable] {
        &self.0
    }
}

impl BlackBoxOracle for Branin {
    fn evaluate(&self, x: &[f64]) -> Outcome {
        let (b, c, t) = (5.1 / (4.0 * PI * PI), 5.0 / PI, 1.0 / (8.0 * PI));
        let y = x[1] - b * x[0] * x[0] + c * x[0] - 6.0;
        Outcome::value(y * y + 10.0 * (1.0 - t) * x[0].cos() + 10.0)
    }
}

#[test]
fn random_search_on_branin_is_reproducible() {
    let r = registry();
    let p = Branin::new();
    let run = |seed: i64| {
        let o = Options::new().with("max_evaluations", 300).with("seed", seed);
        r.solve(ProblemRef::BlackBox(&p), &o, &mut Silent).unwrap()
    };
    let (a, b, c) = (run(4), run(4), run(5));
    assert_eq!(a.solver.id, "random");
    assert_eq!((a.termination, a.point), (Termination::EvaluationLimit, PointKind::BestObserved { feasible: true }));
    assert_eq!(a.x, b.x);
    assert_ne!(a.x, c.x);
    // Global minimum 0.397887; 300 uniform samples land well below 3.
    assert!(a.objective.unwrap() < 3.0 && c.objective.unwrap() < 3.0, "{:?} {:?}", a.objective, c.objective);
    assert_eq!(a.stats.evaluations, 300);
}

/// `(n − 3)² + [c != "b"] + (x − 0.5)²`; fails when `x < 0.1`.
struct Mixed(Vec<Variable>);

impl Problem for Mixed {
    fn variables(&self) -> &[Variable] {
        &self.0
    }
}

impl BlackBoxOracle for Mixed {
    fn evaluate(&self, x: &[f64]) -> Outcome {
        if x[0] < 0.1 {
            return Outcome::Failed;
        }
        assert!(x[1].fract() == 0.0 && (0.0..=6.0).contains(&x[1]) && [0.0, 1.0, 2.0].contains(&x[2]));
        Outcome::value((x[1] - 3.0).powi(2) + f64::from(u8::from(x[2] != 1.0)) + (x[0] - 0.5).powi(2))
    }
}

#[test]
fn random_search_handles_mixed_variables_and_failures() {
    let p = Mixed(vec![
        Variable::continuous("x", 0.0, 1.0),
        Variable::integer("n", 0.0, 6.0),
        Variable::categorical("c", &["a", "b", "c"]),
    ]);
    let s =
        registry().solve(ProblemRef::BlackBox(&p), &Options::new().with("max_evaluations", 400), &mut Silent).unwrap();
    assert_eq!((s.x[1], s.x[2]), (3.0, 1.0), "{s:?}");
    assert!((s.x[0] - 0.5).abs() < 0.2, "{s:?}");
    // About 10% of the samples fail.
    let note = s.notes.iter().find(|n| n.contains("evaluations failed")).expect("failure note");
    let failed: usize = note.split_whitespace().next().unwrap().parse().unwrap();
    assert!((20..=60).contains(&failed), "{note}");
}

struct CountAndStop {
    seen: usize,
    stop_after: usize,
    objectives: Vec<f64>,
    lines: Vec<String>,
}

impl Observer for CountAndStop {
    fn log(&mut self, line: &str) {
        self.lines.push(line.to_string());
    }
    fn event(&mut self, event: &Event<'_>) -> Flow {
        if let Event::Evaluation { outcome, .. } = event {
            self.objectives.extend(outcome.objective());
        }
        self.seen += 1;
        if self.seen >= self.stop_after {
            Flow::Stop
        } else {
            Flow::Continue
        }
    }
}

/// Maximize the Branin function's negation shifted: f = 400 − branin > 0.
struct Upside(Branin);

impl Problem for Upside {
    fn variables(&self) -> &[Variable] {
        self.0.variables()
    }
    fn sense(&self) -> Sense {
        Sense::Maximize
    }
}

impl BlackBoxOracle for Upside {
    fn evaluate(&self, x: &[f64]) -> Outcome {
        Outcome::value(400.0 - self.0.evaluate(x).objective().unwrap())
    }
}

#[test]
fn observers_stop_solves_and_see_the_objective_as_stated() {
    let p = Upside(Branin::new());
    let mut obs = CountAndStop { seen: 0, stop_after: 10, objectives: Vec::new(), lines: Vec::new() };
    let o = Options::new().with("verbosity", 2);
    let s = registry().solve(ProblemRef::BlackBox(&p), &o, &mut obs).unwrap();
    assert_eq!((s.termination, s.stats.evaluations), (Termination::Interrupted, 10));
    assert!(obs.objectives.iter().all(|f| *f > 0.0), "{:?}", obs.objectives);
    let best = obs.objectives.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(s.objective, Some(best));
    assert!(obs.lines[0].starts_with("rui: BLACKBOX (n=2, m=0) -> random"), "{:?}", obs.lines);
    // Header line, dispatch line and 10 evaluation lines (the solver logs the minimized objective).
    assert_eq!(obs.lines.len(), 12, "{:?}", obs.lines);
}

#[test]
fn a_zero_time_limit_stops_before_the_first_evaluation() {
    let s = registry()
        .solve(ProblemRef::BlackBox(&Branin::new()), &Options::new().with("time_limit", 0.0), &mut Silent)
        .unwrap();
    assert_eq!((s.termination, s.point, s.stats.evaluations), (Termination::TimeLimit, PointKind::NoPoint, 0));
    assert!(!s.is_success());
}

// --- ask/tell -----------------------------------------------------------------------------------

/// Two variables, one constraint `c(x) <= 0`.
struct Space(Vec<Variable>, Sense);

impl Problem for Space {
    fn variables(&self) -> &[Variable] {
        &self.0
    }
    fn num_constraints(&self) -> usize {
        1
    }
    fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
        lower[0] = f64::NEG_INFINITY;
        upper[0] = 0.0;
    }
    fn sense(&self) -> Sense {
        self.1
    }
}

#[test]
fn ask_tell_session_with_parallel_points_failures_and_maximization() {
    let space = Space(
        vec![Variable::continuous("lr", 1e-4, 1e-1).log(), Variable::integer("layers", 1.0, 8.0)],
        Sense::Maximize,
    );
    let r = registry();
    let mut s = r.session(&space, &Options::new().with("seed", 2)).unwrap();
    assert_eq!(s.solver().id, "random");
    let batch = s.ask(4).unwrap();
    assert_eq!(batch.len(), 4);
    for x in &batch {
        assert!((1e-4..=1e-1).contains(&x[0]) && x[1].fract() == 0.0 && (1.0..=8.0).contains(&x[1]), "{x:?}");
    }
    // Told out of order; one failure; one infeasible; a point nobody asked for (existing data).
    s.tell(&batch[2], Outcome::constrained(0.7, vec![-1.0])).unwrap();
    s.tell(&batch[0], Outcome::Failed).unwrap();
    s.tell(&batch[1], Outcome::constrained(0.9, vec![0.5])).unwrap();
    s.tell(&[0.01, 3.0], Outcome::constrained(0.8, vec![-0.1])).unwrap();
    s.mark_pending(&batch[3]).unwrap();
    assert!(matches!(s.tell(&[0.5, 3.0], Outcome::constrained(0.1, vec![0.0])), Err(Error::InvalidPoint(_))));
    assert!(matches!(s.tell(&[0.01, 2.5], Outcome::constrained(0.1, vec![0.0])), Err(Error::InvalidPoint(_))));
    assert_eq!(s.n_evaluations(), 4);

    let sol = s.solution();
    assert_eq!((sol.termination, sol.point), (Termination::AskTell, PointKind::BestObserved { feasible: true }));
    // The best feasible value of the maximization: 0.8 (0.9 is infeasible).
    assert_eq!((sol.x.clone(), sol.objective), (vec![0.01, 3.0], Some(0.8)));
    assert!(sol.notes.iter().any(|n| n == "1 of 4 evaluations failed"), "{:?}", sol.notes);

    // pgd cannot run a session.
    let e = r.session(&space, &Options::new().with("method", "pgd")).err().unwrap();
    assert!(e.to_string().starts_with("`pgd` cannot solve this BLACKBOX (n=2, m=1) problem"), "{e}");
}

#[test]
fn explanation_with_the_reference_registry() {
    let p = Branin::new();
    let text = registry().explain(ProblemRef::BlackBox(&p), &Options::new()).to_string();
    let v = rui_ref_version();
    let expected = format!(
        "problem  BLACKBOX: 2 variables (2 continuous), 0 constraints; objective black-box; function values only; \
         minimize\nsolvers  in priority order\n  pgd     {v:<8}  unsupported: needs first derivatives; the problem \
         provides function values only\n  random  {v:<8}  supported\nchoice   rui: BLACKBOX (n=2, m=0) -> random {v} \
         (automatic)"
    );
    assert_eq!(text, expected);
}
