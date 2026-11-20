//! The dispatch table of docs/design.md §4.4, one test per row, with mock solvers that declare the
//! capabilities of the nlp and bo plugins; plus priority, naming, bridges and explanations.

mod common;

use common::*;
use rui_api::{
    Access, Error, Event, Flow, FnClass, FunctionClasses, Observer, Options, ProblemRef, Sense, Silent, Variable,
    Verdict,
};
use rui_models::Registry;

fn production() -> (Registry, MockLog, MockLog) {
    let (nlp, nlp_seen) = Mock::new("nlp", nlp_caps());
    let (bo, bo_seen) = Mock::new("bo", bo_caps());
    (Registry::new().with(nlp).with(bo), nlp_seen, bo_seen)
}

type MockLog = std::sync::Arc<std::sync::Mutex<Vec<Seen>>>;

fn chosen(r: &Registry, p: ProblemRef<'_>) -> Result<&'static str, Error> {
    r.solve(p, &Options::new(), &mut Silent).map(|s| s.solver.id)
}

// --- rows of the table --------------------------------------------------------------------------

#[test]
fn row1_smooth_continuous_with_second_derivatives_goes_to_nlp() {
    let (r, _, _) = production();
    let p = Smooth::bounded(3).constraint(0.0, 1.0).constraint(2.0, 2.0);
    assert_eq!(chosen(&r, ProblemRef::Smooth(&p)), Ok("nlp"));
    let unbounded = Smooth::new(vec![Variable::free("x"), Variable::free("y")]);
    assert_eq!(chosen(&r, ProblemRef::Smooth(&unbounded)), Ok("nlp"));
}

#[test]
fn row1_linear_programs_go_to_nlp_until_a_specialized_solver_exists() {
    let (r, _, _) = production();
    let mut p = Smooth::bounded(2).constraint(0.0, 1.0);
    p.classes = Some(FunctionClasses { objective: FnClass::Linear, constraints: vec![FnClass::Linear] });
    let e = r.explain(ProblemRef::Smooth(&p), &Options::new());
    assert_eq!(e.signature.short(), "LP (n=2, m=1)");
    assert_eq!(e.chosen(), Some("nlp"));
}

#[test]
fn row2_bounded_black_box_with_integer_and_categorical_goes_to_bo() {
    let (r, _, bo_seen) = production();
    let p = BlackBox::new(mixed_box()).constraint(f64::NEG_INFINITY, 0.05);
    assert_eq!(chosen(&r, ProblemRef::BlackBox(&p)), Ok("bo"));
    assert!(bo_seen.lock().unwrap()[0].black_box);
}

#[test]
fn row3_black_box_with_an_unbounded_variable_has_no_solver() {
    let (r, _, _) = production();
    let p = BlackBox::new(vec![Variable::continuous("x", 0.0, 1.0), Variable::free("y")]);
    let Err(Error::NoSolver { candidates, .. }) = chosen(&r, ProblemRef::BlackBox(&p)) else { panic!() };
    let bo = &candidates.iter().find(|(id, _)| id == "bo").unwrap().1;
    assert_eq!(bo, &Verdict::Unsupported(vec!["needs finite bounds on every variable (1 without)".to_string()]));
}

#[test]
fn row4_first_derivatives_only_has_no_solver_but_bo_can_be_named() {
    let (r, _, bo_seen) = production();
    let mut p = Smooth::bounded(2);
    p.derivatives = Access::Gradients;
    let err = chosen(&r, ProblemRef::Smooth(&p)).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("nlp: needs second derivatives; the problem provides first derivatives"), "{text}");
    assert!(text.contains("bo: usable only when named (method=bo)"), "{text}");

    let s = r.solve(ProblemRef::Smooth(&p), &Options::new().with("method", "bo"), &mut Silent).unwrap();
    assert_eq!(s.solver.id, "bo");
    assert!(s.notes[0].contains("named; derivatives ignored: solved as a black box"), "{:?}", s.notes);
    let seen = bo_seen.lock().unwrap();
    assert!(seen[0].black_box);
    assert_eq!(seen[0].objective, Some(0.5)); // f(0.5, 0.5) of the smooth problem
}

#[test]
fn row5_algebraic_problem_with_integers_has_no_solver_but_bo_can_be_named() {
    let (r, _, _) = production();
    let mut vars = vec![Variable::continuous("x", 0.0, 1.0)];
    vars.push(Variable::integer("n", 0.0, 5.0));
    let mut p = Smooth::new(vars).constraint(f64::NEG_INFINITY, 3.0);
    p.classes = Some(FunctionClasses { objective: FnClass::Quadratic, constraints: vec![FnClass::Linear] });
    let Err(Error::NoSolver { problem, candidates }) = chosen(&r, ProblemRef::Smooth(&p)) else { panic!() };
    assert_eq!(problem, "MIQP (n=2, m=1)");
    assert_eq!(
        candidates[0],
        ("nlp".to_string(), Verdict::Unsupported(vec!["does not handle integer variables (1)".into()]))
    );
    assert!(matches!(candidates[1].1, Verdict::Lossy(_)));
    let named = r.solve(ProblemRef::Smooth(&p), &Options::new().with("method", "bo"), &mut Silent).unwrap();
    assert_eq!(named.solver.id, "bo");
}

#[test]
fn row5_algebraic_problem_with_categorical_variables_has_no_solver() {
    let (r, _, _) = production();
    let p = Smooth::new(vec![Variable::continuous("x", 0.0, 1.0), Variable::categorical("c", &["a", "b"])]);
    let Err(Error::NoSolver { candidates, .. }) = chosen(&r, ProblemRef::Smooth(&p)) else { panic!() };
    assert_eq!(candidates[0].1, Verdict::Unsupported(vec!["does not handle categorical variables (1)".into()]));
}

#[test]
fn black_box_with_an_equality_constraint_has_no_solver() {
    let (r, _, _) = production();
    let p = BlackBox::new(mixed_box()).constraint(0.01, 0.01);
    let Err(Error::NoSolver { candidates, .. }) = chosen(&r, ProblemRef::BlackBox(&p)) else { panic!() };
    assert_eq!(candidates[1].1, Verdict::Unsupported(vec!["does not handle equality constraints (1)".into()]));
}

// --- priority and naming ------------------------------------------------------------------------

#[test]
fn registration_order_does_not_change_the_choice() {
    let (nlp, _) = Mock::new("nlp", nlp_caps());
    let (everything, _) = Mock::new("bo", bo_caps());
    let r = Registry::new().with(everything).with(nlp);
    assert_eq!(r.solvers().map(|s| s.info().id).collect::<Vec<_>>(), vec!["nlp", "bo"]);
    assert_eq!(chosen(&r, ProblemRef::Smooth(&Smooth::bounded(2))), Ok("nlp"));
}

#[test]
fn unknown_ids_rank_after_known_ones_in_registration_order() {
    let (z1, _) = Mock::new("zeta", nlp_caps());
    let (z2, _) = Mock::new("alpha", nlp_caps());
    let (nlp, _) = Mock::new("nlp", nlp_caps());
    let r = Registry::new().with(z1).with(z2).with(nlp);
    assert_eq!(r.solvers().map(|s| s.info().id).collect::<Vec<_>>(), vec!["nlp", "zeta", "alpha"]);
    let p = Smooth::bounded(2);
    assert_eq!(chosen(&r, ProblemRef::Smooth(&p)), Ok("nlp"));
    let (z1, _) = Mock::new("zeta", nlp_caps());
    let (z2, _) = Mock::new("alpha", nlp_caps());
    assert_eq!(chosen(&Registry::new().with(z1).with(z2), ProblemRef::Smooth(&p)), Ok("zeta"));
}

#[test]
fn naming_an_unsupported_or_unknown_solver_is_an_error() {
    let (r, _, _) = production();
    let p = Smooth::new(vec![Variable::free("x")]);
    let e = r.solve(ProblemRef::Smooth(&p), &Options::new().with("method", "bo"), &mut Silent).unwrap_err();
    assert_eq!(
        e,
        Error::Unsupported {
            method: "bo".into(),
            problem: "NLP (n=1, m=0)".into(),
            reasons: vec!["needs finite bounds on every variable (1 without)".into()]
        }
    );
    let e = r.solve(ProblemRef::Smooth(&p), &Options::new().with("method", "ipopt"), &mut Silent).unwrap_err();
    assert_eq!(e, Error::UnknownMethod { name: "ipopt".into(), available: vec!["nlp".into(), "bo".into()] });
    assert_eq!(e.to_string(), "unknown method `ipopt`; available: nlp, bo");
    let e = r.solve(ProblemRef::Smooth(&p), &Options::new().with("method", 3), &mut Silent).unwrap_err();
    assert!(matches!(e, Error::InvalidOption(_)));
}

#[test]
fn method_auto_and_case_are_accepted() {
    let (r, _, _) = production();
    let p = Smooth::bounded(1);
    for m in ["auto", "AUTO", " ", "NLP"] {
        let s = r.solve(ProblemRef::Smooth(&p), &Options::new().with("method", m), &mut Silent).unwrap();
        assert_eq!(s.solver.id, "nlp", "method={m:?}");
    }
}

#[test]
fn empty_registry() {
    let r = Registry::new();
    let e = chosen(&r, ProblemRef::Smooth(&Smooth::bounded(1))).unwrap_err();
    assert_eq!(e.to_string(), "no solver is registered");
}

#[test]
#[should_panic(expected = "registered twice")]
fn duplicate_ids_are_rejected() {
    let (a, _) = Mock::new("nlp", nlp_caps());
    let (b, _) = Mock::new("nlp", nlp_caps());
    let _ = Registry::new().with(a).with(b);
}

#[test]
#[should_panic(expected = "declared twice or is universal")]
fn options_clashing_with_universal_ones_are_rejected() {
    let (a, _) = Mock::new("nlp", nlp_caps());
    let a = a.with_options(vec![rui_api::OptionSpec::new("x", rui_api::Kind::Bool, "").alias("verbosity")]);
    let _ = Registry::new().with(a);
}

// --- bridges ------------------------------------------------------------------------------------

struct Recorder {
    objectives: Vec<f64>,
}

impl Observer for Recorder {
    fn event(&mut self, event: &Event<'_>) -> Flow {
        if let Event::Iteration { objective, .. } = event {
            self.objectives.push(*objective);
        }
        Flow::Continue
    }
}

#[test]
fn maximization_is_presented_as_minimization_and_mapped_back() {
    let (r, nlp_seen, _) = production();
    let mut p = Smooth::bounded(2); // start (0.5, 0.5): f = 0.5
    p.sense = Sense::Maximize;
    let mut rec = Recorder { objectives: Vec::new() };
    let s = r.solve(ProblemRef::Smooth(&p), &Options::new(), &mut rec).unwrap();
    let seen = nlp_seen.lock().unwrap();
    assert_eq!((seen[0].sense, seen[0].objective), (Sense::Minimize, Some(-0.5)));
    assert_eq!(s.objective, Some(0.5));
    assert_eq!(rec.objectives, vec![0.5]);
    assert_eq!(s.notes[0], "rui: NLP (n=2, m=0) -> nlp 9.9.9 (automatic; maximize -> minimize)");
}

#[test]
fn maximized_black_box() {
    let (r, _, bo_seen) = production();
    let mut p = BlackBox::new(vec![Variable::continuous("x", 1.0, 2.0)]);
    p.sense = Sense::Maximize;
    let s = r.solve(ProblemRef::BlackBox(&p), &Options::new(), &mut Silent).unwrap();
    assert_eq!(bo_seen.lock().unwrap()[0].objective, Some(-1.0));
    assert_eq!(s.objective, Some(1.0));
}

// --- explanations and validation ----------------------------------------------------------------

#[test]
fn explanation_lists_every_solver_and_the_choice() {
    let (r, _, _) = production();
    let p = Smooth::bounded(2).constraint(1.0, 1.0);
    let text = r.explain(ProblemRef::Smooth(&p), &Options::new()).to_string();
    let expected = "\
problem  NLP: 2 variables (2 continuous), 1 constraint (1 equality; 1 smooth nonlinear); objective smooth nonlinear; \
first and second derivatives; minimize
solvers  in priority order
  nlp  9.9.9     supported
  bo   9.9.9     unsupported: does not handle equality constraints (1)
choice   rui: NLP (n=2, m=1) -> nlp 9.9.9 (automatic)";
    assert_eq!(text, expected);

    let mut first_order = Smooth::bounded(2);
    first_order.derivatives = Access::Gradients;
    let e = r.explain(ProblemRef::Smooth(&first_order), &Options::new());
    assert_eq!(e.chosen(), None);
    assert!(e.to_string().contains("choice   none: no registered solver takes this NLP (n=2, m=0) problem"));
    let e = r.explain(ProblemRef::Smooth(&first_order), &Options::new().with("method", "bo"));
    assert_eq!(e.chosen(), Some("bo"));
}

#[test]
fn invalid_problems_are_rejected_before_dispatch() {
    let (r, nlp_seen, _) = production();
    let cases: Vec<(Smooth, &str)> = vec![
        (Smooth::new(vec![]), "no variables"),
        (Smooth::new(vec![Variable::continuous("x", 2.0, 1.0)]), "exceeds upper bound"),
        (Smooth::new(vec![Variable::continuous("x", f64::NAN, 1.0)]), "NaN"),
        (Smooth::new(vec![Variable::continuous("x", f64::INFINITY, f64::INFINITY)]), "admit no value"),
        (Smooth::new(vec![Variable::integer("n", 0.5, 3.0)]), "must be integers"),
        (Smooth::new(vec![Variable::continuous("x", 0.0, 1.0).with_start(2.0)]), "outside the bounds"),
        (Smooth::new(vec![Variable::integer("n", 0.0, 3.0).with_start(1.5)]), "must be integral"),
        (Smooth::new(vec![Variable::continuous("x", 0.0, 1.0).log()]), "positive lower bound"),
        (Smooth::new(vec![Variable::categorical("c", &[] as &[&str])]), "at least one level"),
        (Smooth::bounded(1).constraint(2.0, 1.0), "constraint 0: lower bound"),
    ];
    for (p, needle) in &cases {
        let e = r.solve(ProblemRef::Smooth(p), &Options::new(), &mut Silent).unwrap_err();
        assert!(matches!(&e, Error::InvalidProblem(m) if m.contains(needle)), "{needle}: {e}");
    }
    let mut wrong_classes = Smooth::bounded(1).constraint(0.0, 1.0);
    wrong_classes.classes = Some(FunctionClasses { objective: FnClass::Linear, constraints: vec![] });
    assert!(r.solve(ProblemRef::Smooth(&wrong_classes), &Options::new(), &mut Silent).is_err());
    assert!(nlp_seen.lock().unwrap().is_empty());
}

/// A problem that forgets to fill its constraint bounds.
struct ForgetsBounds(Vec<Variable>);
impl rui_api::Problem for ForgetsBounds {
    fn variables(&self) -> &[Variable] {
        &self.0
    }
    fn num_constraints(&self) -> usize {
        1
    }
}
impl rui_api::BlackBoxOracle for ForgetsBounds {
    fn evaluate(&self, _x: &[f64]) -> rui_api::Outcome {
        rui_api::Outcome::Failed
    }
}

#[test]
fn unset_constraint_bounds_are_caught() {
    let (r, _, _) = production();
    let p = ForgetsBounds(vec![Variable::continuous("x", 0.0, 1.0)]);
    let e = r.solve(ProblemRef::BlackBox(&p), &Options::new(), &mut Silent).unwrap_err();
    assert!(e.to_string().contains("does the problem implement `constraint_bounds`?"), "{e}");
}
