//! Option resolution (docs/design.md §4.5) and ask/tell sessions.

mod common;

use common::*;
use rui_api::{common as opts, Error, Kind, OptionSpec, Options, Outcome, ProblemRef, Sense, Silent, Value, Variable};
use rui_models::Registry;

fn registry() -> (Registry, std::sync::Arc<std::sync::Mutex<Vec<Seen>>>) {
    let (nlp, seen) = Mock::new("nlp", nlp_caps());
    let nlp = nlp.with_options(vec![
        opts::max_iterations(3000).alias("max_iter"),
        opts::tol(1e-8),
        OptionSpec::new("mu_init", Kind::positive_float(), "initial barrier parameter").with_default(0.1),
        OptionSpec::new("linear_solver", Kind::choice(&["rui", "dense"]), "").with_default("rui"),
    ]);
    let (bo, _) = Mock::new("bo", bo_caps());
    let bo = bo.with_options(vec![
        opts::max_evaluations(100),
        opts::seed(0),
        OptionSpec::new("acquisition", Kind::choice(&["logei", "ucb"]), "").with_default("logei"),
    ]);
    (Registry::new().with(nlp).with(bo), seen)
}

fn solve(options: Options) -> Result<(rui_api::ResolvedOptions, Vec<String>), Error> {
    let (r, seen) = registry();
    let p = Smooth::bounded(2);
    let s = r.solve(ProblemRef::Smooth(&p), &options, &mut Silent)?;
    let resolved = seen.lock().unwrap()[0].options.clone();
    Ok((resolved, s.notes))
}

fn invalid(options: Options) -> String {
    match solve(options) {
        Err(Error::InvalidOption(m)) => m,
        other => panic!("expected an invalid option, got {other:?}"),
    }
}

#[test]
fn defaults_values_and_aliases() {
    let (o, notes) = solve(Options::new()).unwrap();
    assert_eq!(o.i64("max_iterations"), 3000);
    assert_eq!(o.f64("tol"), 1e-8);
    assert_eq!(o.text("linear_solver"), "rui");
    assert_eq!(o.f64("time_limit"), f64::INFINITY);
    assert_eq!(o.i64("verbosity"), 1);
    assert!(!o.is_given("tol"));
    assert_eq!(notes.len(), 1);

    let text = Options::parse("tol=1e-10 max_iter=50 mu_init=1 linear_solver=DENSE time_limit=5").unwrap();
    let (o, _) = solve(text).unwrap();
    assert_eq!(o.f64("tol"), 1e-10);
    assert_eq!(o.i64("max_iterations"), 50);
    assert_eq!(o.get("mu_init"), Some(&Value::Float(1.0)));
    assert_eq!(o.text("linear_solver"), "dense");
    assert_eq!(o.f64("time_limit"), 5.0);
    assert!(o.is_given("tol") && o.is_given("max_iterations"));

    let (o, _) = solve(Options::new().with("nlp.mu_init", 0.5)).unwrap();
    assert_eq!(o.f64("mu_init"), 0.5);
}

#[test]
fn scoped_options_of_other_solvers_and_unused_common_options_are_noticed() {
    let (_, notes) = solve(Options::new().with("bo.acquisition", "ucb").with("max_evaluations", 10)).unwrap();
    assert_eq!(
        notes[1..],
        [
            "option `bo.acquisition` ignored: it applies only when `bo` solves the problem".to_string(),
            "option `max_evaluations` ignored: `nlp` does not use it".to_string(),
        ]
    );
}

#[test]
fn errors_say_what_is_wrong() {
    assert_eq!(invalid(Options::new().with("tlo", 1e-8)), "unknown option `tlo` for `nlp` (did you mean `tol`?)");
    assert_eq!(
        invalid(Options::new().with("max_iters", 5)),
        "unknown option `max_iters` for `nlp` (did you mean `max_iter`?)"
    );
    assert_eq!(
        invalid(Options::new().with("acquisition", "ucb")),
        "option `acquisition` belongs to `bo` but this problem is solved by `nlp`; write `bo.acquisition` to set it \
         only for `bo`"
    );
    assert_eq!(invalid(Options::new().with("ipopt.tol", 1)), "option `ipopt.tol`: no solver `ipopt` is registered");
    assert_eq!(invalid(Options::new().with("nlp.seed", 1)), "option `nlp.seed`: `nlp` does not use `seed`");
    assert_eq!(
        invalid(Options::new().with("nlp.mu_inti", 1)),
        "unknown option `nlp.mu_inti` for `nlp` (did you mean `mu_init`?)"
    );
    assert_eq!(
        invalid(Options::new().with("max_iter", 5).with("max_iterations", 6)),
        "option `max_iterations` is given twice (as `max_iter` and `max_iterations`)"
    );
    assert_eq!(
        invalid(Options::new().with("tol", 1e-6).with("nlp.tol", 1e-7)),
        "option `tol` is given twice (as `tol` and `nlp.tol`)"
    );
    assert_eq!(invalid(Options::new().with("tol", "small")), "option `tol`: expected a number > 0.0, got `small`");
    assert_eq!(invalid(Options::new().with("max_iter", -1)), "option `max_iter`: -1 is not an integer >= 0");
    assert_eq!(invalid(Options::new().with("verbosity", 7)), "option `verbosity`: 7 is not an integer in [0, 3]");
    assert_eq!(
        invalid(Options::new().with("linear_solver", "ma57")),
        "option `linear_solver`: expected one of rui, dense, got `ma57`"
    );
}

#[test]
fn time_limit_reaches_the_context() {
    // A solver that sees an expired deadline: the mock does not check, so use the context directly.
    let mut silent = Silent;
    let mut ctx = rui_api::Context::new(&mut silent).with_time_limit(0.0);
    assert_eq!(ctx.stop_reason(), Some(rui_api::Termination::TimeLimit));
    let mut ctx = rui_api::Context::new(&mut silent).with_time_limit(f64::INFINITY);
    assert_eq!(ctx.stop_reason(), None);
}

// --- sessions -----------------------------------------------------------------------------------

struct Space {
    vars: Vec<Variable>,
    m: usize,
    sense: Sense,
}

impl rui_api::Problem for Space {
    fn variables(&self) -> &[Variable] {
        &self.vars
    }
    fn num_constraints(&self) -> usize {
        self.m
    }
    fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
        lower.fill(f64::NEG_INFINITY);
        upper.fill(0.0);
    }
    fn sense(&self) -> Sense {
        self.sense
    }
}

#[test]
fn sessions_go_to_solvers_with_ask_tell_and_validate_what_is_told() {
    let (r, _) = registry();
    let space = Space { vars: vec![Variable::continuous("x", 0.0, 1.0)], m: 1, sense: Sense::Maximize };
    let mut s = r.session(&space, &Options::new().with("seed", 3)).unwrap();
    assert_eq!(s.solver().id, "bo");
    assert_eq!(s.notes()[0], "rui: BLACKBOX (n=1, m=1) -> bo 9.9.9 (automatic; maximize -> minimize)");
    assert_eq!(s.ask(2).unwrap().len(), 2);
    assert!(s.ask(0).unwrap().is_empty());

    assert!(matches!(s.tell(&[0.5, 0.5], Outcome::value(1.0)), Err(Error::InvalidPoint(_))));
    assert!(matches!(s.tell(&[0.5], Outcome::value(1.0)), Err(Error::InvalidValue(_)))); // missing constraint
    assert!(matches!(s.tell(&[0.5], Outcome::constrained(f64::NAN, vec![0.0])), Err(Error::InvalidValue(_))));
    s.tell(&[0.5], Outcome::constrained(2.0, vec![-1.0])).unwrap();
    s.tell(&[0.7], Outcome::Failed).unwrap();
    assert_eq!(s.n_evaluations(), 2);
    let sol = s.solution();
    // The mock reports the last value it was told; it received the negated objective.
    assert_eq!(sol.objective, None); // last told was a failure
    let mut s = r.session(&space, &Options::new()).unwrap();
    s.tell(&[0.5], Outcome::constrained(2.0, vec![-1.0])).unwrap();
    assert_eq!(s.solution().objective, Some(2.0));

    // nlp has no ask/tell interface.
    let e = r.session(&space, &Options::new().with("method", "nlp")).err().unwrap();
    assert!(e.to_string().contains("has no ask/tell interface"), "{e}");
}
