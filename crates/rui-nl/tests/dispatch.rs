//! `.nl` files dispatched through `rui_models::Registry` with the reference solvers: the function classes
//! reach the signature, maximization is bridged, and a constrained model that only a black-box solver
//! could take is refused automatically and solved when that solver is named.

use rui_api::{kkt_check, Error, Options, PointKind, ProblemRef, Silent};
use rui_expr::ExprProblem;
use rui_models::Registry;
use rui_ref::{ProjectedGradient, RandomSearch};

fn load(n: usize, m: usize, body: &str) -> ExprProblem {
    let header = format!(
        "g3 1 1 0\t# problem t\n {n} {m} 1 0 0\n {m} 1\n 0 0\n {n} {n} {n}\n 0 0 0 1\n 0 0 0 0 0\n 0 0\n 0 0\n 0 0 0 0 0\n"
    );
    ExprProblem::new(&rui_nl::parse(&(header + body)).unwrap()).unwrap()
}

fn registry() -> Registry {
    Registry::new().with(ProjectedGradient).with(RandomSearch)
}

/// `min (x0 − 3)² + (x1 + 1)²` on `[0, 2]²`.
const BOWL: &str = "O0 0\no54\n2\no5\no0\nv0\nn-3\nn2\no5\no0\nv1\nn1\nn2\nb\n0 0 2\n0 0 2\nG0 2\n0 0\n1 0\n";

#[test]
fn a_bound_constrained_quadratic_goes_to_pgd() {
    let p = load(2, 0, BOWL);
    let s = registry().solve(ProblemRef::Smooth(&p), &Options::parse("tol=1e-10").unwrap(), &mut Silent).unwrap();
    assert!(s.notes[0].starts_with("rui: QP (n=2, m=0) -> pgd "), "{}", s.notes[0]);
    assert_eq!(s.point, PointKind::KktPoint { verified: true });
    assert!((s.x[0] - 2.0).abs() < 1e-9 && s.x[1].abs() < 1e-9, "{:?}", s.x);
    assert!((s.objective.unwrap() - 2.0).abs() < 1e-9, "{:?}", s.objective);
    assert!(kkt_check(&p, &s.x, s.duals.as_ref().unwrap()).unwrap().within(1e-8));
}

#[test]
fn a_maximized_model_is_bridged_and_reported_in_its_own_sense() {
    // max 5 − (x0 − 1)² on [−5, 5], starting at 4
    let body = "O0 1\no0\no16\no5\no0\nv0\nn-1\nn2\nn5\nx1\n0 4\nb\n0 -5 5\nG0 1\n0 0\n";
    let p = load(1, 0, body);
    let s = registry().solve(ProblemRef::Smooth(&p), &Options::new(), &mut Silent).unwrap();
    assert!(s.notes[0].contains("-> pgd") && s.notes[0].contains("maximize -> minimize"), "{}", s.notes[0]);
    assert!((s.x[0] - 1.0).abs() < 1e-5, "{:?}", s.x);
    assert!((s.objective.unwrap() - 5.0).abs() < 1e-9, "{:?}", s.objective);
}

#[test]
fn a_constrained_model_needs_a_named_black_box_solver() {
    // min (x0 − 1)² + (x1 − 1)² subject to x0 + x1 <= 1 on [0, 2]²
    let body = "C0\nn0\nO0 0\no54\n2\no5\no0\nv0\nn-1\nn2\no5\no0\nv1\nn-1\nn2\nr\n1 1\nb\n0 0 2\n0 0 2\nJ0 2\n0 1\n1 1\nG0 2\n0 0\n1 0\n";
    let p = load(2, 1, body);
    let reg = registry();
    let err = reg.solve(ProblemRef::Smooth(&p), &Options::new(), &mut Silent).unwrap_err();
    assert!(matches!(err, Error::NoSolver { .. }), "{err}");
    let text = err.to_string();
    assert!(text.starts_with("no registered solver takes this QP (n=2, m=1) problem automatically"), "{text}");
    assert!(text.contains("random: usable only when named (method=random)"), "{text}");

    let options = Options::new().with("method", "random").with("max_evaluations", 400);
    let s = reg.solve(ProblemRef::Smooth(&p), &options, &mut Silent).unwrap();
    assert!(s.notes[0].contains("-> random") && s.notes[0].contains("named"), "{}", s.notes[0]);
    assert_eq!(s.point, PointKind::BestObserved { feasible: true });
    assert!(s.x[0] + s.x[1] <= 1.0 && s.constraints[0] <= 1.0, "{s:?}");
}
