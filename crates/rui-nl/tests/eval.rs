//! `.nl` text → `rui_expr::ExprProblem`: values against hand computation, derivatives against central
//! differences, element sharing, quadratic terms, domain errors, function classes.

use rui_api::{EvalError, FnClass, FunctionClasses, Problem, Sense, SmoothOracle};
use rui_expr::ExprProblem;

/// Header for a problem with `n` variables, `m` constraints and one objective.
fn header(n: usize, m: usize) -> String {
    format!(
        "g3 1 1 0\t# problem t\n {n} {m} 1 0 0\n {m} 1\n 0 0\n {n} {n} {n}\n 0 0 0 1\n 0 0 0 0 0\n 0 0\n 0 0\n 0 0 0 0 0\n"
    )
}

fn problem(body: &str, n: usize, m: usize) -> ExprProblem {
    let text = header(n, m) + body;
    ExprProblem::new(&rui_nl::parse(&text).unwrap()).unwrap()
}

fn ok<T>(r: Result<T, EvalError>) -> Result<T, String> {
    r.map_err(|e| e.to_string())
}

/// Dense `m x n` Jacobian.
fn jacobian(p: &dyn SmoothOracle, x: &[f64]) -> Result<Vec<Vec<f64>>, String> {
    let structure = p.jacobian_structure();
    let mut values = vec![0.0; structure.len()];
    ok(p.jacobian_values(x, &mut values))?;
    let mut jac = vec![vec![0.0; x.len()]; p.num_constraints()];
    for (&(i, j), v) in structure.iter().zip(&values) {
        jac[i][j] += v;
    }
    Ok(jac)
}

/// `σ ∇f + Jᵀ λ`.
fn lagrangian_gradient(p: &dyn SmoothOracle, x: &[f64], sigma: f64, lambda: &[f64]) -> Result<Vec<f64>, String> {
    let mut g = vec![0.0; x.len()];
    ok(p.gradient(x, &mut g))?;
    for v in &mut g {
        *v *= sigma;
    }
    for (row, l) in jacobian(p, x)?.iter().zip(lambda) {
        for (gj, a) in g.iter_mut().zip(row) {
            *gj += l * a;
        }
    }
    Ok(g)
}

/// Compares the gradient, the Jacobian and the Hessian of the Lagrangian with central differences of
/// the next lower derivative; relative tolerance `tol` (absolute below 1).
fn check_derivatives(p: &dyn SmoothOracle, x: &[f64], tol: f64) -> Result<(), String> {
    let (n, m) = (x.len(), p.num_constraints());
    let close = |a: f64, b: f64| (a - b).abs() <= tol * 1f64.max(a.abs()).max(b.abs());
    let step = |xj: f64| 1e-6 * 1f64.max(xj.abs());
    let mut g = vec![0.0; n];
    ok(p.gradient(x, &mut g))?;
    let jac = jacobian(p, x)?;
    let (sigma, lambda): (f64, Vec<f64>) = (0.7, (0..m).map(|i| 1.0 - 0.6 * i as f64).collect());
    let mut hess = vec![vec![0.0; n]; n];
    let structure = p.hessian_structure();
    let mut values = vec![0.0; structure.len()];
    ok(p.hessian_values(x, sigma, &lambda, &mut values))?;
    for (&(i, j), v) in structure.iter().zip(&values) {
        hess[i][j] += v;
        if i != j {
            hess[j][i] += v;
        }
    }
    let mut xs = x.to_vec();
    for j in 0..n {
        let h = step(x[j]);
        let (mut cp, mut cm) = (vec![0.0; m], vec![0.0; m]);
        xs[j] = x[j] + h;
        let fp = ok(p.objective(&xs))?;
        ok(p.constraints(&xs, &mut cp))?;
        let lp = lagrangian_gradient(p, &xs, sigma, &lambda)?;
        xs[j] = x[j] - h;
        let fm = ok(p.objective(&xs))?;
        ok(p.constraints(&xs, &mut cm))?;
        let lm = lagrangian_gradient(p, &xs, sigma, &lambda)?;
        xs[j] = x[j];
        let fd = (fp - fm) / (2.0 * h);
        if !close(g[j], fd) {
            return Err(format!("gradient[{j}] = {} but the difference quotient is {fd}", g[j]));
        }
        for i in 0..m {
            let fd = (cp[i] - cm[i]) / (2.0 * h);
            if !close(jac[i][j], fd) {
                return Err(format!("jacobian[{i}][{j}] = {} but the difference quotient is {fd}", jac[i][j]));
            }
        }
        for k in 0..n {
            let fd = (lp[k] - lm[k]) / (2.0 * h);
            if !close(hess[k][j], fd) {
                return Err(format!("hessian[{k}][{j}] = {} but the difference quotient is {fd}", hess[k][j]));
            }
        }
    }
    Ok(())
}

// objective: maximize -(x0^2 + x0*x1) + exp(x1)*sin(x2) + 3*(x1/x2) - log(x0)^2 + atan2(x0, x1) - 1
// constraints: c0 = sqrt(x0*x2) + x1 + 4 (linear part from J: x1)
//              c1 = x0^2.5 + x1^x0 + tanh(x2) + max(x0, x1) + (x0 + x1)*(x0 - x2) - x0*x1 + 2
const BODY: &str = "C0
o0
o39
o2
v0
v2
n4
C1
o54
7
o5
v0
n2.5
o5
v1
v0
o37
v2
o12
2
v0
v1
o2
o0
v0
v1
o1
v0
v2
o16
o2
v0
v1
n2
O0 1
o54
6
o16
o0
o5
v0
n2
o2
v0
v1
o2
o44
v1
o41
v2
o2
n3
o3
v1
v2
o16
o5
o43
v0
n2
o48
v0
v1
o16
n1
x3
0 1.3
1 0.7
2 1.1
r
1 10
3
b
2 0.1
3
0 0.5 3
J0 3
0 0
1 1
2 0
J1 3
0 0
1 0
2 0
G0 3
0 0
1 0
2 0
";

#[test]
fn values_match_hand_computation() {
    let p = problem(BODY, 3, 2);
    assert_eq!(p.sense(), Sense::Maximize);
    let x: [f64; 3] = [1.3, 0.7, 1.1];
    let (a, b, c) = (x[0], x[1], x[2]);
    let f = -(a * a + a * b) + b.exp() * c.sin() + 3.0 * (b / c) - a.ln().powi(2) + a.atan2(b) - 1.0;
    assert!((p.objective(&x).unwrap() - f).abs() < 1e-13, "the objective is evaluated as written");
    let mut g = [0.0; 2];
    p.constraints(&x, &mut g).unwrap();
    assert!((g[0] - ((a * c).sqrt() + b + 4.0)).abs() < 1e-13);
    let c1 = a.powf(2.5) + b.powf(a) + c.tanh() + a.max(b) + (a + b) * (a - c) - a * b + 2.0;
    assert!((g[1] - c1).abs() < 1e-12);
    let starts: Vec<Option<f64>> = p.variables().iter().map(|v| v.start).collect();
    assert_eq!(starts, vec![Some(1.3), Some(0.7), Some(1.1)]);
}

#[test]
fn derivatives_match_finite_differences() {
    let p = problem(BODY, 3, 2);
    for x in [[1.3, 0.7, 1.1], [0.6, 1.9, 2.4], [2.2, 0.4, 0.8]] {
        check_derivatives(&p, &x, 1e-6).unwrap_or_else(|e| panic!("at {x:?}: {e}"));
    }
    let c = p.function_classes().unwrap();
    assert_eq!((c.objective, c.constraints), (FnClass::Smooth, vec![FnClass::Smooth, FnClass::Smooth]));
}

#[test]
fn identical_elements_are_shared() {
    // c0 = t*sqrt(y0^2 + y1^2) + y0, c1 = -t*sqrt(y0^2 + y1^2) + y1 (the WM_CFy pattern)
    let el = "o2\nv2\no39\no0\no5\nv0\nn2\no5\nv1\nn2\n";
    let body = format!(
        "C0\n{el}C1\no16\n{el}O0 0\nv2\nr\n1 0\n1 0\nb\n3\n3\n2 0.5\nJ0 3\n0 1\n1 0\n2 0\nJ1 3\n0 0\n1 1\n2 0\nG0 1\n2 1\n"
    );
    let p = problem(&body, 3, 2);
    assert_eq!((p.stats().elements, p.stats().unique_elements), (2, 1));
    check_derivatives(&p, &[0.3, -0.8, 1.7], 1e-6).unwrap();
    // the Lagrangian Hessian uses the combined weight lambda0 - lambda1
    let mut h = vec![0.0; p.hessian_structure().len()];
    p.hessian_values(&[0.3, -0.8, 1.7], 1.0, &[2.0, 2.0], &mut h).unwrap();
    assert!(h.iter().all(|v| v.abs() < 1e-15), "equal and opposite multipliers cancel");
}

#[test]
fn quadratic_terms_have_exact_derivatives() {
    // objective 0.5*x0^2 + 3*x0*x1 written as quad terms; constraint x0*x1 + x1^2
    let body = "C0\no0\no2\nv0\nv1\no5\nv1\nn2\nO0 0\no0\no2\nn0.5\no5\nv0\nn2\no2\nn3\no2\nv0\nv1\nr\n3\nb\n3\n3\nJ0 2\n0 0\n1 0\nG0 2\n0 0\n1 0\n";
    let p = problem(body, 2, 1);
    assert_eq!((p.stats().quadratic_terms, p.stats().unique_elements), (4, 0));
    check_derivatives(&p, &[0.7, -1.2], 1e-6).unwrap();
    assert_eq!(
        p.function_classes(),
        Some(FunctionClasses { objective: FnClass::Quadratic, constraints: vec![FnClass::Quadratic] })
    );
}

#[test]
fn domain_errors_are_reported() {
    let body = "C0\no43\nv0\nO0 0\nn0\nr\n3\nb\n3\nJ0 1\n0 0\n";
    let p = problem(body, 1, 1);
    let mut g = [0.0];
    assert!(p.constraints(&[-1.0], &mut g).is_err());
    assert!(p.constraints(&[2.0], &mut g).is_ok());
}
