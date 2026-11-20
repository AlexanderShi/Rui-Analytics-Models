//! Dispatch end to end with the reference solvers: `cargo run -p rui-ref --example quickstart`.

use rui_api::{BlackBoxOracle, EvalError, Options, Outcome, Printer, Problem, ProblemRef, SmoothOracle, Variable};
use rui_models::Registry;
use rui_ref::{ProjectedGradient, RandomSearch};

/// A black box over mixed variables: `(n − 3)² + [activation != tanh] + (log10 lr + 2)²`.
struct Tuning(Vec<Variable>);

impl Problem for Tuning {
    fn name(&self) -> String {
        "tuning".into()
    }
    fn variables(&self) -> &[Variable] {
        &self.0
    }
}

impl BlackBoxOracle for Tuning {
    fn evaluate(&self, x: &[f64]) -> Outcome {
        Outcome::value((x[1] - 3.0).powi(2) + f64::from(u8::from(x[2] != 1.0)) + (x[0].log10() + 2.0).powi(2))
    }
}

/// `min (x0 − 3)² + (x1 + 1)²` on `[0, 2]²`, with the gradient.
struct Bowl(Vec<Variable>);

impl Problem for Bowl {
    fn name(&self) -> String {
        "bowl".into()
    }
    fn variables(&self) -> &[Variable] {
        &self.0
    }
}

impl SmoothOracle for Bowl {
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        Ok((x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2))
    }
    fn gradient(&self, x: &[f64], grad: &mut [f64]) -> Result<(), EvalError> {
        grad[0] = 2.0 * (x[0] - 3.0);
        grad[1] = 2.0 * (x[1] + 1.0);
        Ok(())
    }
}

fn main() {
    let registry = Registry::new().with(ProjectedGradient).with(RandomSearch);

    let tuning = Tuning(vec![
        Variable::continuous("lr", 1e-4, 1e-1).log(),
        Variable::integer("layers", 1.0, 8.0),
        Variable::categorical("activation", &["relu", "tanh", "gelu"]),
    ]);
    let r = registry.solve(ProblemRef::BlackBox(&tuning), &Options::new().with("max_evaluations", 200), &mut Printer);
    let r = r.expect("dispatch");
    println!("{} | {}: x = {:?}, f = {:?}\n", r.termination, r.point, r.x, r.objective);

    let bowl = Bowl(vec![Variable::continuous("x0", 0.0, 2.0), Variable::continuous("x1", 0.0, 2.0)]);
    println!("{}\n", registry.explain(ProblemRef::Smooth(&bowl), &Options::new()));
    let r = registry.solve(ProblemRef::Smooth(&bowl), &Options::parse("tol=1e-10").unwrap(), &mut Printer).unwrap();
    println!("{} | {}: x = {:?}, f = {:?}, duals = {:?}\n", r.termination, r.point, r.x, r.objective, r.duals);

    match registry.solve(ProblemRef::Smooth(&bowl), &Options::new().with("mehtod", "random"), &mut Printer) {
        Ok(_) => unreachable!(),
        Err(e) => println!("error: {e}"),
    }
}
