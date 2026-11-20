//! Bridges: views that present a problem in the form a solver takes.
//!
//! * [`Negated`]: a maximization problem as the minimization of `-f` (the multipliers are unchanged:
//!   they are defined for the minimized function, see `rui_api::Duals`);
//! * [`AsBlackBox`]: a smooth problem as a black box (only when a black-box solver is named);
//! * [`FlipObserver`]: progress events of a negated problem reported with the original objective.

use rui_api::{
    Access, BlackBoxOracle, Context, EvalError, Event, Flow, FunctionClasses, Observer, Outcome, Problem, ProblemRef,
    ResolvedOptions, Sense, SmoothOracle, Solution, Solver, Variable,
};

/// Runs `solver` on `problem` after applying the bridges its form requires.
pub(crate) fn run(
    solver: &dyn Solver,
    problem: ProblemRef<'_>,
    options: &ResolvedOptions,
    ctx: &mut Context<'_>,
) -> Solution {
    let negated_smooth;
    let negated_black_box;
    let as_black_box;
    let mut p = problem;
    if problem.sense() == Sense::Maximize {
        p = match problem {
            ProblemRef::Smooth(s) => {
                negated_smooth = Negated(s);
                ProblemRef::Smooth(&negated_smooth)
            }
            ProblemRef::BlackBox(b) => {
                negated_black_box = Negated(b);
                ProblemRef::BlackBox(&negated_black_box)
            }
        };
    }
    if solver.info().capabilities.uses == Access::Values {
        if let ProblemRef::Smooth(s) = p {
            as_black_box = AsBlackBox(s);
            p = ProblemRef::BlackBox(&as_black_box);
        }
    }
    solver.solve(p, options, ctx)
}

/// `problem` with sense [`Sense::Minimize`] and, where it has one, its objective negated.
pub(crate) struct Negated<'a, P: ?Sized>(pub &'a P);

impl<P: Problem + ?Sized> Problem for Negated<'_, P> {
    fn name(&self) -> String {
        self.0.name()
    }
    fn variables(&self) -> &[Variable] {
        self.0.variables()
    }
    fn num_constraints(&self) -> usize {
        self.0.num_constraints()
    }
    fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
        self.0.constraint_bounds(lower, upper)
    }
    fn sense(&self) -> Sense {
        Sense::Minimize
    }
}

impl<P: SmoothOracle + ?Sized> SmoothOracle for Negated<'_, P> {
    fn derivatives(&self) -> Access {
        self.0.derivatives()
    }
    fn function_classes(&self) -> Option<FunctionClasses> {
        self.0.function_classes()
    }
    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        self.0.objective(x).map(|f| -f)
    }
    fn gradient(&self, x: &[f64], grad: &mut [f64]) -> Result<(), EvalError> {
        self.0.gradient(x, grad)?;
        for g in grad.iter_mut() {
            *g = -*g;
        }
        Ok(())
    }
    fn constraints(&self, x: &[f64], g: &mut [f64]) -> Result<(), EvalError> {
        self.0.constraints(x, g)
    }
    fn jacobian_structure(&self) -> Vec<(usize, usize)> {
        self.0.jacobian_structure()
    }
    fn jacobian_values(&self, x: &[f64], values: &mut [f64]) -> Result<(), EvalError> {
        self.0.jacobian_values(x, values)
    }
    fn hessian_structure(&self) -> Vec<(usize, usize)> {
        self.0.hessian_structure()
    }
    fn hessian_values(&self, x: &[f64], obj_factor: f64, lambda: &[f64], values: &mut [f64]) -> Result<(), EvalError> {
        self.0.hessian_values(x, -obj_factor, lambda, values)
    }
}

impl<P: BlackBoxOracle + ?Sized> BlackBoxOracle for Negated<'_, P> {
    fn evaluate(&self, x: &[f64]) -> Outcome {
        negate(&self.0.evaluate(x))
    }
    fn noisy(&self) -> bool {
        self.0.noisy()
    }
}

pub(crate) fn negate(outcome: &Outcome) -> Outcome {
    match outcome {
        Outcome::Value { objective, constraints } => {
            Outcome::Value { objective: -objective, constraints: constraints.clone() }
        }
        Outcome::Failed => Outcome::Failed,
    }
}

/// A smooth problem evaluated as a black box; an evaluation error is a failed evaluation.
pub(crate) struct AsBlackBox<'a, P: ?Sized>(pub &'a P);

impl<P: SmoothOracle + ?Sized> Problem for AsBlackBox<'_, P> {
    fn name(&self) -> String {
        self.0.name()
    }
    fn variables(&self) -> &[Variable] {
        self.0.variables()
    }
    fn num_constraints(&self) -> usize {
        self.0.num_constraints()
    }
    fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
        self.0.constraint_bounds(lower, upper)
    }
    fn sense(&self) -> Sense {
        self.0.sense()
    }
}

impl<P: SmoothOracle + ?Sized> BlackBoxOracle for AsBlackBox<'_, P> {
    fn evaluate(&self, x: &[f64]) -> Outcome {
        let Ok(objective) = self.0.objective(x) else { return Outcome::Failed };
        let mut constraints = vec![0.0; self.0.num_constraints()];
        if self.0.constraints(x, &mut constraints).is_err() {
            return Outcome::Failed;
        }
        Outcome::Value { objective, constraints }
    }
}

/// Reports the events of a negated (maximization) problem with the original objective values.
pub(crate) struct FlipObserver<'a>(pub &'a mut dyn Observer);

impl Observer for FlipObserver<'_> {
    fn log(&mut self, line: &str) {
        self.0.log(line)
    }
    fn interrupted(&mut self) -> bool {
        self.0.interrupted()
    }
    fn event(&mut self, event: &Event<'_>) -> Flow {
        match *event {
            Event::Iteration { iteration, objective, infeasibility, optimality } => {
                self.0.event(&Event::Iteration { iteration, objective: -objective, infeasibility, optimality })
            }
            Event::Evaluation { index, x, outcome, best } => {
                let flipped = negate(outcome);
                self.0.event(&Event::Evaluation { index, x, outcome: &flipped, best: best.map(|b| -b) })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f = x², undefined (error) for x > 1; one constraint g = 2x, maximized.
    struct P(Vec<Variable>);
    impl Problem for P {
        fn variables(&self) -> &[Variable] {
            &self.0
        }
        fn num_constraints(&self) -> usize {
            1
        }
        fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
            lower[0] = 0.0;
            upper[0] = 1.0;
        }
        fn sense(&self) -> Sense {
            Sense::Maximize
        }
    }
    impl SmoothOracle for P {
        fn derivatives(&self) -> Access {
            Access::Hessians
        }
        fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
            if x[0] > 1.0 {
                Err(EvalError::new("outside the domain"))
            } else {
                Ok(x[0] * x[0])
            }
        }
        fn gradient(&self, x: &[f64], grad: &mut [f64]) -> Result<(), EvalError> {
            grad[0] = 2.0 * x[0];
            Ok(())
        }
        fn constraints(&self, x: &[f64], g: &mut [f64]) -> Result<(), EvalError> {
            g[0] = 2.0 * x[0];
            Ok(())
        }
        fn hessian_structure(&self) -> Vec<(usize, usize)> {
            vec![(0, 0)]
        }
        fn hessian_values(&self, _x: &[f64], obj_factor: f64, _l: &[f64], values: &mut [f64]) -> Result<(), EvalError> {
            values[0] = 2.0 * obj_factor;
            Ok(())
        }
    }

    #[test]
    fn negated_smooth_problem() {
        let p = P(vec![Variable::continuous("x", -2.0, 2.0)]);
        let n = Negated(&p);
        assert_eq!(n.sense(), Sense::Minimize);
        assert_eq!(n.objective(&[0.5]), Ok(-0.25));
        let mut g = [0.0];
        n.gradient(&[0.5], &mut g).unwrap();
        assert_eq!(g, [-1.0]);
        let mut h = [0.0];
        n.hessian_values(&[0.5], 1.0, &[0.0], &mut h).unwrap();
        assert_eq!(h, [-2.0]);
        n.constraints(&[0.5], &mut g).unwrap();
        assert_eq!(g, [1.0]);
    }

    #[test]
    fn smooth_problem_as_black_box() {
        let p = P(vec![Variable::continuous("x", -2.0, 2.0)]);
        let b = AsBlackBox(&p);
        assert_eq!(b.evaluate(&[0.5]), Outcome::constrained(0.25, vec![1.0]));
        assert_eq!(b.evaluate(&[1.5]), Outcome::Failed);
        let nb = Negated(&b);
        assert_eq!(nb.evaluate(&[0.5]), Outcome::constrained(-0.25, vec![1.0]));
        assert_eq!(nb.sense(), Sense::Minimize);
    }
}
