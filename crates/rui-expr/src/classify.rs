//! Function classes from the expression graph: the polynomial degree of every node, in one pass over
//! the nodes (operands precede the nodes that use them).
//!
//! The degree is an upper bound: `(x − x)·y` counts as quadratic. Errors only go towards a less
//! structured class, never towards a more structured one, so a solver that requires linear or quadratic
//! functions never receives anything else.
//!
//! The contract has no class for nonsmooth functions. Functions that use `abs`, `min`, `max`, `floor` or
//! `ceil` of non-constant arguments are reported as [`FnClass::Smooth`]: they are differentiable almost
//! everywhere, and the evaluator returns the derivative of the active piece, as the AMPL solver library
//! does.

use crate::model::{BinOp, Function, Model, Node, NodeId};
use rui_api::{FnClass, FunctionClasses};

/// Degree of a polynomial, capped at [`CAP`]; `None` when the node is not a polynomial.
type Degree = Option<u8>;

/// "Cubic or higher".
const CAP: u8 = 3;

impl Model {
    /// Class of the objective and of each constraint. Requires a model that passes [`Model::check`];
    /// on other models it errs towards [`FnClass::Smooth`].
    pub fn classes(&self) -> FunctionClasses {
        let d = degrees(self);
        FunctionClasses {
            objective: class(&self.objective, &d),
            constraints: self.constraints.iter().map(|c| class(&c.body, &d)).collect(),
        }
    }
}

fn class(f: &Function, d: &[Degree]) -> FnClass {
    match f.expr.map_or(Some(0), |r| d.get(r as usize).copied().flatten()) {
        Some(0 | 1) => FnClass::Linear,
        Some(2) => FnClass::Quadratic,
        _ => FnClass::Smooth,
    }
}

fn degrees(model: &Model) -> Vec<Degree> {
    let mut d: Vec<Degree> = Vec::with_capacity(model.nodes.len());
    for node in &model.nodes {
        let at = |id: NodeId| d.get(id as usize).copied().flatten();
        let list = |s: u32, l: u32| -> Vec<Degree> {
            model.kids.get(s as usize..(s + l) as usize).unwrap_or(&[]).iter().map(|&k| at(k)).collect()
        };
        let deg = match *node {
            Node::Num(_) => Some(0),
            Node::Var(_) => Some(1),
            Node::Neg(a) => at(a),
            Node::Bin(BinOp::Add | BinOp::Sub, a, b) => highest(&[at(a), at(b)]),
            Node::Sum(s, l) => highest(&list(s, l)),
            Node::Bin(BinOp::Mul, a, b) => product(at(a), at(b)),
            Node::Bin(BinOp::Div, a, b) => {
                if at(b) == Some(0) {
                    at(a)
                } else {
                    None
                }
            }
            Node::Bin(BinOp::Pow, a, b) => {
                let exponent = match model.nodes.get(b as usize) {
                    Some(Node::Num(c)) => Some(*c),
                    _ => None,
                };
                power(at(a), at(b), exponent)
            }
            Node::Bin(BinOp::Atan2, a, b) => constant(&[at(a), at(b)]),
            Node::Fun(_, a) => constant(&[at(a)]),
            Node::Min(s, l) | Node::Max(s, l) => constant(&list(s, l)),
        };
        d.push(deg);
    }
    d
}

/// Degree of a sum: the highest degree of its terms.
fn highest(terms: &[Degree]) -> Degree {
    terms.iter().try_fold(0, |acc, t| t.map(|t| acc.max(t)))
}

fn product(a: Degree, b: Degree) -> Degree {
    Some(a?.saturating_add(b?).min(CAP))
}

/// `a^b`: a polynomial when `b` is a non-negative integer constant, or when both are constants.
fn power(base: Degree, exponent: Degree, value: Option<f64>) -> Degree {
    match (base?, exponent?) {
        (0, 0) => Some(0),
        (b, 0) => match value {
            Some(c) if c >= 0.0 && c.fract() == 0.0 => Some((f64::from(b) * c).min(f64::from(CAP)) as u8),
            _ => None,
        },
        _ => None,
    }
}

/// Degree 0 when every argument is constant; otherwise not a polynomial.
fn constant(args: &[Degree]) -> Degree {
    args.iter().all(|&a| a == Some(0)).then_some(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Fun;
    use rui_api::Variable;

    fn with_objective(build: impl FnOnce(&mut Model, NodeId, NodeId) -> NodeId) -> FnClass {
        let mut m = Model::new("t");
        m.variables = vec![Variable::free("x"), Variable::free("y")];
        let x = m.add(Node::Var(0));
        let y = m.add(Node::Var(1));
        let root = build(&mut m, x, y);
        m.objective.expr = Some(root);
        m.check().unwrap();
        m.classes().objective
    }

    #[test]
    fn degrees_give_the_classes() {
        use FnClass::*;
        let num = |m: &mut Model, c: f64| m.add(Node::Num(c));
        // x / 2 + 3
        assert_eq!(
            with_objective(|m, x, _| {
                let two = num(m, 2.0);
                let q = m.add(Node::Bin(BinOp::Div, x, two));
                let three = num(m, 3.0);
                m.add(Node::Bin(BinOp::Add, q, three))
            }),
            Linear
        );
        // exp(2) * y: a function of a constant is a constant
        assert_eq!(
            with_objective(|m, _, y| {
                let two = num(m, 2.0);
                let e = m.add(Node::Fun(Fun::Exp, two));
                m.add(Node::Bin(BinOp::Mul, e, y))
            }),
            Linear
        );
        assert_eq!(with_objective(|m, x, y| m.add(Node::Bin(BinOp::Mul, x, y))), Quadratic);
        // (x + y)^2 and -(x*y) + x
        assert_eq!(
            with_objective(|m, x, y| {
                let s = m.add(Node::Bin(BinOp::Add, x, y));
                let two = num(m, 2.0);
                m.add(Node::Bin(BinOp::Pow, s, two))
            }),
            Quadratic
        );
        assert_eq!(
            with_objective(|m, x, y| {
                let p = m.add(Node::Bin(BinOp::Mul, x, y));
                let n = m.add(Node::Neg(p));
                let (s, l) = m.add_list(&[n, x]);
                m.add(Node::Sum(s, l))
            }),
            Quadratic
        );
        // x^2 * y, x^3, x^0.5, 2^x, x / y, sqrt(x), abs(x), max(x, y)
        assert_eq!(
            with_objective(|m, x, y| {
                let p = m.add(Node::Bin(BinOp::Mul, x, x));
                m.add(Node::Bin(BinOp::Mul, p, y))
            }),
            Smooth
        );
        for (exponent, class) in [(3.0, Smooth), (0.5, Smooth), (-1.0, Smooth), (1.0, Linear), (0.0, Linear)] {
            assert_eq!(
                with_objective(|m, x, _| {
                    let c = num(m, exponent);
                    m.add(Node::Bin(BinOp::Pow, x, c))
                }),
                class,
                "x^{exponent}"
            );
        }
        assert_eq!(
            with_objective(|m, x, _| {
                let two = num(m, 2.0);
                m.add(Node::Bin(BinOp::Pow, two, x))
            }),
            Smooth
        );
        assert_eq!(with_objective(|m, x, y| m.add(Node::Bin(BinOp::Div, x, y))), Smooth);
        assert_eq!(with_objective(|m, x, _| m.add(Node::Fun(Fun::Sqrt, x))), Smooth);
        assert_eq!(with_objective(|m, x, _| m.add(Node::Fun(Fun::Abs, x))), Smooth);
        assert_eq!(
            with_objective(|m, x, y| {
                let (s, l) = m.add_list(&[x, y]);
                m.add(Node::Max(s, l))
            }),
            Smooth
        );
    }

    #[test]
    fn functions_without_expression_are_linear() {
        let mut m = Model::new("t");
        m.variables = vec![Variable::free("x")];
        m.constraints.push(crate::Constraint {
            lower: 0.0,
            upper: 1.0,
            body: Function { linear: vec![(0, 2.0)], expr: None },
        });
        let c = m.classes();
        assert_eq!((c.objective, c.constraints), (FnClass::Linear, vec![FnClass::Linear]));
    }
}
