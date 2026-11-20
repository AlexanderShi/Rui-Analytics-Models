//! The model: variables, functions (a linear part plus an expression), and the expression graph.

use rui_api::{Error, Sense, Variable};

/// Index of a node in [`Model::nodes`].
pub type NodeId = u32;

/// One-argument functions (the operator set of the AMPL `.nl` format).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fun {
    Floor,
    Ceil,
    Abs,
    Tanh,
    Tan,
    Sqrt,
    Sinh,
    Sin,
    Log10,
    /// Natural logarithm.
    Log,
    Exp,
    Cosh,
    Cos,
    Atanh,
    Atan,
    Asinh,
    Asin,
    Acosh,
    Acos,
}

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    /// `atan2(y, x)`, the first operand being `y`.
    Atan2,
}

/// A node of the expression graph.
///
/// A node's operands are nodes created before it: every operand id is smaller than the id of the node
/// that uses it ([`Model::check`] verifies this). The graph therefore has no cycles, and a node may be
/// shared by several parents and several functions. N-ary nodes keep their operands in
/// [`Model::kids`]`[start..start + len]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Node {
    /// A finite constant.
    Num(f64),
    /// Variable `j`, an index into [`Model::variables`].
    Var(u32),
    Bin(BinOp, NodeId, NodeId),
    Neg(NodeId),
    Fun(Fun, NodeId),
    /// Sum of the operands `(start, len)` in [`Model::kids`].
    Sum(u32, u32),
    /// Smallest operand (the first one on ties).
    Min(u32, u32),
    /// Largest operand (the first one on ties).
    Max(u32, u32),
}

/// `Σ_j a_j x_j + e(x)`: a sparse linear part and an optional expression.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Function {
    /// `(variable, coefficient)` pairs, sorted by variable without repeats. Variables that occur only in
    /// the expression may be listed with coefficient 0 (the `.nl` format does so) but need not be.
    pub linear: Vec<(u32, f64)>,
    /// Root of the expression. It may contain linear terms and constants as well.
    pub expr: Option<NodeId>,
}

/// A constraint `lower <= body(x) <= upper`; an absent side is `±f64::INFINITY`, an equality has
/// `lower == upper`.
#[derive(Clone, Debug, PartialEq)]
pub struct Constraint {
    pub lower: f64,
    pub upper: f64,
    pub body: Function,
}

/// An algebraic model: minimize or maximize `f(x)` subject to `g_L <= g(x) <= g_U` and the variables'
/// bounds.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Model {
    pub name: String,
    pub variables: Vec<Variable>,
    pub sense: Sense,
    /// The objective; the zero function (the default) for a feasibility problem.
    pub objective: Function,
    pub constraints: Vec<Constraint>,
    /// The expression graph shared by all functions.
    pub nodes: Vec<Node>,
    /// Operand lists of the n-ary nodes.
    pub kids: Vec<NodeId>,
}

impl Model {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), ..Self::default() }
    }

    /// Appends `node` to the graph and returns its id. Its operands must already be in the graph.
    pub fn add(&mut self, node: Node) -> NodeId {
        self.nodes.push(node);
        (self.nodes.len() - 1) as NodeId
    }

    /// Appends an operand list and returns the `(start, len)` that [`Node::Sum`], [`Node::Min`] and
    /// [`Node::Max`] take.
    pub fn add_list(&mut self, operands: &[NodeId]) -> (u32, u32) {
        let start = self.kids.len() as u32;
        self.kids.extend_from_slice(operands);
        (start, operands.len() as u32)
    }

    /// Checks the structure: operands precede the nodes that use them, operand lists are non-empty and in
    /// range, variable indices exist, constants and coefficients are finite, linear parts are sorted
    /// without repeats. Bounds and start values are not checked here; `rui-models` validates them before
    /// every solve.
    pub fn check(&self) -> Result<(), Error> {
        let fail = |m: String| Err(Error::InvalidProblem(m));
        let n = self.variables.len();
        for (id, node) in self.nodes.iter().enumerate() {
            let operand = |c: NodeId| {
                if (c as usize) < id {
                    Ok(())
                } else {
                    fail(format!("node {id}: operand {c} is not an earlier node"))
                }
            };
            match *node {
                Node::Num(v) if !v.is_finite() => return fail(format!("node {id}: constant {v:?} is not finite")),
                Node::Num(_) => {}
                Node::Var(j) if j as usize >= n => {
                    return fail(format!("node {id}: variable {j} does not exist (the model has {n})"))
                }
                Node::Var(_) => {}
                Node::Bin(_, a, b) => {
                    operand(a)?;
                    operand(b)?;
                }
                Node::Neg(a) | Node::Fun(_, a) => operand(a)?,
                Node::Sum(s, l) | Node::Min(s, l) | Node::Max(s, l) => {
                    let (s, l) = (s as usize, l as usize);
                    if l == 0 {
                        return fail(format!("node {id}: empty operand list"));
                    }
                    if s + l > self.kids.len() {
                        return fail(format!("node {id}: operand list {s}..{} is out of range", s + l));
                    }
                    for &k in &self.kids[s..s + l] {
                        operand(k)?;
                    }
                }
            }
        }
        let function = |who: &str, f: &Function| {
            if let Some(r) = f.expr {
                if r as usize >= self.nodes.len() {
                    return fail(format!("{who}: expression root {r} is not a node"));
                }
            }
            for &(j, c) in &f.linear {
                if j as usize >= n {
                    return fail(format!("{who}: variable {j} does not exist (the model has {n})"));
                }
                if !c.is_finite() {
                    return fail(format!("{who}: coefficient {c:?} of variable {j} is not finite"));
                }
            }
            if f.linear.windows(2).any(|w| w[0].0 >= w[1].0) {
                return fail(format!("{who}: the linear part is not sorted by variable or repeats one"));
            }
            Ok(())
        };
        function("objective", &self.objective)?;
        for (i, c) in self.constraints.iter().enumerate() {
            function(&format!("constraint {i}"), &c.body)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_vars() -> Model {
        let mut m = Model::new("t");
        m.variables = vec![Variable::free("x"), Variable::free("y")];
        m
    }

    fn message(r: Result<(), Error>) -> String {
        match r {
            Err(Error::InvalidProblem(m)) => m,
            other => panic!("expected InvalidProblem, got {other:?}"),
        }
    }

    #[test]
    fn a_well_formed_model_passes() {
        let mut m = two_vars();
        let x = m.add(Node::Var(0));
        let y = m.add(Node::Var(1));
        let (s, l) = m.add_list(&[x, y, x]);
        let sum = m.add(Node::Sum(s, l));
        m.objective = Function { linear: vec![(0, 1.0), (1, 0.0)], expr: Some(sum) };
        assert_eq!(m.check(), Ok(()));
    }

    #[test]
    fn structural_errors_are_reported() {
        let mut m = two_vars();
        m.nodes = vec![Node::Neg(0)];
        assert_eq!(message(m.check()), "node 0: operand 0 is not an earlier node");

        let mut m = two_vars();
        m.add(Node::Var(2));
        assert_eq!(message(m.check()), "node 0: variable 2 does not exist (the model has 2)");

        let mut m = two_vars();
        m.add(Node::Num(f64::NAN));
        assert_eq!(message(m.check()), "node 0: constant NaN is not finite");

        let mut m = two_vars();
        m.add(Node::Var(0));
        m.add(Node::Max(0, 0));
        assert_eq!(message(m.check()), "node 1: empty operand list");

        let mut m = two_vars();
        m.add(Node::Sum(0, 1));
        assert_eq!(message(m.check()), "node 0: operand list 0..1 is out of range");

        let mut m = two_vars();
        m.constraints.push(Constraint {
            lower: 0.0,
            upper: 1.0,
            body: Function { linear: vec![(1, 1.0), (0, 1.0)], expr: None },
        });
        assert_eq!(message(m.check()), "constraint 0: the linear part is not sorted by variable or repeats one");

        let mut m = two_vars();
        m.objective.expr = Some(0);
        assert_eq!(message(m.check()), "objective: expression root 0 is not a node");
    }
}
