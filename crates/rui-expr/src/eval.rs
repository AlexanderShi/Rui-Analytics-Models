//! Exact evaluation of a [`Model`]: values, gradient, Jacobian and Hessian of the Lagrangian, as a
//! [`SmoothOracle`].
//!
//! Each expression is split into *elements* at top-level sums and constant factors (`2*(a + 3*b)` gives
//! the elements `a` and `b` with coefficients 2 and 6). Products `x_i*x_j` and squares `x_i^2` become
//! quadratic terms with closed-form derivatives; every other element is compiled to a small tape.
//! Identical elements (same operations, constants and variables) are stored once and shared by every
//! function that uses them, so a subexpression repeated in many constraints (the AMPL-NLP test problem
//! WM_CFy repeats `t*sqrt(sum y^2/N)` in 378 constraints) is evaluated once per call and its Hessian is
//! computed once with the combined weight `sum coef * multiplier`.
//!
//! Derivatives: reverse mode for gradients; forward-over-reverse with one tangent direction per element
//! variable for the element's dense Hessian. The Hessian structure is the union of the element cliques
//! (lower triangle), plus the entries of the quadratic terms. The Jacobian structure of a constraint is
//! its linear variables together with the variables of its expression; for a `.nl` file that is exactly
//! the constraint's `J` segment, as in the AMPL solver library.
//!
//! The objective is evaluated as written, in the model's [`Sense`]; `rui-models` turns maximization
//! into minimization before a solver sees the problem.
//!
//! An element value or derivative that is not finite (log of a negative number, division by zero, ...)
//! makes the call return an [`EvalError`]; solvers treat it as recoverable (a shorter step).

use crate::model::{BinOp, Fun, Function, Model, Node, NodeId};
use rui_api::{Access, Error, EvalError, FunctionClasses, Problem, Sense, SmoothOracle, Variable};
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Var(u32),
    Const(f64),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Div(u32, u32),
    Pow(u32, u32),
    PowC(u32, f64),
    Scale(u32, f64),
    Neg(u32),
    F(Fun, u32),
    Atan2(u32, u32),
    /// `(start, len)` into `args`, whose entries are element-relative op indices.
    Sum(u32, u32),
    Min(u32, u32),
    Max(u32, u32),
}

#[derive(Clone, Copy, Debug)]
struct Element {
    op_start: u32,
    op_len: u32,
    var_start: u32,
    k: u32,
    /// `k(k+1)/2` Hessian positions, local pairs `(s, t)` with `t <= s` in row order.
    hpos_start: u32,
}

#[derive(Clone, Copy, Debug)]
struct Use {
    el: u32,
    coef: f64,
    /// Constraint uses: `k` Jacobian positions of the element's variables.
    jpos_start: u32,
}

#[derive(Clone, Copy, Debug)]
struct Quad {
    a: u32,
    b: u32,
    c: f64,
    hpos: u32,
    ja: u32,
    jb: u32,
}

/// Size statistics, for logs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvalStats {
    /// Uses of general (tape) elements, over all functions.
    pub elements: usize,
    /// Distinct general elements.
    pub unique_elements: usize,
    pub quadratic_terms: usize,
    pub largest_element_vars: usize,
    pub largest_element_ops: usize,
    pub jacobian_nnz: usize,
    pub hessian_nnz: usize,
}

#[derive(Default)]
struct Scratch {
    v: Vec<f64>,
    adj: Vec<f64>,
    dot: Vec<f64>,
    adjdot: Vec<f64>,
    h: Vec<f64>,
    g: Vec<f64>,
    weight: Vec<f64>,
    el_val: Vec<f64>,
    el_grad: Vec<f64>,
}

/// A [`Model`] with exact first and second derivatives.
///
/// Evaluation uses internal scratch space, so an `ExprProblem` is `Send` but not `Sync`: one solve at a
/// time per instance.
pub struct ExprProblem {
    name: String,
    variables: Vec<Variable>,
    sense: Sense,
    classes: FunctionClasses,
    m: usize,
    gl: Vec<f64>,
    gu: Vec<f64>,
    obj_const: f64,
    obj_lin: Vec<(u32, f64)>,
    con_const: Vec<f64>,
    row_ptr: Vec<u32>,
    jcol: Vec<u32>,
    jlin: Vec<f64>,
    ops: Vec<Op>,
    args: Vec<u32>,
    vars: Vec<u32>,
    elements: Vec<Element>,
    uses: Vec<Use>,
    /// Uses of function `f` (0 = objective, `i + 1` = constraint `i`): `use_ptr[f]..use_ptr[f+1]`.
    use_ptr: Vec<u32>,
    jpos: Vec<u32>,
    quads: Vec<Quad>,
    quad_ptr: Vec<u32>,
    hpos: Vec<u32>,
    hess: Vec<(usize, usize)>,
    /// Elements used by the objective / by some constraint.
    obj_elements: Vec<u32>,
    con_elements: Vec<u32>,
    max_ops: usize,
    max_k: usize,
    stats: EvalStats,
    scratch: RefCell<Scratch>,
}

fn is_num(model: &Model, id: NodeId) -> Option<f64> {
    match model.nodes[id as usize] {
        Node::Num(c) => Some(c),
        _ => None,
    }
}

fn operands(model: &Model, s: u32, l: u32) -> &[NodeId] {
    &model.kids[s as usize..(s + l) as usize]
}

/// Splits `root` into `(coef, element root)` pairs and a constant.
fn split(model: &Model, root: NodeId, out: &mut Vec<(f64, NodeId)>, constant: &mut f64) {
    let mut stack = vec![(1.0f64, root)];
    while let Some((c, id)) = stack.pop() {
        match model.nodes[id as usize] {
            Node::Num(v) => *constant += c * v,
            Node::Bin(BinOp::Add, a, b) => {
                stack.push((c, b));
                stack.push((c, a));
            }
            Node::Bin(BinOp::Sub, a, b) => {
                stack.push((-c, b));
                stack.push((c, a));
            }
            Node::Sum(s, l) => {
                for &k in operands(model, s, l).iter().rev() {
                    stack.push((c, k));
                }
            }
            Node::Neg(a) => stack.push((-c, a)),
            Node::Bin(BinOp::Mul, a, b) if is_num(model, a).is_some() => stack.push((c * is_num(model, a).unwrap(), b)),
            Node::Bin(BinOp::Mul, a, b) if is_num(model, b).is_some() => stack.push((c * is_num(model, b).unwrap(), a)),
            Node::Bin(BinOp::Div, a, b) if is_num(model, b).is_some_and(|d| d != 0.0) => {
                stack.push((c / is_num(model, b).unwrap(), a))
            }
            _ => {
                if c != 0.0 {
                    out.push((c, id))
                }
            }
        }
    }
}

/// Element kinds recognized before compiling a tape.
enum Shape {
    Linear(u32),
    Quad(u32, u32),
    General,
}

fn shape(model: &Model, id: NodeId) -> Shape {
    let node = |i: NodeId| model.nodes[i as usize];
    match node(id) {
        Node::Var(j) => Shape::Linear(j),
        Node::Bin(BinOp::Mul, a, b) => match (node(a), node(b)) {
            (Node::Var(i), Node::Var(j)) => Shape::Quad(i, j),
            _ => Shape::General,
        },
        Node::Bin(BinOp::Pow, a, b) => match (node(a), node(b)) {
            (Node::Var(i), Node::Num(2.0)) => Shape::Quad(i, i),
            _ => Shape::General,
        },
        _ => Shape::General,
    }
}

/// Every variable reachable from `root`, once each (nodes already stamped with `mark` are skipped, so
/// shared subgraphs are visited once).
fn variables_of(model: &Model, root: NodeId, mark: u32, stamp: &mut [u32], out: &mut Vec<u32>) {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut stamp[id as usize], mark) == mark {
            continue;
        }
        match model.nodes[id as usize] {
            Node::Num(_) => {}
            Node::Var(j) => out.push(j),
            Node::Bin(_, a, b) => {
                stack.push(a);
                stack.push(b);
            }
            Node::Neg(a) | Node::Fun(_, a) => stack.push(a),
            Node::Sum(s, l) | Node::Min(s, l) | Node::Max(s, l) => stack.extend_from_slice(operands(model, s, l)),
        }
    }
}

/// Compiles the subgraph at `root` into ops (post order) with local variable slots. A node shared
/// inside the subgraph is compiled once.
fn compile(model: &Model, root: NodeId, ops: &mut Vec<Op>, args: &mut Vec<u32>, vars: &mut Vec<u32>) {
    let mut slot: HashMap<u32, u32> = HashMap::new();
    let mut index: HashMap<NodeId, u32> = HashMap::new();
    // iterative post order: (node, children pushed?)
    let mut stack = vec![(root, false)];
    while let Some((id, expanded)) = stack.pop() {
        if index.contains_key(&id) {
            continue;
        }
        let node = model.nodes[id as usize];
        if !expanded {
            stack.push((id, true));
            match node {
                Node::Num(_) | Node::Var(_) => {}
                Node::Bin(BinOp::Mul, a, b) if is_num(model, a).is_some() => stack.push((b, false)),
                Node::Bin(BinOp::Mul, a, b) if is_num(model, b).is_some() => stack.push((a, false)),
                Node::Bin(BinOp::Div, a, b) if is_num(model, b).is_some_and(|d| d != 0.0) => stack.push((a, false)),
                Node::Bin(BinOp::Pow, a, b) if is_num(model, b).is_some() => stack.push((a, false)),
                Node::Bin(_, a, b) => {
                    stack.push((b, false));
                    stack.push((a, false));
                }
                Node::Neg(a) | Node::Fun(_, a) => stack.push((a, false)),
                Node::Sum(s, l) | Node::Min(s, l) | Node::Max(s, l) => {
                    for &k in operands(model, s, l).iter().rev() {
                        stack.push((k, false));
                    }
                }
            }
            continue;
        }
        let at = |n: NodeId| index[&n];
        let op = match node {
            Node::Num(c) => Op::Const(c),
            Node::Var(j) => {
                let next = vars.len() as u32;
                let s = *slot.entry(j).or_insert_with(|| {
                    vars.push(j);
                    next
                });
                Op::Var(s)
            }
            Node::Bin(BinOp::Mul, a, b) if is_num(model, a).is_some() => Op::Scale(at(b), is_num(model, a).unwrap()),
            Node::Bin(BinOp::Mul, a, b) if is_num(model, b).is_some() => Op::Scale(at(a), is_num(model, b).unwrap()),
            Node::Bin(BinOp::Div, a, b) if is_num(model, b).is_some_and(|d| d != 0.0) => {
                Op::Scale(at(a), 1.0 / is_num(model, b).unwrap())
            }
            Node::Bin(BinOp::Pow, a, b) if is_num(model, b).is_some() => Op::PowC(at(a), is_num(model, b).unwrap()),
            Node::Bin(o, a, b) => {
                let (a, b) = (at(a), at(b));
                match o {
                    BinOp::Add => Op::Add(a, b),
                    BinOp::Sub => Op::Sub(a, b),
                    BinOp::Mul => Op::Mul(a, b),
                    BinOp::Div => Op::Div(a, b),
                    BinOp::Pow => Op::Pow(a, b),
                    BinOp::Atan2 => Op::Atan2(a, b),
                }
            }
            Node::Neg(a) => Op::Neg(at(a)),
            Node::Fun(g, a) => Op::F(g, at(a)),
            Node::Sum(s, l) | Node::Min(s, l) | Node::Max(s, l) => {
                let start = args.len() as u32;
                for &k in operands(model, s, l) {
                    args.push(at(k));
                }
                match node {
                    Node::Sum(..) => Op::Sum(start, l),
                    Node::Min(..) => Op::Min(start, l),
                    _ => Op::Max(start, l),
                }
            }
        };
        index.insert(id, ops.len() as u32);
        ops.push(op);
    }
}

/// Value, first and second derivative of a one-argument function.
fn fun3(g: Fun, a: f64) -> (f64, f64, f64) {
    match g {
        Fun::Floor => (a.floor(), 0.0, 0.0),
        Fun::Ceil => (a.ceil(), 0.0, 0.0),
        Fun::Abs => (a.abs(), if a >= 0.0 { 1.0 } else { -1.0 }, 0.0),
        Fun::Exp => {
            let e = a.exp();
            (e, e, e)
        }
        Fun::Log => (a.ln(), 1.0 / a, -1.0 / (a * a)),
        Fun::Log10 => {
            let l = std::f64::consts::LN_10;
            (a.log10(), 1.0 / (a * l), -1.0 / (a * a * l))
        }
        Fun::Sqrt => {
            let s = a.sqrt();
            (s, 0.5 / s, -0.25 / (s * a))
        }
        Fun::Sin => (a.sin(), a.cos(), -a.sin()),
        Fun::Cos => (a.cos(), -a.sin(), -a.cos()),
        Fun::Tan => {
            let t = a.tan();
            let d = 1.0 + t * t;
            (t, d, 2.0 * t * d)
        }
        Fun::Sinh => (a.sinh(), a.cosh(), a.sinh()),
        Fun::Cosh => (a.cosh(), a.sinh(), a.cosh()),
        Fun::Tanh => {
            let t = a.tanh();
            let d = 1.0 - t * t;
            (t, d, -2.0 * t * d)
        }
        Fun::Atan => {
            let d = 1.0 / (1.0 + a * a);
            (a.atan(), d, -2.0 * a * d * d)
        }
        Fun::Asin => {
            let r = 1.0 - a * a;
            (a.asin(), 1.0 / r.sqrt(), a / (r * r.sqrt()))
        }
        Fun::Acos => {
            let r = 1.0 - a * a;
            (a.acos(), -1.0 / r.sqrt(), -a / (r * r.sqrt()))
        }
        Fun::Asinh => {
            let r = 1.0 + a * a;
            (a.asinh(), 1.0 / r.sqrt(), -a / (r * r.sqrt()))
        }
        Fun::Acosh => {
            let r = a * a - 1.0;
            (a.acosh(), 1.0 / r.sqrt(), -a / (r * r.sqrt()))
        }
        Fun::Atanh => {
            let r = 1.0 - a * a;
            (a.atanh(), 1.0 / r, 2.0 * a / (r * r))
        }
    }
}

/// `a^c` for a constant exponent: value, first and second derivative.
fn powc3(a: f64, c: f64) -> (f64, f64, f64) {
    if c == 2.0 {
        return (a * a, 2.0 * a, 2.0);
    }
    if c == 1.0 {
        return (a, 1.0, 0.0);
    }
    if c == 0.0 {
        return (1.0, 0.0, 0.0);
    }
    let d2 = if c * (c - 1.0) == 0.0 { 0.0 } else { c * (c - 1.0) * a.powf(c - 2.0) };
    (a.powf(c), c * a.powf(c - 1.0), d2)
}

/// `a^b` with both arguments variable: value and partials `(f_a, f_b, f_aa, f_ab, f_bb)`.
fn pow6(a: f64, b: f64) -> (f64, f64, f64, f64, f64, f64) {
    let v = a.powf(b);
    let l = a.ln();
    let fa = if b == 0.0 { 0.0 } else { b * a.powf(b - 1.0) };
    let fb = v * l;
    let faa = if b * (b - 1.0) == 0.0 { 0.0 } else { b * (b - 1.0) * a.powf(b - 2.0) };
    let fab = a.powf(b - 1.0) * (1.0 + b * l);
    let fbb = v * l * l;
    (v, fa, fb, faa, fab, fbb)
}

/// Index of the selected argument of a min/max list (the first extreme one).
fn pick(v: &[f64], args: &[u32], max: bool) -> usize {
    let mut best = 0;
    for (k, &a) in args.iter().enumerate() {
        let (x, y) = (v[a as usize], v[args[best] as usize]);
        if (max && x > y) || (!max && x < y) {
            best = k;
        }
    }
    best
}

fn func_name(func: usize) -> String {
    if func == 0 {
        "objective".into()
    } else {
        format!("constraint {}", func - 1)
    }
}

impl ExprProblem {
    /// Checks the model ([`Model::check`]), classifies its functions and prepares the evaluation.
    /// Fails on a structural error and on a constant subexpression that is not finite (`log(0)`).
    pub fn new(model: &Model) -> Result<ExprProblem, Error> {
        model.check()?;
        let m = model.constraints.len();
        // Jacobian structure (CSR): each row is the union of the constraint's linear variables and the
        // variables of its expression, sorted; linear coefficients default to 0.
        let mut row_ptr = Vec::with_capacity(m + 1);
        let (mut jcol, mut jlin) = (Vec::new(), Vec::new());
        row_ptr.push(0u32);
        let mut stamp = vec![u32::MAX; model.nodes.len()];
        let (mut row, mut found) = (Vec::new(), Vec::new());
        for (i, c) in model.constraints.iter().enumerate() {
            row.clear();
            row.extend_from_slice(&c.body.linear);
            if let Some(r) = c.body.expr {
                found.clear();
                variables_of(model, r, i as u32, &mut stamp, &mut found);
                row.extend(found.iter().map(|&j| (j, 0.0)));
            }
            // stable sort, then keep the first entry of each variable: the linear coefficient wins
            row.sort_by_key(|e: &(u32, f64)| e.0);
            row.dedup_by_key(|e| e.0);
            for &(j, a) in &row {
                jcol.push(j);
                jlin.push(a);
            }
            row_ptr.push(jcol.len() as u32);
        }
        let find_j = |i: usize, j: u32, jcol: &[u32], row_ptr: &[u32]| -> u32 {
            let (s, e) = (row_ptr[i] as usize, row_ptr[i + 1] as usize);
            let p = jcol[s..e].binary_search(&j).expect("a row lists every variable of its expression");
            (s + p) as u32
        };
        let mut obj_lin: Vec<(u32, f64)> = model.objective.linear.clone();
        let mut con_const = vec![0.0; m];
        let mut obj_const = 0.0;
        let mut p = ExprProblem {
            name: model.name.clone(),
            variables: model.variables.clone(),
            sense: model.sense,
            classes: model.classes(),
            m,
            gl: model.constraints.iter().map(|c| c.lower).collect(),
            gu: model.constraints.iter().map(|c| c.upper).collect(),
            obj_const: 0.0,
            obj_lin: Vec::new(),
            con_const: Vec::new(),
            row_ptr,
            jcol,
            jlin,
            ops: Vec::new(),
            args: Vec::new(),
            vars: Vec::new(),
            elements: Vec::new(),
            uses: Vec::new(),
            use_ptr: vec![0],
            jpos: Vec::new(),
            quads: Vec::new(),
            quad_ptr: vec![0],
            hpos: Vec::new(),
            hess: Vec::new(),
            obj_elements: Vec::new(),
            con_elements: Vec::new(),
            max_ops: 1,
            max_k: 1,
            stats: EvalStats::default(),
            scratch: RefCell::new(Scratch::default()),
        };
        // elements per function; tapes compiled once, deduplicated by content
        let mut dedup: HashMap<Vec<u64>, u32> = HashMap::new();
        let mut parts = Vec::new();
        let (mut tops, mut targs, mut tvars) = (Vec::new(), Vec::new(), Vec::new());
        for func in 0..=m {
            let f: &Function = if func == 0 { &model.objective } else { &model.constraints[func - 1].body };
            parts.clear();
            let mut constant = 0.0;
            if let Some(r) = f.expr {
                split(model, r, &mut parts, &mut constant);
            }
            for &(coef, id) in parts.iter() {
                match shape(model, id) {
                    Shape::Linear(j) => {
                        if func == 0 {
                            match obj_lin.iter_mut().find(|e| e.0 == j) {
                                Some(e) => e.1 += coef,
                                None => obj_lin.push((j, coef)),
                            }
                        } else {
                            let q = find_j(func - 1, j, &p.jcol, &p.row_ptr);
                            p.jlin[q as usize] += coef;
                        }
                    }
                    Shape::Quad(a, b) => {
                        let (ja, jb) = if func == 0 {
                            (u32::MAX, u32::MAX)
                        } else {
                            (find_j(func - 1, a, &p.jcol, &p.row_ptr), find_j(func - 1, b, &p.jcol, &p.row_ptr))
                        };
                        p.quads.push(Quad { a, b, c: coef, hpos: 0, ja, jb });
                    }
                    Shape::General => {
                        tops.clear();
                        targs.clear();
                        tvars.clear();
                        compile(model, id, &mut tops, &mut targs, &mut tvars);
                        if tvars.is_empty() {
                            // constant subexpression
                            let mut v = vec![0.0; tops.len()];
                            let val = forward(&tops, &targs, &[], &mut v);
                            if !val.is_finite() {
                                return Err(Error::InvalidProblem(format!(
                                    "{}: a constant subexpression is not finite",
                                    func_name(func)
                                )));
                            }
                            constant += coef * val;
                            continue;
                        }
                        let key = element_key(&tops, &targs, &tvars);
                        let el = match dedup.get(&key) {
                            Some(&e) => e,
                            None => {
                                let e = p.elements.len() as u32;
                                let op_start = p.ops.len() as u32;
                                let arg_base = p.args.len() as u32;
                                for op in &tops {
                                    p.ops.push(match *op {
                                        Op::Sum(s, l) => Op::Sum(s + arg_base, l),
                                        Op::Min(s, l) => Op::Min(s + arg_base, l),
                                        Op::Max(s, l) => Op::Max(s + arg_base, l),
                                        o => o,
                                    });
                                }
                                p.args.extend_from_slice(&targs);
                                let var_start = p.vars.len() as u32;
                                p.vars.extend_from_slice(&tvars);
                                p.elements.push(Element {
                                    op_start,
                                    op_len: tops.len() as u32,
                                    var_start,
                                    k: tvars.len() as u32,
                                    hpos_start: 0,
                                });
                                p.max_ops = p.max_ops.max(tops.len());
                                p.max_k = p.max_k.max(tvars.len());
                                dedup.insert(key, e);
                                e
                            }
                        };
                        let jpos_start = p.jpos.len() as u32;
                        if func > 0 {
                            let e = p.elements[el as usize];
                            for s in 0..e.k {
                                let j = p.vars[(e.var_start + s) as usize];
                                let q = find_j(func - 1, j, &p.jcol, &p.row_ptr);
                                p.jpos.push(q);
                            }
                        }
                        p.uses.push(Use { el, coef, jpos_start });
                        p.stats.elements += 1;
                    }
                }
            }
            if func == 0 {
                obj_const = constant;
            } else {
                con_const[func - 1] = constant;
            }
            p.use_ptr.push(p.uses.len() as u32);
            p.quad_ptr.push(p.quads.len() as u32);
        }
        obj_lin.sort_by_key(|e| e.0);
        p.obj_lin = obj_lin;
        p.obj_const = obj_const;
        p.con_const = con_const;
        // elements needed by the objective / by the constraints
        let mut in_obj = vec![false; p.elements.len()];
        let mut in_con = vec![false; p.elements.len()];
        for (u, use_) in p.uses.iter().enumerate() {
            if (u as u32) < p.use_ptr[1] {
                in_obj[use_.el as usize] = true;
            } else {
                in_con[use_.el as usize] = true;
            }
        }
        p.obj_elements = (0..p.elements.len() as u32).filter(|&e| in_obj[e as usize]).collect();
        p.con_elements = (0..p.elements.len() as u32).filter(|&e| in_con[e as usize]).collect();
        // Hessian structure: lower triangle (row >= col) of element cliques and quadratic terms
        let key = |i: u32, j: u32| -> u64 {
            let (r, c) = if i >= j { (i, j) } else { (j, i) };
            ((r as u64) << 32) | c as u64
        };
        let mut keys: Vec<u64> = Vec::new();
        for e in &p.elements {
            let vs = &p.vars[e.var_start as usize..(e.var_start + e.k) as usize];
            for s in 0..vs.len() {
                for t in 0..=s {
                    keys.push(key(vs[s], vs[t]));
                }
            }
        }
        for q in &p.quads {
            keys.push(key(q.a, q.b));
        }
        keys.sort_unstable();
        keys.dedup();
        let pos = |k: u64| keys.binary_search(&k).expect("key collected above") as u32;
        for e in p.elements.iter_mut() {
            e.hpos_start = p.hpos.len() as u32;
            let vs = &p.vars[e.var_start as usize..(e.var_start + e.k) as usize];
            for s in 0..vs.len() {
                for t in 0..=s {
                    p.hpos.push(pos(key(vs[s], vs[t])));
                }
            }
        }
        for q in p.quads.iter_mut() {
            q.hpos = pos(key(q.a, q.b));
        }
        p.hess = keys.iter().map(|&k| ((k >> 32) as usize, (k & 0xffff_ffff) as usize)).collect();
        p.stats.unique_elements = p.elements.len();
        p.stats.quadratic_terms = p.quads.len();
        p.stats.largest_element_vars = p.elements.iter().map(|e| e.k as usize).max().unwrap_or(0);
        p.stats.largest_element_ops = p.elements.iter().map(|e| e.op_len as usize).max().unwrap_or(0);
        p.stats.jacobian_nnz = p.jcol.len();
        p.stats.hessian_nnz = p.hess.len();
        {
            let mut s = p.scratch.borrow_mut();
            s.v = vec![0.0; p.max_ops];
            s.adj = vec![0.0; p.max_ops];
            s.dot = vec![0.0; p.max_ops * p.max_k];
            s.adjdot = vec![0.0; p.max_ops * p.max_k];
            s.h = vec![0.0; p.max_k * p.max_k];
            s.g = vec![0.0; p.max_k];
            s.weight = vec![0.0; p.elements.len()];
            s.el_val = vec![0.0; p.elements.len()];
            s.el_grad = vec![0.0; p.vars.len()];
        }
        Ok(p)
    }

    /// Size statistics of the compiled model.
    pub fn stats(&self) -> &EvalStats {
        &self.stats
    }

    fn el_ops(&self, e: &Element) -> &[Op] {
        &self.ops[e.op_start as usize..(e.op_start + e.op_len) as usize]
    }

    fn el_x<'a>(&self, e: &Element, x: &[f64], buf: &'a mut Vec<f64>) -> &'a [f64] {
        buf.clear();
        buf.extend(self.vars[e.var_start as usize..(e.var_start + e.k) as usize].iter().map(|&j| x[j as usize]));
        buf
    }

    fn obj_value(&self, x: &[f64]) -> Result<f64, EvalError> {
        let mut s = self.scratch.borrow_mut();
        let s = &mut *s;
        let mut f = self.obj_const;
        for &(j, c) in &self.obj_lin {
            f += c * x[j as usize];
        }
        for q in &self.quads[self.quad_ptr[0] as usize..self.quad_ptr[1] as usize] {
            f += q.c * x[q.a as usize] * x[q.b as usize];
        }
        let mut xl = Vec::with_capacity(self.max_k);
        for u in &self.uses[self.use_ptr[0] as usize..self.use_ptr[1] as usize] {
            let e = self.elements[u.el as usize];
            let xe = self.el_x(&e, x, &mut xl);
            let v = forward(self.el_ops(&e), &self.args, xe, &mut s.v);
            f += u.coef * v;
        }
        if !f.is_finite() {
            return Err(EvalError::new("objective is not finite"));
        }
        Ok(f)
    }

    /// Values of the constraint elements (each unique element once).
    fn con_element_values(&self, x: &[f64], s: &mut Scratch) -> Result<(), EvalError> {
        let mut xl = Vec::with_capacity(self.max_k);
        for &el in &self.con_elements {
            let e = self.elements[el as usize];
            let xe = self.el_x(&e, x, &mut xl);
            let v = forward(self.el_ops(&e), &self.args, xe, &mut s.v);
            if !v.is_finite() {
                return Err(EvalError::new(format!("element {el} is not finite")));
            }
            s.el_val[el as usize] = v;
        }
        Ok(())
    }

    /// Values and gradients of the elements in `list` (gradients into `el_grad`).
    fn element_gradients(&self, x: &[f64], list: &[u32], s: &mut Scratch) -> Result<(), EvalError> {
        let mut xl = Vec::with_capacity(self.max_k);
        for &el in list {
            let e = self.elements[el as usize];
            let xe = self.el_x(&e, x, &mut xl);
            let (vs, k) = (e.var_start as usize, e.k as usize);
            let g = &mut s.el_grad[vs..vs + k];
            let v = reverse(self.el_ops(&e), &self.args, xe, &mut s.v, &mut s.adj, g);
            if !v.is_finite() || g.iter().any(|d| !d.is_finite()) {
                return Err(EvalError::new(format!("element {el}: value or gradient is not finite")));
            }
            s.el_val[el as usize] = v;
        }
        Ok(())
    }
}

/// Canonical content of an element: ops (with constants' bits) and its variables.
fn element_key(ops: &[Op], args: &[u32], vars: &[u32]) -> Vec<u64> {
    let mut k = Vec::with_capacity(2 * ops.len() + vars.len() + 1);
    let pair = |t: u64, a: u32, b: u32| (t << 56) | ((a as u64) << 28) | b as u64;
    for op in ops {
        match *op {
            Op::Var(s) => k.push(pair(1, s, 0)),
            Op::Const(c) => {
                k.push(pair(2, 0, 0));
                k.push(c.to_bits());
            }
            Op::Add(a, b) => k.push(pair(3, a, b)),
            Op::Sub(a, b) => k.push(pair(4, a, b)),
            Op::Mul(a, b) => k.push(pair(5, a, b)),
            Op::Div(a, b) => k.push(pair(6, a, b)),
            Op::Pow(a, b) => k.push(pair(7, a, b)),
            Op::PowC(a, c) => {
                k.push(pair(8, a, 0));
                k.push(c.to_bits());
            }
            Op::Scale(a, c) => {
                k.push(pair(9, a, 0));
                k.push(c.to_bits());
            }
            Op::Neg(a) => k.push(pair(10, a, 0)),
            Op::F(g, a) => k.push(pair(11, a, g as u32)),
            Op::Atan2(a, b) => k.push(pair(12, a, b)),
            Op::Sum(s, l) | Op::Min(s, l) | Op::Max(s, l) => {
                let t = match op {
                    Op::Sum(..) => 13,
                    Op::Min(..) => 14,
                    _ => 15,
                };
                k.push(pair(t, l, 0));
                k.extend(args[s as usize..(s + l) as usize].iter().map(|&a| a as u64));
            }
        }
    }
    k.push(u64::MAX);
    k.extend(vars.iter().map(|&j| j as u64));
    k
}

/// Forward sweep: values of all ops; returns the element value. `xe` are the element's variable values
/// (local slots).
fn forward(ops: &[Op], args: &[u32], xe: &[f64], v: &mut [f64]) -> f64 {
    for (i, op) in ops.iter().enumerate() {
        v[i] = match *op {
            Op::Var(s) => xe[s as usize],
            Op::Const(c) => c,
            Op::Add(a, b) => v[a as usize] + v[b as usize],
            Op::Sub(a, b) => v[a as usize] - v[b as usize],
            Op::Mul(a, b) => v[a as usize] * v[b as usize],
            Op::Div(a, b) => v[a as usize] / v[b as usize],
            Op::Pow(a, b) => v[a as usize].powf(v[b as usize]),
            Op::PowC(a, c) => powc3(v[a as usize], c).0,
            Op::Scale(a, c) => c * v[a as usize],
            Op::Neg(a) => -v[a as usize],
            Op::F(g, a) => fun3(g, v[a as usize]).0,
            Op::Atan2(a, b) => v[a as usize].atan2(v[b as usize]),
            Op::Sum(s, l) => args[s as usize..(s + l) as usize].iter().map(|&a| v[a as usize]).sum(),
            Op::Min(s, l) | Op::Max(s, l) => {
                let ar = &args[s as usize..(s + l) as usize];
                v[ar[pick(v, ar, matches!(op, Op::Max(..)))] as usize]
            }
        };
    }
    v[ops.len() - 1]
}

/// Forward and reverse sweep: element value, gradient into `g` (local slots, overwritten).
fn reverse(ops: &[Op], args: &[u32], xe: &[f64], v: &mut [f64], adj: &mut [f64], g: &mut [f64]) -> f64 {
    let val = forward(ops, args, xe, v);
    let len = ops.len();
    adj[..len].fill(0.0);
    g.fill(0.0);
    adj[len - 1] = 1.0;
    for i in (0..len).rev() {
        let ai = adj[i];
        if ai == 0.0 {
            continue;
        }
        match ops[i] {
            Op::Var(s) => g[s as usize] += ai,
            Op::Const(_) => {}
            Op::Add(a, b) => {
                adj[a as usize] += ai;
                adj[b as usize] += ai;
            }
            Op::Sub(a, b) => {
                adj[a as usize] += ai;
                adj[b as usize] -= ai;
            }
            Op::Mul(a, b) => {
                let (va, vb) = (v[a as usize], v[b as usize]);
                adj[a as usize] += ai * vb;
                adj[b as usize] += ai * va;
            }
            Op::Div(a, b) => {
                let vb = v[b as usize];
                adj[a as usize] += ai / vb;
                adj[b as usize] -= ai * v[i] / vb;
            }
            Op::Pow(a, b) => {
                let (_, fa, fb, _, _, _) = pow6(v[a as usize], v[b as usize]);
                adj[a as usize] += ai * fa;
                adj[b as usize] += ai * fb;
            }
            Op::PowC(a, c) => adj[a as usize] += ai * powc3(v[a as usize], c).1,
            Op::Scale(a, c) => adj[a as usize] += ai * c,
            Op::Neg(a) => adj[a as usize] -= ai,
            Op::F(f, a) => adj[a as usize] += ai * fun3(f, v[a as usize]).1,
            Op::Atan2(a, b) => {
                let (y, x) = (v[a as usize], v[b as usize]);
                let r = y * y + x * x;
                adj[a as usize] += ai * x / r;
                adj[b as usize] -= ai * y / r;
            }
            Op::Sum(s, l) => {
                for &a in &args[s as usize..(s + l) as usize] {
                    adj[a as usize] += ai;
                }
            }
            Op::Min(s, l) | Op::Max(s, l) => {
                let ar = &args[s as usize..(s + l) as usize];
                let w = ar[pick(v, ar, matches!(ops[i], Op::Max(..)))];
                adj[w as usize] += ai;
            }
        }
    }
    val
}

/// `out += c * src` over `k` entries.
#[inline]
fn axpy(out: &mut [f64], c: f64, src: &[f64]) {
    for (o, s) in out.iter_mut().zip(src) {
        *o += c * s;
    }
}

/// Dense Hessian of one element (local `k x k`, full) by forward-over-reverse.
#[allow(clippy::too_many_arguments)]
fn hessian(
    ops: &[Op],
    args: &[u32],
    xe: &[f64],
    k: usize,
    v: &mut [f64],
    adj: &mut [f64],
    dot: &mut [f64],
    adjdot: &mut [f64],
    h: &mut [f64],
) -> f64 {
    let len = ops.len();
    let val = forward(ops, args, xe, v);
    // tangents: dot[i*k..(i+1)*k]
    dot[..len * k].fill(0.0);
    for i in 0..len {
        let (before, cur) = dot.split_at_mut(i * k);
        let d = &mut cur[..k];
        match ops[i] {
            Op::Var(s) => d[s as usize] = 1.0,
            Op::Const(_) => {}
            Op::Add(a, b) => {
                axpy(d, 1.0, &before[a as usize * k..a as usize * k + k]);
                axpy(d, 1.0, &before[b as usize * k..b as usize * k + k]);
            }
            Op::Sub(a, b) => {
                axpy(d, 1.0, &before[a as usize * k..a as usize * k + k]);
                axpy(d, -1.0, &before[b as usize * k..b as usize * k + k]);
            }
            Op::Mul(a, b) => {
                let (va, vb) = (v[a as usize], v[b as usize]);
                axpy(d, vb, &before[a as usize * k..a as usize * k + k]);
                axpy(d, va, &before[b as usize * k..b as usize * k + k]);
            }
            Op::Div(a, b) => {
                let vb = v[b as usize];
                axpy(d, 1.0 / vb, &before[a as usize * k..a as usize * k + k]);
                axpy(d, -v[i] / vb, &before[b as usize * k..b as usize * k + k]);
            }
            Op::Pow(a, b) => {
                let (_, fa, fb, _, _, _) = pow6(v[a as usize], v[b as usize]);
                axpy(d, fa, &before[a as usize * k..a as usize * k + k]);
                axpy(d, fb, &before[b as usize * k..b as usize * k + k]);
            }
            Op::PowC(a, c) => axpy(d, powc3(v[a as usize], c).1, &before[a as usize * k..a as usize * k + k]),
            Op::Scale(a, c) => axpy(d, c, &before[a as usize * k..a as usize * k + k]),
            Op::Neg(a) => axpy(d, -1.0, &before[a as usize * k..a as usize * k + k]),
            Op::F(f, a) => axpy(d, fun3(f, v[a as usize]).1, &before[a as usize * k..a as usize * k + k]),
            Op::Atan2(a, b) => {
                let (y, x) = (v[a as usize], v[b as usize]);
                let r = y * y + x * x;
                axpy(d, x / r, &before[a as usize * k..a as usize * k + k]);
                axpy(d, -y / r, &before[b as usize * k..b as usize * k + k]);
            }
            Op::Sum(s, l) => {
                for &a in &args[s as usize..(s + l) as usize] {
                    axpy(d, 1.0, &before[a as usize * k..a as usize * k + k]);
                }
            }
            Op::Min(s, l) | Op::Max(s, l) => {
                let ar = &args[s as usize..(s + l) as usize];
                let w = ar[pick(v, ar, matches!(ops[i], Op::Max(..)))] as usize;
                axpy(d, 1.0, &before[w * k..w * k + k]);
            }
        }
    }
    // reverse with adjoint tangents
    adj[..len].fill(0.0);
    adjdot[..len * k].fill(0.0);
    h[..k * k].fill(0.0);
    adj[len - 1] = 1.0;
    for i in (0..len).rev() {
        let ai = adj[i];
        // adjdot of op i is final now; operands have smaller indices
        let (lo, hi) = adjdot.split_at_mut(i * k);
        let adi = &hi[..k];
        let zero_adi = adi.iter().all(|&t| t == 0.0);
        if ai == 0.0 && zero_adi {
            continue;
        }
        macro_rules! dot_of {
            ($a:expr) => {
                &dot[$a as usize * k..$a as usize * k + k]
            };
        }
        macro_rules! ad {
            ($a:expr) => {
                &mut lo[$a as usize * k..$a as usize * k + k]
            };
        }
        match ops[i] {
            Op::Var(s) => {
                let row = &mut h[s as usize * k..s as usize * k + k];
                axpy(row, 1.0, adi);
            }
            Op::Const(_) => {}
            Op::Add(a, b) => {
                adj[a as usize] += ai;
                adj[b as usize] += ai;
                axpy(ad!(a), 1.0, adi);
                axpy(ad!(b), 1.0, adi);
            }
            Op::Sub(a, b) => {
                adj[a as usize] += ai;
                adj[b as usize] -= ai;
                axpy(ad!(a), 1.0, adi);
                axpy(ad!(b), -1.0, adi);
            }
            Op::Mul(a, b) => {
                let (va, vb) = (v[a as usize], v[b as usize]);
                adj[a as usize] += ai * vb;
                adj[b as usize] += ai * va;
                // d(adj_a) = adi*vb + ai*dot_b ; d(adj_b) = adi*va + ai*dot_a
                if a == b {
                    axpy(ad!(a), vb + va, adi);
                    axpy(ad!(a), 2.0 * ai, dot_of!(a));
                } else {
                    axpy(ad!(a), vb, adi);
                    axpy(ad!(a), ai, dot_of!(b));
                    axpy(ad!(b), va, adi);
                    axpy(ad!(b), ai, dot_of!(a));
                }
            }
            Op::Div(a, b) => {
                let (va, vb) = (v[a as usize], v[b as usize]);
                let (fa, fb) = (1.0 / vb, -va / (vb * vb));
                let (fab, fbb) = (-1.0 / (vb * vb), 2.0 * va / (vb * vb * vb));
                adj[a as usize] += ai * fa;
                adj[b as usize] += ai * fb;
                if a == b {
                    // a/a: constant 1 (derivatives cancel)
                    axpy(ad!(a), fa + fb, adi);
                    axpy(ad!(a), ai * (2.0 * fab + fbb), dot_of!(a));
                } else {
                    axpy(ad!(a), fa, adi);
                    axpy(ad!(a), ai * fab, dot_of!(b));
                    axpy(ad!(b), fb, adi);
                    axpy(ad!(b), ai * fab, dot_of!(a));
                    axpy(ad!(b), ai * fbb, dot_of!(b));
                }
            }
            Op::Pow(a, b) => {
                let (_, fa, fb, faa, fab, fbb) = pow6(v[a as usize], v[b as usize]);
                adj[a as usize] += ai * fa;
                adj[b as usize] += ai * fb;
                if a == b {
                    axpy(ad!(a), fa + fb, adi);
                    axpy(ad!(a), ai * (faa + 2.0 * fab + fbb), dot_of!(a));
                } else {
                    axpy(ad!(a), fa, adi);
                    axpy(ad!(a), ai * faa, dot_of!(a));
                    axpy(ad!(a), ai * fab, dot_of!(b));
                    axpy(ad!(b), fb, adi);
                    axpy(ad!(b), ai * fab, dot_of!(a));
                    axpy(ad!(b), ai * fbb, dot_of!(b));
                }
            }
            Op::PowC(a, c) => {
                let (_, d1, d2) = powc3(v[a as usize], c);
                adj[a as usize] += ai * d1;
                axpy(ad!(a), d1, adi);
                axpy(ad!(a), ai * d2, dot_of!(a));
            }
            Op::Scale(a, c) => {
                adj[a as usize] += ai * c;
                axpy(ad!(a), c, adi);
            }
            Op::Neg(a) => {
                adj[a as usize] -= ai;
                axpy(ad!(a), -1.0, adi);
            }
            Op::F(f, a) => {
                let (_, d1, d2) = fun3(f, v[a as usize]);
                adj[a as usize] += ai * d1;
                axpy(ad!(a), d1, adi);
                axpy(ad!(a), ai * d2, dot_of!(a));
            }
            Op::Atan2(a, b) => {
                let (y, x) = (v[a as usize], v[b as usize]);
                let r = y * y + x * x;
                let (fa, fb) = (x / r, -y / r);
                let (faa, fab, fbb) = (-2.0 * x * y / (r * r), (y * y - x * x) / (r * r), 2.0 * x * y / (r * r));
                adj[a as usize] += ai * fa;
                adj[b as usize] += ai * fb;
                if a == b {
                    axpy(ad!(a), fa + fb, adi);
                    axpy(ad!(a), ai * (faa + 2.0 * fab + fbb), dot_of!(a));
                } else {
                    axpy(ad!(a), fa, adi);
                    axpy(ad!(a), ai * faa, dot_of!(a));
                    axpy(ad!(a), ai * fab, dot_of!(b));
                    axpy(ad!(b), fb, adi);
                    axpy(ad!(b), ai * fab, dot_of!(a));
                    axpy(ad!(b), ai * fbb, dot_of!(b));
                }
            }
            Op::Sum(s, l) => {
                for &a in &args[s as usize..(s + l) as usize] {
                    adj[a as usize] += ai;
                    axpy(ad!(a), 1.0, adi);
                }
            }
            Op::Min(s, l) | Op::Max(s, l) => {
                let ar = &args[s as usize..(s + l) as usize];
                let w = ar[pick(v, ar, matches!(ops[i], Op::Max(..)))];
                adj[w as usize] += ai;
                axpy(ad!(w), 1.0, adi);
            }
        }
    }
    val
}

impl Problem for ExprProblem {
    fn name(&self) -> String {
        if self.name.is_empty() {
            "unnamed".to_string()
        } else {
            self.name.clone()
        }
    }

    fn variables(&self) -> &[Variable] {
        &self.variables
    }

    fn num_constraints(&self) -> usize {
        self.m
    }

    fn constraint_bounds(&self, lower: &mut [f64], upper: &mut [f64]) {
        lower.copy_from_slice(&self.gl);
        upper.copy_from_slice(&self.gu);
    }

    fn sense(&self) -> Sense {
        self.sense
    }
}

impl SmoothOracle for ExprProblem {
    fn derivatives(&self) -> Access {
        Access::Hessians
    }

    fn function_classes(&self) -> Option<FunctionClasses> {
        Some(self.classes.clone())
    }

    fn objective(&self, x: &[f64]) -> Result<f64, EvalError> {
        self.obj_value(x)
    }

    fn gradient(&self, x: &[f64], grad: &mut [f64]) -> Result<(), EvalError> {
        let mut s = self.scratch.borrow_mut();
        let s = &mut *s;
        grad.fill(0.0);
        for &(j, c) in &self.obj_lin {
            grad[j as usize] += c;
        }
        for q in &self.quads[self.quad_ptr[0] as usize..self.quad_ptr[1] as usize] {
            grad[q.a as usize] += q.c * x[q.b as usize];
            grad[q.b as usize] += q.c * x[q.a as usize];
        }
        self.element_gradients(x, &self.obj_elements, s)?;
        for u in &self.uses[self.use_ptr[0] as usize..self.use_ptr[1] as usize] {
            let e = self.elements[u.el as usize];
            for t in 0..e.k as usize {
                let j = self.vars[e.var_start as usize + t] as usize;
                grad[j] += u.coef * s.el_grad[e.var_start as usize + t];
            }
        }
        if grad.iter().any(|g| !g.is_finite()) {
            return Err(EvalError::new("objective gradient is not finite"));
        }
        Ok(())
    }

    fn constraints(&self, x: &[f64], g: &mut [f64]) -> Result<(), EvalError> {
        let mut s = self.scratch.borrow_mut();
        let s = &mut *s;
        self.con_element_values(x, s)?;
        for i in 0..self.m {
            let mut v = self.con_const[i];
            for p in self.row_ptr[i] as usize..self.row_ptr[i + 1] as usize {
                v += self.jlin[p] * x[self.jcol[p] as usize];
            }
            for q in &self.quads[self.quad_ptr[i + 1] as usize..self.quad_ptr[i + 2] as usize] {
                v += q.c * x[q.a as usize] * x[q.b as usize];
            }
            for u in &self.uses[self.use_ptr[i + 1] as usize..self.use_ptr[i + 2] as usize] {
                v += u.coef * s.el_val[u.el as usize];
            }
            if !v.is_finite() {
                return Err(EvalError::new(format!("{} is not finite", func_name(i + 1))));
            }
            g[i] = v;
        }
        Ok(())
    }

    fn jacobian_structure(&self) -> Vec<(usize, usize)> {
        let mut v = Vec::with_capacity(self.jcol.len());
        for i in 0..self.m {
            let row = &self.jcol[self.row_ptr[i] as usize..self.row_ptr[i + 1] as usize];
            v.extend(row.iter().map(|&j| (i, j as usize)));
        }
        v
    }

    fn jacobian_values(&self, x: &[f64], vals: &mut [f64]) -> Result<(), EvalError> {
        let mut s = self.scratch.borrow_mut();
        let s = &mut *s;
        vals.copy_from_slice(&self.jlin);
        for q in &self.quads[self.quad_ptr[1] as usize..] {
            vals[q.ja as usize] += q.c * x[q.b as usize];
            vals[q.jb as usize] += q.c * x[q.a as usize];
        }
        self.element_gradients(x, &self.con_elements, s)?;
        for u in &self.uses[self.use_ptr[1] as usize..] {
            let e = self.elements[u.el as usize];
            for t in 0..e.k as usize {
                vals[self.jpos[u.jpos_start as usize + t] as usize] += u.coef * s.el_grad[e.var_start as usize + t];
            }
        }
        if vals.iter().any(|v| !v.is_finite()) {
            return Err(EvalError::new("Jacobian is not finite"));
        }
        Ok(())
    }

    fn hessian_structure(&self) -> Vec<(usize, usize)> {
        self.hess.clone()
    }

    fn hessian_values(&self, x: &[f64], obj_factor: f64, lambda: &[f64], vals: &mut [f64]) -> Result<(), EvalError> {
        let mut s = self.scratch.borrow_mut();
        let s = &mut *s;
        vals.fill(0.0);
        s.weight.fill(0.0);
        for func in 0..=self.m {
            let w = if func == 0 { obj_factor } else { lambda[func - 1] };
            if w == 0.0 {
                continue;
            }
            for q in &self.quads[self.quad_ptr[func] as usize..self.quad_ptr[func + 1] as usize] {
                // c x_a x_b: d2/dadb = c (a != b); c x_a^2: d2/da2 = 2c
                vals[q.hpos as usize] += if q.a == q.b { 2.0 * w * q.c } else { w * q.c };
            }
            for u in &self.uses[self.use_ptr[func] as usize..self.use_ptr[func + 1] as usize] {
                s.weight[u.el as usize] += w * u.coef;
            }
        }
        let mut xl = Vec::with_capacity(self.max_k);
        for (el, e) in self.elements.iter().enumerate() {
            let w = s.weight[el];
            if w == 0.0 {
                continue;
            }
            let k = e.k as usize;
            let xe = self.el_x(e, x, &mut xl);
            let v =
                hessian(self.el_ops(e), &self.args, xe, k, &mut s.v, &mut s.adj, &mut s.dot, &mut s.adjdot, &mut s.h);
            if !v.is_finite() {
                return Err(EvalError::new(format!("element {el} is not finite")));
            }
            let mut p = e.hpos_start as usize;
            for a in 0..k {
                for b in 0..=a {
                    // symmetric up to rounding; average the two triangles
                    let hab = if a == b { s.h[a * k + a] } else { 0.5 * (s.h[a * k + b] + s.h[b * k + a]) };
                    vals[self.hpos[p] as usize] += w * hab;
                    p += 1;
                }
            }
        }
        if vals.iter().any(|v| !v.is_finite()) {
            return Err(EvalError::new("Hessian is not finite"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Constraint;
    use rui_api::FnClass;

    /// Values, gradient, Jacobian and Hessian of the Lagrangian of `p` at `x`, densified.
    fn snapshot(p: &ExprProblem, x: &[f64]) -> Vec<f64> {
        let (n, m) = (x.len(), p.num_constraints());
        let mut out = vec![p.objective(x).unwrap()];
        let mut g = vec![0.0; n];
        p.gradient(x, &mut g).unwrap();
        out.extend(&g);
        let mut c = vec![0.0; m];
        p.constraints(x, &mut c).unwrap();
        out.extend(&c);
        let mut jac = vec![0.0; m * n];
        let mut jv = vec![0.0; p.jacobian_structure().len()];
        p.jacobian_values(x, &mut jv).unwrap();
        for (&(i, j), v) in p.jacobian_structure().iter().zip(&jv) {
            jac[i * n + j] += v;
        }
        out.extend(&jac);
        let mut hess = vec![0.0; n * n];
        let mut hv = vec![0.0; p.hessian_structure().len()];
        let lambda: Vec<f64> = (0..m).map(|i| 0.5 + i as f64).collect();
        p.hessian_values(x, 0.7, &lambda, &mut hv).unwrap();
        for (&(i, j), v) in p.hessian_structure().iter().zip(&hv) {
            hess[i * n + j] += v;
            if i != j {
                hess[j * n + i] += v;
            }
        }
        out.extend(&hess);
        out
    }

    fn two_vars(sense: Sense) -> Model {
        let mut m = Model::new("t");
        m.variables = vec![Variable::continuous("x", 0.1, 5.0), Variable::continuous("y", -2.0, 2.0)];
        m.sense = sense;
        m
    }

    /// `s = x*y + sin(x)` as a shared node (`shared = true`) or rebuilt at every use; objective
    /// `exp(s) + 3 s`, constraint `s * s + y`.
    fn with_subexpression(shared: bool) -> ExprProblem {
        let mut m = two_vars(Sense::Minimize);
        let s = |m: &mut Model| {
            let x = m.add(Node::Var(0));
            let y = m.add(Node::Var(1));
            let xy = m.add(Node::Bin(BinOp::Mul, x, y));
            let sx = m.add(Node::Fun(Fun::Sin, x));
            m.add(Node::Bin(BinOp::Add, xy, sx))
        };
        let s1 = s(&mut m);
        let s2 = if shared { s1 } else { s(&mut m) };
        let s3 = if shared { s1 } else { s(&mut m) };
        let s4 = if shared { s1 } else { s(&mut m) };
        let e = m.add(Node::Fun(Fun::Exp, s1));
        let three = m.add(Node::Num(3.0));
        let t = m.add(Node::Bin(BinOp::Mul, three, s2));
        m.objective.expr = Some(m.add(Node::Bin(BinOp::Add, e, t)));
        let sq = m.add(Node::Bin(BinOp::Mul, s3, s4));
        let y = m.add(Node::Var(1));
        let body = m.add(Node::Bin(BinOp::Add, sq, y));
        m.constraints.push(Constraint { lower: 0.0, upper: 4.0, body: Function { linear: vec![], expr: Some(body) } });
        ExprProblem::new(&m).unwrap()
    }

    #[test]
    fn shared_nodes_evaluate_like_repeated_subtrees() {
        let (dag, tree) = (with_subexpression(true), with_subexpression(false));
        for x in [[0.8, 0.3], [2.1, -1.4], [0.2, 1.9]] {
            let (a, b) = (snapshot(&dag, &x), snapshot(&tree, &x));
            for (k, (u, v)) in a.iter().zip(&b).enumerate() {
                assert!((u - v).abs() <= 1e-12 * 1f64.max(u.abs()), "entry {k} at {x:?}: {u} vs {v}");
            }
        }
        // `s * s`: 5 ops for `s` and one product when `s` is shared, 11 when it is repeated
        assert_eq!((dag.stats().largest_element_ops, tree.stats().largest_element_ops), (6, 11));
    }

    #[test]
    fn jacobian_rows_include_the_expression_variables() {
        // x*y + 2 y with only `y` in the linear part
        let mut m = two_vars(Sense::Minimize);
        let x = m.add(Node::Var(0));
        let y = m.add(Node::Var(1));
        let xy = m.add(Node::Bin(BinOp::Mul, x, y));
        m.constraints.push(Constraint {
            lower: f64::NEG_INFINITY,
            upper: 1.0,
            body: Function { linear: vec![(1, 2.0)], expr: Some(xy) },
        });
        let p = ExprProblem::new(&m).unwrap();
        assert_eq!(p.jacobian_structure(), vec![(0, 0), (0, 1)]);
        let mut v = [0.0; 2];
        p.jacobian_values(&[3.0, 0.5], &mut v).unwrap();
        assert_eq!(v, [0.5, 5.0]);
        assert_eq!(p.function_classes().unwrap().constraints, vec![FnClass::Quadratic]);
    }

    #[test]
    fn a_maximized_objective_is_evaluated_as_written() {
        let mut m = two_vars(Sense::Maximize);
        let x = m.add(Node::Var(0));
        let l = m.add(Node::Fun(Fun::Log, x));
        m.objective = Function { linear: vec![(1, -1.0)], expr: Some(l) };
        let p = ExprProblem::new(&m).unwrap();
        assert_eq!(p.sense(), Sense::Maximize);
        assert_eq!(p.objective(&[1.0, 2.0]), Ok(-2.0));
        let mut g = [0.0; 2];
        p.gradient(&[2.0, 2.0], &mut g).unwrap();
        assert_eq!(g, [0.5, -1.0]);
        let mut h = vec![0.0; p.hessian_structure().len()];
        p.hessian_values(&[2.0, 2.0], 1.0, &[], &mut h).unwrap();
        assert_eq!((p.hessian_structure(), h), (vec![(0, 0)], vec![-0.25]));
    }

    #[test]
    fn construction_errors_are_invalid_problems() {
        let mut m = two_vars(Sense::Minimize);
        let zero = m.add(Node::Num(0.0));
        let l = m.add(Node::Fun(Fun::Log, zero));
        m.objective.expr = Some(l);
        let err = ExprProblem::new(&m).err().unwrap();
        assert_eq!(err, Error::InvalidProblem("objective: a constant subexpression is not finite".into()));

        let mut m = two_vars(Sense::Minimize);
        m.add(Node::Var(7));
        assert!(matches!(ExprProblem::new(&m), Err(Error::InvalidProblem(_))));
    }
}
