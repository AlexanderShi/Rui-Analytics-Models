//! # rui-nl
//!
//! Reader for ASCII (`g`) `.nl` files, following D. M. Gay, "Writing .nl Files" (Sandia, 2005). The
//! result is a [`rui_expr::Model`]; [`rui_expr::ExprProblem`] evaluates it with exact derivatives and
//! `rui-models` dispatches it like any other smooth problem.
//!
//! ```no_run
//! let model = rui_nl::read("hs071.nl")?;
//! let problem = rui_expr::ExprProblem::new(&model)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The reader accepts a subset of the format and rejects everything else with an error rather than
//! misreading it.
//!
//! * Accepted: at most one objective; the segments `C`, `O`, `x`, `r`, `b`, `k`, `J`, `G`; the operators
//!   `+ - * / ^`, unary minus, `sumlist`, `min` / `max` lists, `atan2` and the one-argument functions of
//!   [`rui_expr::Fun`].
//! * Rejected for now: defined variables (`V` segments), suffixes (`S`), dual start values (`d`),
//!   discrete variables, several objectives, logical and complementarity constraints, imported
//!   functions (`F`), network constraints, the binary (`b`) format. Reading what Pyomo's and JuMP's
//!   writers produce, and writing `.sol` files, is milestone M1 (`docs/design.md` §5.1).
//!
//! Variables and constraints have no names: `.nl` files carry none (AMPL writes them to separate `.col`
//! and `.row` files). A start value from the `x` segment is kept as written, even outside the variable's
//! bounds; variables without one have no start value.

#![forbid(unsafe_code)]

use rui_api::{Sense, Variable};
use rui_expr::{BinOp, Constraint, Fun, Function, Model, Node, NodeId};
use std::fmt;
use std::path::{Path, PathBuf};

/// Why a `.nl` file could not be read.
#[derive(Debug)]
pub enum NlError {
    /// The file could not be read from disk.
    Io { path: PathBuf, error: std::io::Error },
    /// The text is not a `.nl` file this reader accepts.
    Syntax { line: usize, message: String },
}

impl fmt::Display for NlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NlError::Io { path, error } => write!(f, "{}: {error}", path.display()),
            NlError::Syntax { line, message } => write!(f, ".nl line {line}: {message}"),
        }
    }
}

impl std::error::Error for NlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NlError::Io { error, .. } => Some(error),
            NlError::Syntax { .. } => None,
        }
    }
}

/// Reads and parses the `.nl` file at `path`.
pub fn read(path: impl AsRef<Path>) -> Result<Model, NlError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|error| NlError::Io { path: path.to_path_buf(), error })?;
    parse(&text)
}

fn fun_from_opcode(op: u32) -> Option<Fun> {
    Some(match op {
        13 => Fun::Floor,
        14 => Fun::Ceil,
        15 => Fun::Abs,
        37 => Fun::Tanh,
        38 => Fun::Tan,
        39 => Fun::Sqrt,
        40 => Fun::Sinh,
        41 => Fun::Sin,
        42 => Fun::Log10,
        43 => Fun::Log,
        44 => Fun::Exp,
        45 => Fun::Cosh,
        46 => Fun::Cos,
        47 => Fun::Atanh,
        49 => Fun::Atan,
        50 => Fun::Asinh,
        51 => Fun::Asin,
        52 => Fun::Acosh,
        53 => Fun::Acos,
        _ => return None,
    })
}

struct Lines<'a> {
    it: std::str::Lines<'a>,
    line: usize,
}

impl<'a> Lines<'a> {
    fn next(&mut self) -> Result<&'a str, NlError> {
        self.line += 1;
        self.it.next().ok_or(NlError::Syntax { line: self.line, message: "unexpected end of file".into() })
    }

    fn error(&self, message: impl Into<String>) -> NlError {
        NlError::Syntax { line: self.line, message: message.into() }
    }

    fn err<T>(&self, message: impl Into<String>) -> Result<T, NlError> {
        Err(self.error(message))
    }
}

fn ints(lx: &Lines, s: &str, k: usize) -> Result<Vec<i64>, NlError> {
    let body = s.split('#').next().unwrap_or("");
    let v: Result<Vec<i64>, _> = body.split_whitespace().take(k).map(|t| t.parse::<i64>()).collect();
    match v {
        Ok(v) if v.len() == k => Ok(v),
        _ => lx.err(format!("expected {k} integers in {s:?}")),
    }
}

fn num(lx: &Lines, t: &str) -> Result<f64, NlError> {
    match t.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => lx.err(format!("bad number {t:?}")),
    }
}

/// Bound line of an `r` or `b` segment: `(lower, upper)`.
fn bound(lx: &Lines, s: &str) -> Result<(f64, f64), NlError> {
    let t: Vec<&str> = s.split_whitespace().collect();
    let inf = f64::INFINITY;
    let want = |k: usize| if t.len() == k { Ok(()) } else { lx.err(format!("bad bound line {s:?}")) };
    match t.first().copied() {
        Some("0") => {
            want(3)?;
            Ok((num(lx, t[1])?, num(lx, t[2])?))
        }
        Some("1") => {
            want(2)?;
            Ok((-inf, num(lx, t[1])?))
        }
        Some("2") => {
            want(2)?;
            Ok((num(lx, t[1])?, inf))
        }
        Some("3") => {
            want(1)?;
            Ok((-inf, inf))
        }
        Some("4") => {
            want(2)?;
            let v = num(lx, t[1])?;
            Ok((v, v))
        }
        _ => lx.err(format!("unsupported bound line {s:?}")),
    }
}

/// Parses the text of an ASCII `.nl` file.
pub fn parse(text: &str) -> Result<Model, NlError> {
    let mut lx = Lines { it: text.lines(), line: 0 };
    let h1 = lx.next()?;
    if !h1.starts_with('g') {
        return lx.err("only ASCII ('g') .nl files are supported");
    }
    let name = h1.split("# problem").nth(1).map(|s| s.trim().to_string()).unwrap_or_default();
    let s2 = lx.next()?;
    let h2 = ints(&lx, s2, 3)?;
    let (n, m, n_obj) = (h2[0] as usize, h2[1] as usize, h2[2] as usize);
    if n_obj > 1 {
        return lx.err("more than one objective");
    }
    let s3 = lx.next()?;
    ints(&lx, s3, 2)?; // nonlinear constraints, objectives
    let s4 = lx.next()?;
    if ints(&lx, s4, 2)? != [0, 0] {
        return lx.err("network constraints are not supported");
    }
    lx.next()?; // nonlinear variables in constraints, objectives, both (any order is accepted)
    let s6 = lx.next()?;
    if ints(&lx, s6, 2)? != [0, 0] {
        return lx.err("linear network variables or imported functions are not supported");
    }
    let s7 = lx.next()?;
    if ints(&lx, s7, 5)?.iter().any(|&v| v != 0) {
        return lx.err("discrete variables are not supported");
    }
    lx.next()?; // nonzeros in Jacobian, gradients
    lx.next()?; // max name lengths
    let s10 = lx.next()?;
    if ints(&lx, s10, 5)?.iter().any(|&v| v != 0) {
        return lx.err("common expressions (defined variables) are not supported");
    }
    let inf = f64::INFINITY;
    let mut model = Model::new(name);
    let mut x0: Vec<Option<f64>> = vec![None; n];
    let (mut xl, mut xu) = (vec![-inf; n], vec![inf; n]);
    let (mut gl, mut gu) = (vec![-inf; m], vec![inf; m]);
    let mut con_expr = vec![NodeId::MAX; m];
    let mut obj_expr: Option<NodeId> = None;
    let mut jac: Vec<Vec<(u32, f64)>> = vec![Vec::new(); m];
    let mut grad: Vec<(u32, f64)> = Vec::new();
    let mut seen_b = false;
    let mut seen_r = m == 0;
    loop {
        lx.line += 1;
        let Some(s) = lx.it.next() else { break };
        if s.is_empty() {
            continue;
        }
        let (tag, rest) = s.split_at(1);
        let head: Vec<&str> = rest.split_whitespace().collect();
        let idx = |k: usize| -> Result<usize, NlError> {
            head.get(k).and_then(|t| t.parse::<usize>().ok()).ok_or(lx.error(format!("bad segment header {s:?}")))
        };
        match tag {
            "C" => {
                let i = idx(0)?;
                if i >= m || con_expr[i] != NodeId::MAX {
                    return lx.err(format!("bad or repeated constraint {i}"));
                }
                con_expr[i] = tree(&mut model, n, &mut lx)?;
            }
            "O" => {
                if idx(0)? != 0 || n_obj != 1 || obj_expr.is_some() {
                    return lx.err("bad objective segment");
                }
                model.sense = match idx(1)? {
                    0 => Sense::Minimize,
                    1 => Sense::Maximize,
                    _ => return lx.err("bad objective sense"),
                };
                obj_expr = Some(tree(&mut model, n, &mut lx)?);
            }
            "x" => {
                for _ in 0..idx(0)? {
                    let t = lx.next()?;
                    let mut it = t.split_whitespace();
                    let (Some(j), Some(v)) = (it.next(), it.next()) else {
                        return lx.err("bad x line");
                    };
                    let j: usize = j.parse().map_err(|_| lx.error("bad x index"))?;
                    if j >= n {
                        return lx.err("x index out of range");
                    }
                    x0[j] = Some(num(&lx, v)?);
                }
            }
            "r" => {
                for i in 0..m {
                    let t = lx.next()?;
                    (gl[i], gu[i]) = bound(&lx, t)?;
                }
                seen_r = true;
            }
            "b" => {
                for j in 0..n {
                    let t = lx.next()?;
                    (xl[j], xu[j]) = bound(&lx, t)?;
                }
                seen_b = true;
            }
            "k" => {
                for _ in 0..idx(0)? {
                    lx.next()?; // cumulative column counts: implied by the J segments
                }
            }
            "J" => {
                let (i, cnt) = (idx(0)?, idx(1)?);
                if i >= m || !jac[i].is_empty() {
                    return lx.err(format!("bad or repeated J segment {i}"));
                }
                jac[i] = coef_lines(&mut lx, cnt, n)?;
            }
            "G" => {
                if idx(0)? != 0 || n_obj != 1 || !grad.is_empty() {
                    return lx.err("bad G segment");
                }
                let cnt = idx(1)?;
                grad = coef_lines(&mut lx, cnt, n)?;
            }
            _ => return lx.err(format!("unsupported segment {s:?}")),
        }
    }
    if !seen_b || !seen_r {
        return lx.err("missing b or r segment");
    }
    if con_expr.contains(&NodeId::MAX) {
        return lx.err("missing C segment");
    }
    if n_obj == 1 && obj_expr.is_none() {
        return lx.err("missing O segment");
    }
    model.variables =
        (0..n).map(|j| Variable { start: x0[j], ..Variable::continuous(String::new(), xl[j], xu[j]) }).collect();
    model.objective = Function { linear: grad, expr: obj_expr };
    model.constraints = (0..m)
        .map(|i| Constraint {
            lower: gl[i],
            upper: gu[i],
            body: Function { linear: std::mem::take(&mut jac[i]), expr: Some(con_expr[i]) },
        })
        .collect();
    Ok(model)
}

/// Reads one expression in prefix order, iteratively (sums can be long and products deep). Operands
/// are added to the graph before the node that uses them.
fn tree(model: &mut Model, n: usize, lx: &mut Lines) -> Result<NodeId, NlError> {
    // frames: (opcode, operands collected so far, operands expected)
    let mut stack: Vec<(u32, Vec<NodeId>, usize)> = Vec::new();
    loop {
        let t = lx.next()?;
        let done: NodeId = match t.as_bytes().first() {
            Some(b'n') => model.add(Node::Num(num(lx, &t[1..])?)),
            Some(b'v') => {
                let j: usize = t[1..].trim().parse().map_err(|_| lx.error(format!("bad variable {t:?}")))?;
                if j >= n {
                    return lx.err(format!("variable {j} out of range"));
                }
                model.add(Node::Var(j as u32))
            }
            Some(b'o') => {
                let op: u32 = t[1..]
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or(lx.error(format!("bad operator {t:?}")))?;
                let arity = match op {
                    0 | 1 | 2 | 3 | 5 | 48 => 2,
                    16 => 1,
                    54 | 11 | 12 => {
                        let k: usize = lx.next()?.trim().parse().map_err(|_| lx.error("bad operand count"))?;
                        if k == 0 {
                            return lx.err("empty operand list");
                        }
                        k
                    }
                    _ if fun_from_opcode(op).is_some() => 1,
                    _ => return lx.err(format!("unsupported operator o{op}")),
                };
                stack.push((op, Vec::with_capacity(arity.min(1 << 16)), arity));
                continue;
            }
            _ => return lx.err(format!("expected an expression, found {t:?}")),
        };
        let mut node = done;
        loop {
            let Some(top) = stack.last_mut() else {
                return Ok(node);
            };
            top.1.push(node);
            if top.1.len() < top.2 {
                break;
            }
            let (op, kids, _) = stack.pop().unwrap();
            node = build(model, op, &kids);
        }
    }
}

fn build(model: &mut Model, op: u32, kids: &[NodeId]) -> NodeId {
    let bin = |o| Node::Bin(o, kids[0], kids[1]);
    let node = match op {
        0 => bin(BinOp::Add),
        1 => bin(BinOp::Sub),
        2 => bin(BinOp::Mul),
        3 => bin(BinOp::Div),
        5 => bin(BinOp::Pow),
        48 => bin(BinOp::Atan2),
        16 => Node::Neg(kids[0]),
        54 | 11 | 12 => {
            let (start, len) = model.add_list(kids);
            match op {
                54 => Node::Sum(start, len),
                11 => Node::Min(start, len),
                _ => Node::Max(start, len),
            }
        }
        _ => Node::Fun(fun_from_opcode(op).expect("checked in tree()"), kids[0]),
    };
    model.add(node)
}

fn coef_lines(lx: &mut Lines, cnt: usize, n: usize) -> Result<Vec<(u32, f64)>, NlError> {
    let mut v = Vec::with_capacity(cnt);
    for _ in 0..cnt {
        let t = lx.next()?;
        let mut it = t.split_whitespace();
        let (Some(j), Some(c)) = (it.next(), it.next()) else {
            return lx.err("bad coefficient line");
        };
        let j: usize = j.parse().map_err(|_| lx.error("bad column index"))?;
        if j >= n {
            return lx.err("column index out of range");
        }
        v.push((j as u32, num(lx, c)?));
    }
    v.sort_by_key(|&(j, _)| j);
    if v.windows(2).any(|w| w[0].0 == w[1].0) {
        return lx.err("repeated column in a J or G segment");
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rui_api::FnClass;

    const SMALL: &str = "g3 1 1 0\t# problem small
 2 1 1 0 1\t# vars, constraints, objectives, ranges, eqns
 1 1\t# nonlinear constraints, objectives
 0 0\t# network constraints: nonlinear, linear
 2 2 2\t# nonlinear vars in constraints, objectives, both
 0 0 0 1\t# linear network variables; functions; arith, flags
 0 0 0 0 0\t# discrete variables: binary, integer, nonlinear (b,c,o)
 2 2\t# nonzeros in Jacobian, gradients
 0 0\t# max name lengths: constraints, variables
 0 0 0 0 0\t# common exprs: b,c,o,c1,o1
C0
o2
v0
v1
O0 1
o54
3
o5
v0
n2
o44
v1
n1.5
x1
1 0.5
r
4 2
b
2 -1
3
k1
1
J0 2
0 1
1 0
G0 2
0 0
1 3
";

    #[test]
    fn parses_small_file() {
        let f = parse(SMALL).unwrap();
        assert_eq!(
            (f.variables.len(), f.constraints.len(), f.name.as_str(), f.sense),
            (2, 1, "small", Sense::Maximize)
        );
        let v = &f.variables;
        assert_eq!((v[0].start, v[1].start), (None, Some(0.5)));
        assert_eq!(
            (v[0].lower, v[0].upper, v[1].lower, v[1].upper),
            (-1.0, f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY)
        );
        let c = &f.constraints[0];
        assert_eq!((c.lower, c.upper), (2.0, 2.0));
        assert_eq!(c.body.linear, vec![(0, 1.0), (1, 0.0)]);
        assert_eq!(f.objective.linear, vec![(0, 0.0), (1, 3.0)]);
        assert!(matches!(f.nodes[c.body.expr.unwrap() as usize], Node::Bin(BinOp::Mul, _, _)));
        assert!(matches!(f.nodes[f.objective.expr.unwrap() as usize], Node::Sum(_, 3)));
        assert_eq!(f.check(), Ok(()));
        // x0 * x1 and x0^2 + exp(x1) + 1.5 + 3 x1
        let classes = f.classes();
        assert_eq!((classes.objective, classes.constraints), (FnClass::Smooth, vec![FnClass::Quadratic]));
    }

    #[test]
    fn rejects_unsupported_input() {
        let bad = SMALL.replace("o44\nv1", "o74\nv1");
        assert_eq!(parse(&bad).unwrap_err().to_string(), ".nl line 21: unsupported operator o74");
        let discrete = SMALL.replace(" 0 0 0 0 0\t# discrete", " 0 1 0 0 0\t# discrete");
        assert_eq!(parse(&discrete).unwrap_err().to_string(), ".nl line 7: discrete variables are not supported");
        assert!(parse("b3 1 1 0\n").unwrap_err().to_string().contains("only ASCII"));
    }

    #[test]
    fn missing_files_are_io_errors() {
        let err = read("no/such/file.nl").unwrap_err();
        assert!(matches!(err, NlError::Io { .. }), "{err}");
        assert!(err.to_string().starts_with("no/such/file.nl: "), "{err}");
    }
}
