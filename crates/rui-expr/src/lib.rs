//! # rui-expr
//!
//! Algebraic models for Rui Analytics Models. A [`Model`] holds variables, an objective and constraints
//! `g_L <= g(x) <= g_U`; each function is a sparse linear part plus an optional nonlinear part, a node
//! of an expression graph shared by all functions of the model.
//!
//! * [`Model::classes`] classifies every function (linear, quadratic, other) from the graph, so dispatch
//!   can tell an LP or a QP from a general NLP.
//! * [`ExprProblem`] evaluates a model with exact first and second derivatives and implements
//!   [`rui_api::SmoothOracle`], so every solver behind `rui-models` can take it.
//!
//! Front ends build a `Model`: the `.nl` reader (`rui-nl`) now, the Python modeling API later.
//!
//! Design: `docs/design.md` §4.1.

#![forbid(unsafe_code)]
// Index-based loops are the clearest notation for the evaluation kernels.
#![allow(clippy::needless_range_loop)]

mod classify;
pub mod eval;
pub mod model;

pub use eval::{EvalStats, ExprProblem};
pub use model::{BinOp, Constraint, Fun, Function, Model, Node, NodeId};
