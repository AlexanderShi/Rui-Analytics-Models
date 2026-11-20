//! # rui-api
//!
//! The contract between Rui Analytics Models and the solvers it dispatches to. A solver crate depends
//! on this crate only; the dispatch layer (`rui-models`) and every front end (command line, Python,
//! `.nl` files) depend on it too. No third-party dependency.
//!
//! | module | content |
//! |---|---|
//! | [`problem`] | variables, constraint bounds, the two oracles ([`SmoothOracle`], [`BlackBoxOracle`]) |
//! | [`signature`] | what a problem is ([`Signature`]) and what a solver handles ([`Capabilities`]) |
//! | [`options`] | user options, solver option declarations, resolved values |
//! | [`solution`] | [`Solution`]: why the solver stopped ([`Termination`]) and what the point is ([`PointKind`]) |
//! | [`solver`] | the [`Solver`] and [`AskTell`] traits, the [`Context`] a solve runs in |
//! | [`check`] | independent check of a first-order (KKT) point |
//!
//! Design: `docs/design.md` §4.

#![forbid(unsafe_code)]

pub mod check;
pub mod error;
pub mod options;
pub mod problem;
pub mod signature;
pub mod solution;
pub mod solver;

pub use check::{kkt_check, KktReport};
pub use error::Error;
pub use options::{common, Kind, OptionSpec, Options, ResolvedOptions, Value};
pub use problem::{
    constraint_bound_vectors, constraint_violation, default_start, Access, BlackBoxOracle, EvalError, FnClass,
    FunctionClasses, Outcome, Problem, ProblemRef, Sense, SmoothOracle, VarKind, Variable,
};
pub use signature::{Capabilities, ProblemClass, Signature, Verdict};
pub use solution::{Duals, PointKind, Solution, SolverInfo, Stats, Termination};
pub use solver::{AskTell, Context, Event, Flow, Observer, Printer, Silent, Solver};

/// Version of this crate (the workspace version, see `CHANGELOG.md`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
