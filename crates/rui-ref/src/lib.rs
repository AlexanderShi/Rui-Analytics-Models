//! # rui-ref
//!
//! Reference solvers for Rui Analytics Models. They let the public repository run and test the whole
//! system without the private solvers, and they are the templates for writing a solver plugin: each
//! one is a complete [`rui_api::Solver`] in a few hundred lines.
//!
//! | id | solver | problems |
//! |---|---|---|
//! | `random` | [`RandomSearch`]: uniform sampling, with ask/tell | bounded black boxes: continuous / integer / categorical variables, inequality constraints, failed evaluations |
//! | `pgd` | [`ProjectedGradient`]: projected gradient, Barzilai–Borwein steps, Armijo backtracking | smooth problems with bounds only |
//!
//! They are deliberately simple and are not meant for production use: random search has no model of
//! the function, and the projected gradient method converges slowly on ill-conditioned problems.

#![forbid(unsafe_code)]
// Index-based loops are the clearest notation for the numerical kernels, and `!(a > b)` comparisons
// are used on purpose because they are also true for NaN.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

pub mod pgd;
pub mod random;
mod rng;

pub use pgd::ProjectedGradient;
pub use random::RandomSearch;
