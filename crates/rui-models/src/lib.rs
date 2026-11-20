//! # rui-models
//!
//! The dispatch layer of Rui Analytics Models (`docs/design.md` §4.4): a [`Registry`] of solvers,
//! automatic choice from the problem's [`Signature`], bridges between problem forms, option
//! resolution, and ask/tell [`Session`]s.
//!
//! ```text
//! solve(problem, options):
//!     validate → signature → each solver's verdict → choose (named, or first in PRIORITY that supports
//!     the problem without discarding information) → resolve options → bridge (maximize → minimize;
//!     smooth → black box when a black-box solver is named) → solve → map the result back
//! ```
//!
//! Automatic dispatch never discards information: a problem with derivatives goes to a black-box solver
//! only when that solver is named (`method=...`). Every solve records one line saying what was chosen
//! and why ([`Solution::notes`]); [`Registry::explain`] gives the full decision without solving.

#![forbid(unsafe_code)]
// `!(a > b)` comparisons are used on purpose because they are also true for NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod bridge;
mod dispatch;
mod resolve;
mod session;
mod validate;

pub use dispatch::{Chosen, Explanation};
pub use session::Session;

use rui_api::{common, Error, Observer, Options, Problem, ProblemRef, Sense, Signature, Solution, Solver};
use std::collections::BTreeSet;

/// Order in which automatic dispatch considers solvers: the first registered solver in this order that
/// supports the problem without discarding information is chosen. Solvers with other ids rank after
/// these, in registration order.
pub const PRIORITY: &[&str] = &["nlp", "bo", "pgd", "random"];

fn rank(id: &str) -> usize {
    PRIORITY.iter().position(|p| *p == id).unwrap_or(PRIORITY.len())
}

/// The solvers a program can dispatch to.
#[derive(Default)]
pub struct Registry {
    /// Sorted by [`PRIORITY`] (stable: unknown ids keep registration order).
    solvers: Vec<Box<dyn Solver>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder form of [`add`](Self::add).
    pub fn with<S: Solver + 'static>(mut self, solver: S) -> Self {
        self.add(Box::new(solver));
        self
    }

    /// Registers a solver. Panics if its id is malformed or already registered, or if its option
    /// declarations are inconsistent: these are programming errors in the solver plugin.
    pub fn add(&mut self, solver: Box<dyn Solver>) {
        let info = solver.info();
        let id = info.id;
        assert!(
            !id.is_empty() && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-'),
            "solver id `{id}` must be lowercase letters, digits, `_` or `-`"
        );
        assert!(id != "auto", "solver id `auto` is reserved");
        assert!(self.get(id).is_none(), "solver `{id}` is registered twice");
        check_schema(id, &solver.options());
        self.solvers.push(solver);
        self.solvers.sort_by_key(|s| rank(s.info().id));
    }

    /// Registered solvers in priority order.
    pub fn solvers(&self) -> impl Iterator<Item = &dyn Solver> {
        self.solvers.iter().map(|s| s.as_ref())
    }

    pub fn get(&self, id: &str) -> Option<&dyn Solver> {
        self.solvers().find(|s| s.info().id == id)
    }

    /// The dispatch decision for `problem` (only `method` is read from `options`), without solving.
    pub fn explain(&self, problem: ProblemRef<'_>, options: &Options) -> Explanation {
        let signature = Signature::of(problem);
        let verdicts = dispatch::verdicts(&self.solvers, &signature, false);
        let choice = validate::problem(problem)
            .and_then(|_| dispatch::method_of(options))
            .and_then(|method| dispatch::choose(&signature, &self.solvers, &verdicts, method.as_deref()))
            .map(|c| {
                let info = self.solvers[c.index].info();
                Chosen { id: info.id, line: dispatch::describe(&signature, &info, &c) }
            });
        Explanation { signature, candidates: self.solvers.iter().map(|s| s.info()).zip(verdicts).collect(), choice }
    }

    /// Solves `problem` with the named or automatically chosen solver.
    ///
    /// `Err` means nothing was solved (invalid problem or options, no suitable solver); everything the
    /// solver itself reports, failures included, is an `Ok(Solution)`.
    pub fn solve(
        &self,
        problem: ProblemRef<'_>,
        options: &Options,
        observer: &mut dyn Observer,
    ) -> Result<Solution, Error> {
        validate::problem(problem)?;
        let signature = Signature::of(problem);
        let verdicts = dispatch::verdicts(&self.solvers, &signature, false);
        let method = dispatch::method_of(options)?;
        let choice = dispatch::choose(&signature, &self.solvers, &verdicts, method.as_deref())?;
        let solver = self.solvers[choice.index].as_ref();
        let resolved = resolve::resolve(options, solver, &self.solvers)?;
        let mut notes = vec![dispatch::describe(&signature, &solver.info(), &choice)];
        notes.extend(resolved.notices);

        let maximize = problem.sense() == Sense::Maximize;
        let mut flipped;
        let observer: &mut dyn Observer = if maximize {
            flipped = bridge::FlipObserver(observer);
            &mut flipped
        } else {
            observer
        };
        let verbosity = u8::try_from(resolved.options.i64(common::VERBOSITY)).unwrap_or(u8::MAX);
        let mut ctx = rui_api::Context::new(observer)
            .with_verbosity(verbosity)
            .with_time_limit(resolved.options.f64(common::TIME_LIMIT));
        for n in &notes {
            ctx.log(1, n);
        }
        let mut solution = bridge::run(solver, problem, &resolved.options, &mut ctx);
        if maximize {
            if let Some(f) = solution.objective.as_mut() {
                *f = -*f;
            }
        }
        notes.append(&mut solution.notes);
        solution.notes = notes;
        Ok(solution)
    }

    /// An ask/tell session over `space`: the caller evaluates the points (objective and the
    /// `space.num_constraints()` constraint values) and reports them with [`Session::tell`].
    pub fn session(&self, space: &dyn Problem, options: &Options) -> Result<Session, Error> {
        validate::space(space)?;
        let signature = Signature::of_space(space);
        let verdicts = dispatch::verdicts(&self.solvers, &signature, true);
        let method = dispatch::method_of(options)?;
        let choice = dispatch::choose(&signature, &self.solvers, &verdicts, method.as_deref())?;
        let solver = self.solvers[choice.index].as_ref();
        let resolved = resolve::resolve(options, solver, &self.solvers)?;
        let mut notes = vec![dispatch::describe(&signature, &solver.info(), &choice)];
        notes.extend(resolved.notices);
        let inner = if space.sense() == Sense::Maximize {
            solver.session(&bridge::Negated(space), &resolved.options)?
        } else {
            solver.session(space, &resolved.options)?
        };
        Ok(Session::new(inner, solver.info(), space, notes))
    }
}

/// Panics on inconsistent option declarations (see [`Registry::add`]).
fn check_schema(id: &str, specs: &[rui_api::OptionSpec]) {
    let universal = common::universal();
    let mut seen = BTreeSet::new();
    for spec in universal.iter().chain(specs) {
        for name in std::iter::once(spec.name).chain(spec.aliases.iter().copied()) {
            assert!(
                !name.is_empty() && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "solver `{id}`: option name `{name}` must be lowercase letters, digits or `_`"
            );
            assert!(seen.insert(name), "solver `{id}`: option name `{name}` is declared twice or is universal");
        }
        if let Some(default) = &spec.default {
            assert!(
                spec.kind.coerce(default).as_ref() == Ok(default),
                "solver `{id}`: default `{default}` of option `{}` is not {}",
                spec.name,
                spec.kind.describe()
            );
        }
    }
}
