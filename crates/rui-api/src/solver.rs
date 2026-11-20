//! The solver contract.
//!
//! A solver plugin implements [`Solver`]: it describes itself ([`SolverInfo`], including its
//! [`Capabilities`](crate::Capabilities)), declares its options, and solves. Rui Analytics Models
//! guarantees what a solver receives:
//!
//! * a validated problem (consistent bounds, starts inside the bounds, integral integer bounds);
//! * a **minimization** problem (maximization is negated before, and the result mapped back after);
//! * a [`ProblemRef::BlackBox`] if the solver's capabilities say it uses [`Access::Values`](crate::Access),
//!   otherwise a [`ProblemRef::Smooth`], whose signature its capabilities support;
//! * options validated against its declarations, with defaults filled in.
//!
//! In return a solver must stop when [`Context::stop_reason`] says so (at least between iterations or
//! evaluations), log only through the [`Context`], and never claim more than it knows (see
//! [`PointKind`](crate::PointKind)).

use crate::error::Error;
use crate::options::{OptionSpec, ResolvedOptions};
use crate::problem::{Outcome, Problem, ProblemRef};
use crate::signature::Signature;
use crate::solution::{Solution, SolverInfo, Termination};
use std::fmt;
use std::time::{Duration, Instant};

/// A progress event.
#[derive(Clone, Copy, Debug)]
pub enum Event<'a> {
    /// One iteration of a derivative-based solver.
    Iteration { iteration: usize, objective: f64, infeasibility: f64, optimality: f64 },
    /// One evaluation of a black-box solver. `best`: best feasible objective so far.
    Evaluation { index: usize, x: &'a [f64], outcome: &'a Outcome, best: Option<f64> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    Stop,
}

/// Where a solve's output goes, and who may stop it. Every method has a do-nothing default.
pub trait Observer {
    /// One line of log output.
    fn log(&mut self, line: &str) {
        let _ = line;
    }
    /// A progress event; returning [`Flow::Stop`] stops the solve ([`Termination::Interrupted`]).
    fn event(&mut self, event: &Event<'_>) -> Flow {
        let _ = event;
        Flow::Continue
    }
    /// Polled by solvers; `true` stops the solve (e.g. Ctrl-C was pressed).
    fn interrupted(&mut self) -> bool {
        false
    }
}

/// Discards everything.
pub struct Silent;
impl Observer for Silent {}

/// Prints log lines to standard output.
pub struct Printer;
impl Observer for Printer {
    fn log(&mut self, line: &str) {
        println!("{line}");
    }
}

/// What a solve runs in: log output, verbosity, deadline, interruption.
pub struct Context<'a> {
    observer: &'a mut dyn Observer,
    verbosity: u8,
    start: Instant,
    deadline: Option<Instant>,
    stop_requested: bool,
}

impl<'a> Context<'a> {
    /// Verbosity 1, no deadline; the clock starts now.
    pub fn new(observer: &'a mut dyn Observer) -> Self {
        Context { observer, verbosity: 1, start: Instant::now(), deadline: None, stop_requested: false }
    }

    pub fn with_verbosity(mut self, verbosity: u8) -> Self {
        self.verbosity = verbosity;
        self
    }

    /// Seconds from now; infinite or negative values mean no limit.
    pub fn with_time_limit(mut self, seconds: f64) -> Self {
        self.deadline = if seconds.is_finite() && seconds >= 0.0 {
            Duration::try_from_secs_f64(seconds).ok().and_then(|d| self.start.checked_add(d))
        } else {
            None
        };
        self
    }

    pub fn verbosity(&self) -> u8 {
        self.verbosity
    }

    /// Whether lines of `level` (1 summary, 2 per iteration / evaluation, 3 details) are shown.
    pub fn logs(&self, level: u8) -> bool {
        level <= self.verbosity
    }

    pub fn log(&mut self, level: u8, line: impl fmt::Display) {
        if self.logs(level) {
            self.observer.log(&line.to_string());
        }
    }

    /// Seconds since the context was created.
    pub fn elapsed(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Whether the solve must stop now, and as what.
    pub fn stop_reason(&mut self) -> Option<Termination> {
        if self.stop_requested || self.observer.interrupted() {
            self.stop_requested = true;
            Some(Termination::Interrupted)
        } else if self.deadline.is_some_and(|d| Instant::now() >= d) {
            Some(Termination::TimeLimit)
        } else {
            None
        }
    }

    /// Reports an event; an observer's [`Flow::Stop`] makes the next [`stop_reason`](Self::stop_reason)
    /// return [`Termination::Interrupted`].
    pub fn emit(&mut self, event: &Event<'_>) {
        if self.observer.event(event) == Flow::Stop {
            self.stop_requested = true;
        }
    }
}

/// A solver plugin.
pub trait Solver: Send + Sync {
    fn info(&self) -> SolverInfo;

    /// Declared options, including the [`common`](crate::common) ones the solver honors. Names must be
    /// unique and must not repeat the universal options (`method`, `time_limit`, `verbosity`).
    fn options(&self) -> Vec<OptionSpec>;

    /// Reasons the solver cannot take a problem beyond what its capabilities express (e.g. a size
    /// limit). Empty: no further restriction.
    fn check(&self, signature: &Signature) -> Vec<String> {
        let _ = signature;
        Vec::new()
    }

    fn solve(&self, problem: ProblemRef<'_>, options: &ResolvedOptions, ctx: &mut Context<'_>) -> Solution;

    /// An ask/tell session over `space` (a minimization problem whose functions the caller evaluates).
    /// Solvers whose capabilities set `ask_tell` override this.
    fn session(&self, space: &dyn Problem, options: &ResolvedOptions) -> Result<Box<dyn AskTell>, Error> {
        let _ = (space, options);
        Err(Error::Unsupported {
            method: self.info().id.to_string(),
            problem: "ask/tell session".to_string(),
            reasons: vec!["has no ask/tell interface".to_string()],
        })
    }
}

/// An optimization driven from outside: the caller asks for points, evaluates them however it likes
/// (in parallel, on a cluster, by hand) and tells the results, in any order.
pub trait AskTell: Send {
    /// `q` new points to evaluate; they count as pending until told.
    fn ask(&mut self, q: usize) -> Result<Vec<Vec<f64>>, Error>;
    /// The result of evaluating `x`, which need not have been asked for (existing data).
    fn tell(&mut self, x: &[f64], outcome: Outcome) -> Result<(), Error>;
    /// `x` is being evaluated elsewhere (another worker asked for it); do not propose it again.
    fn mark_pending(&mut self, x: &[f64]) -> Result<(), Error>;
    /// Best point so far ([`Termination::AskTell`]).
    fn solution(&mut self) -> Solution;
    fn n_evaluations(&self) -> usize;
}
