//! Errors of the API and of dispatch. Solver outcomes, including failures, are not errors: they are
//! [`Solution`](crate::Solution)s with a [`Termination`](crate::Termination).

use crate::signature::Verdict;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// The problem data are inconsistent (bounds, start values, constraint counts).
    InvalidProblem(String),
    /// Unknown option, wrong type, value out of range, option given twice.
    InvalidOption(String),
    /// `method=` names no registered solver.
    UnknownMethod { name: String, available: Vec<String> },
    /// The named solver cannot take this problem.
    Unsupported { method: String, problem: String, reasons: Vec<String> },
    /// No registered solver takes this problem automatically; `candidates` gives each solver's verdict.
    NoSolver { problem: String, candidates: Vec<(String, Verdict)> },
    /// A point told to or asked of a session does not belong to its space.
    InvalidPoint(String),
    /// A reported value cannot be used (NaN objective, wrong number of constraint values).
    InvalidValue(String),
    /// Every point of a finite search space has been evaluated.
    SpaceExhausted,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidProblem(m) => write!(f, "invalid problem: {m}"),
            Error::InvalidOption(m) => write!(f, "invalid option: {m}"),
            Error::UnknownMethod { name, available } if available.is_empty() => {
                write!(f, "unknown method `{name}`: no solver is registered")
            }
            Error::UnknownMethod { name, available } => {
                write!(f, "unknown method `{name}`; available: {}", available.join(", "))
            }
            Error::Unsupported { method, problem, reasons } => {
                write!(f, "`{method}` cannot solve this {problem} problem: {}", reasons.join("; "))
            }
            Error::NoSolver { candidates, .. } if candidates.is_empty() => f.write_str("no solver is registered"),
            Error::NoSolver { problem, candidates } => {
                write!(f, "no registered solver takes this {problem} problem automatically")?;
                for (id, verdict) in candidates {
                    match verdict {
                        Verdict::Supported => write!(f, "\n  {id}: supported")?,
                        Verdict::Lossy(why) => write!(f, "\n  {id}: usable only when named (method={id}): {why}")?,
                        Verdict::Unsupported(reasons) => write!(f, "\n  {id}: {}", reasons.join("; "))?,
                    }
                }
                Ok(())
            }
            Error::InvalidPoint(m) => write!(f, "invalid point: {m}"),
            Error::InvalidValue(m) => write!(f, "invalid value: {m}"),
            Error::SpaceExhausted => f.write_str("every point of the finite search space has been evaluated"),
        }
    }
}

impl std::error::Error for Error {}
