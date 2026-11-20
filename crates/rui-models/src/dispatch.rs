//! Choosing a solver.

use rui_api::{common, Error, Options, Signature, Solver, SolverInfo, Value, Verdict};
use std::fmt;

pub(crate) struct Choice {
    pub index: usize,
    pub named: bool,
    pub lossy: bool,
}

/// Each solver's verdict on `sig`: its capabilities, plus its own extra checks, plus (for sessions)
/// whether it has an ask/tell interface.
pub(crate) fn verdicts(solvers: &[Box<dyn Solver>], sig: &Signature, session: bool) -> Vec<Verdict> {
    solvers
        .iter()
        .map(|s| {
            let info = s.info();
            let mut extra = s.check(sig);
            if session && !info.capabilities.ask_tell {
                extra.insert(0, "has no ask/tell interface".to_string());
            }
            match (info.capabilities.verdict(sig), extra.is_empty()) {
                (v, true) => v,
                (Verdict::Unsupported(mut reasons), false) => {
                    reasons.extend(extra);
                    Verdict::Unsupported(reasons)
                }
                (_, false) => Verdict::Unsupported(extra),
            }
        })
        .collect()
}

/// The `method` option: `None` for absent, empty or `auto`.
pub(crate) fn method_of(options: &Options) -> Result<Option<String>, Error> {
    match options.get(common::METHOD) {
        None => Ok(None),
        Some(Value::Text(s)) => {
            let s = s.trim().to_ascii_lowercase();
            Ok((!s.is_empty() && s != "auto").then_some(s))
        }
        Some(v) => Err(Error::InvalidOption(format!("option `method` must be a solver id, got `{v}`"))),
    }
}

pub(crate) fn choose(
    sig: &Signature,
    solvers: &[Box<dyn Solver>],
    verdicts: &[Verdict],
    method: Option<&str>,
) -> Result<Choice, Error> {
    if let Some(name) = method {
        let Some(index) = solvers.iter().position(|s| s.info().id == name) else {
            return Err(Error::UnknownMethod {
                name: name.to_string(),
                available: solvers.iter().map(|s| s.info().id.to_string()).collect(),
            });
        };
        return match &verdicts[index] {
            Verdict::Supported => Ok(Choice { index, named: true, lossy: false }),
            Verdict::Lossy(_) => Ok(Choice { index, named: true, lossy: true }),
            Verdict::Unsupported(reasons) => {
                Err(Error::Unsupported { method: name.to_string(), problem: sig.short(), reasons: reasons.clone() })
            }
        };
    }
    match verdicts.iter().position(|v| *v == Verdict::Supported) {
        Some(index) => Ok(Choice { index, named: false, lossy: false }),
        None => Err(Error::NoSolver {
            problem: sig.short(),
            candidates: solvers.iter().map(|s| s.info().id.to_string()).zip(verdicts.iter().cloned()).collect(),
        }),
    }
}

/// The dispatch line: `rui: NLP (n=4, m=2) -> nlp 0.1.3 (automatic; maximize -> minimize)`.
pub(crate) fn describe(sig: &Signature, info: &SolverInfo, choice: &Choice) -> String {
    let mut how = vec![if choice.named { "named" } else { "automatic" }];
    if sig.sense == rui_api::Sense::Maximize {
        how.push("maximize -> minimize");
    }
    if choice.lossy {
        how.push("derivatives ignored: solved as a black box");
    }
    format!("rui: {} -> {} {} ({})", sig.short(), info.id, info.version, how.join("; "))
}

/// The dispatch decision for one problem, as [`Registry::explain`](crate::Registry::explain) gives it.
#[derive(Debug)]
pub struct Explanation {
    pub signature: Signature,
    /// Every registered solver in priority order, with its verdict.
    pub candidates: Vec<(SolverInfo, Verdict)>,
    /// The chosen solver, or why none is chosen.
    pub choice: Result<Chosen, Error>,
}

/// The solver dispatch chose and the line it logs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chosen {
    pub id: &'static str,
    /// `rui: NLP (n=4, m=2) -> nlp 0.1.3 (automatic)`
    pub line: String,
}

impl Explanation {
    /// Id of the chosen solver.
    pub fn chosen(&self) -> Option<&'static str> {
        self.choice.as_ref().ok().map(|c| c.id)
    }
}

impl fmt::Display for Explanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "problem  {}", self.signature)?;
        writeln!(f, "solvers  in priority order")?;
        let width = self.candidates.iter().map(|(i, _)| i.id.len()).max().unwrap_or(0);
        for (info, verdict) in &self.candidates {
            writeln!(f, "  {:<width$}  {:<8}  {verdict}", info.id, info.version)?;
        }
        match &self.choice {
            Ok(c) => write!(f, "choice   {}", c.line),
            Err(e) => write!(f, "choice   none: {e}"),
        }
    }
}
