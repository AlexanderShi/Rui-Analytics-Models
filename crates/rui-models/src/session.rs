//! Ask/tell sessions.

use rui_api::{AskTell, Error, Outcome, Problem, Sense, Solution, SolverInfo};

/// An optimization driven by the caller: [`ask`](Self::ask) for points, evaluate them (in any order, in
/// parallel, elsewhere), [`tell`](Self::tell) the results. Created by
/// [`Registry::session`](crate::Registry::session).
///
/// Objective values are those of the problem as stated: for a maximization problem tell the values
/// to maximize; the session negates them for the solver and maps the solution back.
pub struct Session {
    inner: Box<dyn AskTell>,
    solver: SolverInfo,
    sense: Sense,
    n: usize,
    m: usize,
    notes: Vec<String>,
}

impl Session {
    pub(crate) fn new(inner: Box<dyn AskTell>, solver: SolverInfo, space: &dyn Problem, notes: Vec<String>) -> Self {
        Session { inner, solver, sense: space.sense(), n: space.variables().len(), m: space.num_constraints(), notes }
    }

    pub fn solver(&self) -> SolverInfo {
        self.solver
    }

    /// Dispatch decision and option notices.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// `q` points to evaluate.
    pub fn ask(&mut self, q: usize) -> Result<Vec<Vec<f64>>, Error> {
        if q == 0 {
            return Ok(Vec::new());
        }
        self.inner.ask(q)
    }

    pub fn ask_one(&mut self) -> Result<Vec<f64>, Error> {
        self.inner.ask(1)?.pop().ok_or_else(|| Error::InvalidValue("the solver returned no point".to_string()))
    }

    /// The result of evaluating `x` (asked or not). A failed evaluation is [`Outcome::Failed`]; a value
    /// must be finite and give one entry per constraint.
    pub fn tell(&mut self, x: &[f64], outcome: Outcome) -> Result<(), Error> {
        self.check_len(x)?;
        let outcome = match outcome {
            Outcome::Failed => Outcome::Failed,
            Outcome::Value { objective, constraints } => {
                if !objective.is_finite() {
                    return Err(Error::InvalidValue(format!(
                        "objective {objective:?} is not finite; report a failed evaluation as Outcome::Failed"
                    )));
                }
                if constraints.len() != self.m {
                    return Err(Error::InvalidValue(format!(
                        "{} constraint values for a problem with {} constraints",
                        constraints.len(),
                        self.m
                    )));
                }
                if let Some(c) = constraints.iter().find(|c| !c.is_finite()) {
                    return Err(Error::InvalidValue(format!("constraint value {c:?} is not finite")));
                }
                Outcome::Value { objective: self.sense.sign() * objective, constraints }
            }
        };
        self.inner.tell(x, outcome)
    }

    /// `x` is being evaluated elsewhere: it will not be proposed again.
    pub fn mark_pending(&mut self, x: &[f64]) -> Result<(), Error> {
        self.check_len(x)?;
        self.inner.mark_pending(x)
    }

    pub fn n_evaluations(&self) -> usize {
        self.inner.n_evaluations()
    }

    /// Best point so far, with the objective of the problem as stated.
    pub fn solution(&mut self) -> Solution {
        let mut s = self.inner.solution();
        if let Some(f) = s.objective.as_mut() {
            *f *= self.sense.sign();
        }
        let mut notes = self.notes.clone();
        notes.append(&mut s.notes);
        s.notes = notes;
        s
    }

    fn check_len(&self, x: &[f64]) -> Result<(), Error> {
        if x.len() != self.n {
            return Err(Error::InvalidPoint(format!("{} values for a space of {} variables", x.len(), self.n)));
        }
        Ok(())
    }
}
