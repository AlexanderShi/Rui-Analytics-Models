//! What a problem is ([`Signature`]) and what a solver handles ([`Capabilities`]).
//!
//! Dispatch only looks at signatures: a signature is plain data that can be printed and tested.
//! A solver declares its [`Capabilities`] as data too, so the verdict "can this solver take this
//! problem, and if not why" ([`Capabilities::verdict`]) is computed the same way for every solver.

use crate::problem::{constraint_bound_vectors, Access, FnClass, Problem, ProblemRef, Sense, VarKind};
use std::fmt;

/// Structural summary of a problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub n_variables: usize,
    pub n_continuous: usize,
    pub n_integer: usize,
    pub n_categorical: usize,
    /// Variables with at least one infinite bound.
    pub n_unbounded: usize,
    pub n_constraints: usize,
    /// Constraints with `g_L == g_U` (finite).
    pub n_equalities: usize,
    pub objective: FnClass,
    /// Number of constraints of each class, indexed by `FnClass as usize`
    /// (linear, quadratic, smooth, black-box).
    pub constraint_classes: [usize; 4],
    /// Derivative information available.
    pub access: Access,
    pub sense: Sense,
    pub noisy: bool,
}

impl Signature {
    pub fn of(problem: ProblemRef<'_>) -> Signature {
        match problem {
            ProblemRef::Smooth(p) => {
                let m = p.num_constraints();
                let (objective, constraints) = match p.function_classes() {
                    Some(c) => (c.objective, c.constraints),
                    None => (FnClass::Smooth, vec![FnClass::Smooth; m]),
                };
                Self::build(p, objective, &constraints, p.derivatives(), false)
            }
            ProblemRef::BlackBox(p) => {
                let m = p.num_constraints();
                Self::build(p, FnClass::BlackBox, &vec![FnClass::BlackBox; m], Access::Values, p.noisy())
            }
        }
    }

    /// Signature of a search space handled through ask/tell: the caller evaluates, so the functions
    /// are black boxes.
    pub fn of_space<P: Problem + ?Sized>(space: &P) -> Signature {
        let m = space.num_constraints();
        Self::build(space, FnClass::BlackBox, &vec![FnClass::BlackBox; m], Access::Values, false)
    }

    fn build<P: Problem + ?Sized>(
        p: &P,
        objective: FnClass,
        constraints: &[FnClass],
        access: Access,
        noisy: bool,
    ) -> Signature {
        let vars = p.variables();
        let count = |f: fn(&VarKind) -> bool| vars.iter().filter(|v| f(&v.kind)).count();
        let (lower, upper) = constraint_bound_vectors(p);
        let mut constraint_classes = [0; 4];
        for c in constraints {
            constraint_classes[*c as usize] += 1;
        }
        Signature {
            n_variables: vars.len(),
            n_continuous: count(|k| matches!(k, VarKind::Continuous)),
            n_integer: count(|k| matches!(k, VarKind::Integer)),
            n_categorical: count(|k| matches!(k, VarKind::Categorical(_))),
            n_unbounded: vars.iter().filter(|v| !v.is_bounded()).count(),
            n_constraints: lower.len(),
            n_equalities: lower.iter().zip(&upper).filter(|(l, u)| l.is_finite() && l == u).count(),
            objective,
            constraint_classes,
            access,
            sense: p.sense(),
            noisy,
        }
    }

    pub fn n_inequalities(&self) -> usize {
        self.n_constraints - self.n_equalities
    }

    /// The least structured constraint class present, `None` without constraints.
    pub fn max_constraint_class(&self) -> Option<FnClass> {
        [FnClass::BlackBox, FnClass::Smooth, FnClass::Quadratic, FnClass::Linear]
            .into_iter()
            .find(|c| self.constraint_classes[*c as usize] > 0)
    }

    pub fn is_discrete(&self) -> bool {
        self.n_integer + self.n_categorical > 0
    }

    pub fn class(&self) -> ProblemClass {
        use ProblemClass::*;
        let cmax = self.max_constraint_class();
        if self.objective == FnClass::BlackBox || cmax == Some(FnClass::BlackBox) {
            return BlackBox;
        }
        let linear_constraints = cmax.is_none_or(|c| c == FnClass::Linear);
        let quadratic_constraints = cmax.is_none_or(|c| c <= FnClass::Quadratic);
        let at_most_quadratic = self.objective <= FnClass::Quadratic;
        match (self.is_discrete(), self.objective) {
            (false, FnClass::Linear) if linear_constraints => Lp,
            (false, _) if at_most_quadratic && linear_constraints => Qp,
            (false, _) if at_most_quadratic && quadratic_constraints => Qcqp,
            (false, _) => Nlp,
            (true, FnClass::Linear) if linear_constraints => Milp,
            (true, _) if at_most_quadratic && linear_constraints => Miqp,
            (true, _) if at_most_quadratic && quadratic_constraints => Miqcqp,
            (true, _) => Minlp,
        }
    }

    /// One-line form for logs: `NLP (n=4, m=2)`.
    pub fn short(&self) -> String {
        format!("{} (n={}, m={})", self.class(), self.n_variables, self.n_constraints)
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

impl fmt::Display for Signature {
    /// `NLP: 4 variables (4 continuous), 2 constraints (1 equality; 2 smooth nonlinear); objective smooth
    /// nonlinear; first and second derivatives; minimize`
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut kinds = Vec::new();
        for (n, what) in
            [(self.n_continuous, "continuous"), (self.n_integer, "integer"), (self.n_categorical, "categorical")]
        {
            if n > 0 {
                kinds.push(format!("{n} {what}"));
            }
        }
        if self.n_unbounded > 0 {
            kinds.push(format!("{} without finite bounds", self.n_unbounded));
        }
        write!(
            f,
            "{}: {} variable{} ({})",
            self.class(),
            self.n_variables,
            plural(self.n_variables),
            kinds.join(", ")
        )?;
        write!(f, ", {} constraint{}", self.n_constraints, plural(self.n_constraints))?;
        if self.n_constraints > 0 {
            let classes: Vec<String> = [FnClass::Linear, FnClass::Quadratic, FnClass::Smooth, FnClass::BlackBox]
                .into_iter()
                .filter(|c| self.constraint_classes[*c as usize] > 0)
                .map(|c| format!("{} {c}", self.constraint_classes[c as usize]))
                .collect();
            let eq = self.n_equalities;
            write!(f, " ({eq} equalit{}; {})", if eq == 1 { "y" } else { "ies" }, classes.join(", "))?;
        }
        write!(f, "; objective {}; {}; {}", self.objective, self.access, self.sense)?;
        if self.noisy {
            f.write_str("; noisy")?;
        }
        Ok(())
    }
}

/// Problem class, for logs and messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProblemClass {
    Lp,
    Qp,
    Qcqp,
    Nlp,
    Milp,
    Miqp,
    Miqcqp,
    Minlp,
    BlackBox,
}

impl fmt::Display for ProblemClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ProblemClass::Lp => "LP",
            ProblemClass::Qp => "QP",
            ProblemClass::Qcqp => "QCQP",
            ProblemClass::Nlp => "NLP",
            ProblemClass::Milp => "MILP",
            ProblemClass::Miqp => "MIQP",
            ProblemClass::Miqcqp => "MIQCQP",
            ProblemClass::Minlp => "MINLP",
            ProblemClass::BlackBox => "BLACKBOX",
        })
    }
}

/// What a solver handles. Declared by the solver as data; [`Capabilities::verdict`] turns it into a
/// decision with reasons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Information the solver consumes. [`Access::Values`] means the solver receives a
    /// [`BlackBoxOracle`](crate::BlackBoxOracle); otherwise a [`SmoothOracle`](crate::SmoothOracle).
    pub uses: Access,
    /// Least structured objective the solver handles.
    pub objective: FnClass,
    /// Least structured constraint functions the solver handles.
    pub constraints: FnClass,
    pub integer: bool,
    pub categorical: bool,
    /// Every variable must have finite bounds.
    pub finite_bounds: bool,
    pub inequalities: bool,
    pub equalities: bool,
    /// The solver provides [`Solver::session`](crate::Solver::session).
    pub ask_tell: bool,
}

/// Whether a solver can take a problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Supported,
    /// The solver can take the problem only by discarding information the problem provides (its
    /// derivatives: the problem becomes a black box). Automatic dispatch never does this; naming the
    /// solver does.
    Lossy(String),
    Unsupported(Vec<String>),
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Supported => f.write_str("supported"),
            Verdict::Lossy(why) => write!(f, "only when named: {why}"),
            Verdict::Unsupported(reasons) => write!(f, "unsupported: {}", reasons.join("; ")),
        }
    }
}

fn need(access: Access) -> &'static str {
    match access {
        Access::Values => "function values",
        Access::Gradients => "first derivatives",
        Access::Hessians => "second derivatives",
    }
}

impl Capabilities {
    pub fn verdict(&self, sig: &Signature) -> Verdict {
        let mut reasons = Vec::new();
        let short_of_access = self.uses > sig.access;
        if short_of_access {
            reasons.push(format!("needs {}; the problem provides {}", need(self.uses), sig.access));
        }
        // A black-box function is already explained by the missing derivatives.
        if sig.objective > self.objective && !(short_of_access && sig.objective == FnClass::BlackBox) {
            reasons.push(format!("objective is {}; handles at most {}", sig.objective, self.objective));
        }
        if let Some(c) = sig.max_constraint_class() {
            if c > self.constraints && !(short_of_access && c == FnClass::BlackBox) {
                reasons.push(format!("has {c} constraints; handles at most {}", self.constraints));
            }
        }
        if sig.n_integer > 0 && !self.integer {
            reasons.push(format!("does not handle integer variables ({})", sig.n_integer));
        }
        if sig.n_categorical > 0 && !self.categorical {
            reasons.push(format!("does not handle categorical variables ({})", sig.n_categorical));
        }
        if sig.n_unbounded > 0 && self.finite_bounds {
            reasons.push(format!("needs finite bounds on every variable ({} without)", sig.n_unbounded));
        }
        if sig.n_inequalities() > 0 && !self.inequalities {
            reasons.push(format!("does not handle inequality constraints ({})", sig.n_inequalities()));
        }
        if sig.n_equalities > 0 && !self.equalities {
            reasons.push(format!("does not handle equality constraints ({})", sig.n_equalities));
        }
        if !reasons.is_empty() {
            Verdict::Unsupported(reasons)
        } else if self.uses == Access::Values && sig.access > Access::Values {
            Verdict::Lossy("ignores the derivatives and treats the problem as a black box".to_string())
        } else {
            Verdict::Supported
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig() -> Signature {
        Signature {
            n_variables: 3,
            n_continuous: 3,
            n_integer: 0,
            n_categorical: 0,
            n_unbounded: 0,
            n_constraints: 0,
            n_equalities: 0,
            objective: FnClass::Smooth,
            constraint_classes: [0; 4],
            access: Access::Hessians,
            sense: Sense::Minimize,
            noisy: false,
        }
    }

    fn with_constraints(mut s: Signature, classes: [usize; 4], equalities: usize) -> Signature {
        s.constraint_classes = classes;
        s.n_constraints = classes.iter().sum();
        s.n_equalities = equalities;
        s
    }

    #[test]
    fn classes() {
        let mut s = sig();
        s.objective = FnClass::Linear;
        assert_eq!(with_constraints(s.clone(), [2, 0, 0, 0], 1).class(), ProblemClass::Lp);
        assert_eq!(with_constraints(s.clone(), [2, 1, 0, 0], 0).class(), ProblemClass::Qcqp);
        s.objective = FnClass::Quadratic;
        assert_eq!(with_constraints(s.clone(), [2, 0, 0, 0], 0).class(), ProblemClass::Qp);
        assert_eq!(s.class(), ProblemClass::Qp);
        s.n_integer = 1;
        s.n_continuous = 2;
        assert_eq!(with_constraints(s.clone(), [1, 0, 0, 0], 0).class(), ProblemClass::Miqp);
        assert_eq!(with_constraints(s.clone(), [1, 0, 1, 0], 0).class(), ProblemClass::Minlp);
        s.objective = FnClass::Linear;
        assert_eq!(with_constraints(s.clone(), [3, 0, 0, 0], 0).class(), ProblemClass::Milp);
        assert_eq!(sig().class(), ProblemClass::Nlp);
        let mut b = sig();
        b.objective = FnClass::BlackBox;
        assert_eq!(b.class(), ProblemClass::BlackBox);
        assert_eq!(with_constraints(sig(), [0, 0, 0, 1], 0).class(), ProblemClass::BlackBox);
    }

    fn nlp_like() -> Capabilities {
        Capabilities {
            uses: Access::Hessians,
            objective: FnClass::Smooth,
            constraints: FnClass::Smooth,
            integer: false,
            categorical: false,
            finite_bounds: false,
            inequalities: true,
            equalities: true,
            ask_tell: false,
        }
    }

    fn bo_like() -> Capabilities {
        Capabilities {
            uses: Access::Values,
            objective: FnClass::BlackBox,
            constraints: FnClass::BlackBox,
            integer: true,
            categorical: true,
            finite_bounds: true,
            inequalities: true,
            equalities: false,
            ask_tell: true,
        }
    }

    #[test]
    fn verdicts_give_reasons() {
        assert_eq!(nlp_like().verdict(&sig()), Verdict::Supported);
        assert!(matches!(bo_like().verdict(&sig()), Verdict::Lossy(_)));

        let mut first_order = sig();
        first_order.access = Access::Gradients;
        let Verdict::Unsupported(r) = nlp_like().verdict(&first_order) else { panic!() };
        assert_eq!(r, vec!["needs second derivatives; the problem provides first derivatives".to_string()]);

        let mut black_box = with_constraints(sig(), [0, 0, 0, 2], 1);
        black_box.objective = FnClass::BlackBox;
        black_box.access = Access::Values;
        black_box.n_unbounded = 1;
        // Only the access reason, not one per black-box function.
        let Verdict::Unsupported(r) = nlp_like().verdict(&black_box) else { panic!() };
        assert_eq!(r.len(), 1, "{r:?}");
        let Verdict::Unsupported(r) = bo_like().verdict(&black_box) else { panic!() };
        assert_eq!(r.len(), 2, "{r:?}");
        assert!(r[0].contains("finite bounds") && r[1].contains("equality"));
    }

    #[test]
    fn display_mentions_every_part() {
        let mut s = with_constraints(sig(), [1, 0, 1, 0], 1);
        s.n_integer = 1;
        s.n_continuous = 2;
        s.n_unbounded = 2;
        s.sense = Sense::Maximize;
        assert_eq!(
            s.to_string(),
            "MINLP: 3 variables (2 continuous, 1 integer, 2 without finite bounds), 2 constraints (1 equality; \
             1 linear, 1 smooth nonlinear); objective smooth nonlinear; first and second derivatives; maximize"
        );
        assert_eq!(s.short(), "MINLP (n=3, m=2)");
    }
}
