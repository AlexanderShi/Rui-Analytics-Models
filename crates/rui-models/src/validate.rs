//! Problem validation before dispatch: the guarantees `rui_api::solver` promises to solvers.

use rui_api::{constraint_bound_vectors, Error, FnClass, Problem, ProblemRef, SmoothOracle, VarKind};

pub(crate) fn problem(p: ProblemRef<'_>) -> Result<(), Error> {
    match p {
        ProblemRef::Smooth(s) => {
            space(s)?;
            smooth(s)
        }
        ProblemRef::BlackBox(b) => space(b),
    }
}

/// Variables and constraint bounds.
pub(crate) fn space<P: Problem + ?Sized>(p: &P) -> Result<(), Error> {
    let fail = |m: String| Err(Error::InvalidProblem(m));
    let vars = p.variables();
    if vars.is_empty() {
        return fail("the problem has no variables".to_string());
    }
    for (j, v) in vars.iter().enumerate() {
        let who = if v.name.is_empty() { format!("variable {j}") } else { format!("variable {j} (`{}`)", v.name) };
        let (l, u) = (v.lower, v.upper);
        if matches!(&v.kind, VarKind::Categorical(levels) if levels.is_empty()) {
            return fail(format!("{who}: a categorical variable needs at least one level"));
        }
        if l.is_nan() || u.is_nan() {
            return fail(format!("{who}: a bound is NaN"));
        }
        if l > u {
            return fail(format!("{who}: lower bound {l:?} exceeds upper bound {u:?}"));
        }
        if l == f64::INFINITY || u == f64::NEG_INFINITY {
            return fail(format!("{who}: the bounds [{l:?}, {u:?}] admit no value"));
        }
        match &v.kind {
            VarKind::Continuous => {}
            VarKind::Integer => {
                if (l.is_finite() && l.fract() != 0.0) || (u.is_finite() && u.fract() != 0.0) {
                    return fail(format!(
                        "{who}: the bounds of an integer variable must be integers, got [{l:?}, {u:?}]"
                    ));
                }
            }
            VarKind::Categorical(levels) => {
                if l != 0.0 || u != levels.len() as f64 - 1.0 {
                    return fail(format!(
                        "{who}: the bounds of a categorical variable must be 0 and {}",
                        levels.len() - 1
                    ));
                }
            }
        }
        if v.log_scale && !(l > 0.0) {
            return fail(format!("{who}: a logarithmic scale needs a positive lower bound, got {l:?}"));
        }
        if let Some(s) = v.start {
            if !s.is_finite() || s < l || s > u {
                return fail(format!("{who}: start {s:?} is outside the bounds [{l:?}, {u:?}]"));
            }
            if v.is_discrete() && s.fract() != 0.0 {
                return fail(format!("{who}: the start of a discrete variable must be integral, got {s:?}"));
            }
        }
    }
    let (gl, gu) = constraint_bound_vectors(p);
    for (i, (l, u)) in gl.iter().zip(&gu).enumerate() {
        if l.is_nan() || u.is_nan() {
            return fail(format!("constraint {i}: a bound is NaN (does the problem implement `constraint_bounds`?)"));
        }
        if l > u {
            return fail(format!("constraint {i}: lower bound {l:?} exceeds upper bound {u:?}"));
        }
        if *l == f64::INFINITY || *u == f64::NEG_INFINITY {
            return fail(format!("constraint {i}: the bounds [{l:?}, {u:?}] admit no value"));
        }
    }
    Ok(())
}

fn smooth(p: &dyn SmoothOracle) -> Result<(), Error> {
    if let Some(c) = p.function_classes() {
        if c.constraints.len() != p.num_constraints() {
            return Err(Error::InvalidProblem(format!(
                "function_classes lists {} constraints, the problem has {}",
                c.constraints.len(),
                p.num_constraints()
            )));
        }
        if c.objective == FnClass::BlackBox || c.constraints.contains(&FnClass::BlackBox) {
            return Err(Error::InvalidProblem(
                "a smooth oracle cannot declare black-box functions; use a BlackBoxOracle".to_string(),
            ));
        }
    }
    Ok(())
}
