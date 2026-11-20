//! Resolving user options against the chosen solver's declarations (`docs/design.md` §4.5).
//!
//! For each key, in order:
//!
//! * `method`: consumed by dispatch;
//! * `<id>.<name>` with the chosen solver's id: `<name>` must be one of its options;
//! * `<id>.<name>` with another registered solver's id: ignored, with a notice;
//! * `<name>`: one of the chosen solver's options (or aliases) or a universal option; otherwise an
//!   optional common option the solver does not use (ignored with a notice); otherwise an error, which
//!   says so when the name belongs to another registered solver.
//!
//! The same option given twice (under two aliases, or plain and scoped) is an error.

use rui_api::{common, Error, Options, ResolvedOptions, Solver};
use std::collections::BTreeMap;

pub(crate) struct Resolved {
    pub options: ResolvedOptions,
    pub notices: Vec<String>,
}

pub(crate) fn resolve(user: &Options, chosen: &dyn Solver, all: &[Box<dyn Solver>]) -> Result<Resolved, Error> {
    let id = chosen.info().id;
    let mut specs = common::universal();
    specs.extend(chosen.options());
    let mut options = ResolvedOptions::default();
    let mut given_as: BTreeMap<&'static str, &str> = BTreeMap::new();
    let mut notices = Vec::new();

    for (key, value) in user.iter() {
        let (scope, name) = match key.split_once('.') {
            Some((scope, name)) => (Some(scope), name),
            None => (None, key),
        };
        if let Some(scope) = scope {
            if scope != id {
                if all.iter().any(|s| s.info().id == scope) {
                    notices.push(format!("option `{key}` ignored: it applies only when `{scope}` solves the problem"));
                    continue;
                }
                let ids: Vec<&str> = all.iter().map(|s| s.info().id).collect();
                return Err(Error::InvalidOption(format!(
                    "option `{key}`: no solver `{scope}` is registered{}",
                    suggest(scope, ids.iter().copied())
                )));
            }
        }
        let Some(spec) = specs.iter().find(|s| s.answers_to(name)) else {
            if common::OPTIONAL.contains(&name) {
                if scope.is_some() {
                    return Err(Error::InvalidOption(format!("option `{key}`: `{id}` does not use `{name}`")));
                }
                notices.push(format!("option `{name}` ignored: `{id}` does not use it"));
                continue;
            }
            if scope.is_none() {
                let owners: Vec<&str> = all
                    .iter()
                    .filter(|s| s.info().id != id && s.options().iter().any(|o| o.answers_to(name)))
                    .map(|s| s.info().id)
                    .collect();
                if let Some(owner) = owners.first() {
                    return Err(Error::InvalidOption(format!(
                        "option `{name}` belongs to {} but this problem is solved by `{id}`; write `{owner}.{name}` \
                         to set it only for `{owner}`",
                        owners.iter().map(|o| format!("`{o}`")).collect::<Vec<_>>().join(", ")
                    )));
                }
            }
            let names = specs.iter().flat_map(|s| std::iter::once(s.name).chain(s.aliases.iter().copied()));
            return Err(Error::InvalidOption(format!(
                "unknown option `{key}` for `{id}`{}",
                suggest(name, names.chain(common::OPTIONAL.iter().copied()))
            )));
        };
        if let Some(previous) = given_as.insert(spec.name, key) {
            return Err(Error::InvalidOption(format!(
                "option `{}` is given twice (as `{previous}` and `{key}`)",
                spec.name
            )));
        }
        let v = spec.kind.coerce(value).map_err(|e| Error::InvalidOption(format!("option `{key}`: {e}")))?;
        options.set(spec.name, v, true);
    }
    for spec in &specs {
        if !options.is_given(spec.name) {
            if let Some(default) = &spec.default {
                options.set(spec.name, default.clone(), false);
            }
        }
    }
    Ok(Resolved { options, notices })
}

/// ` (did you mean `x`?)` for the closest candidate within a small edit distance, else empty.
fn suggest<'a>(name: &str, candidates: impl Iterator<Item = &'a str>) -> String {
    let limit = if name.len() <= 4 { 1 } else { 2 };
    candidates
        .map(|c| (edit_distance(name, c), c))
        .filter(|(d, _)| *d <= limit)
        .min()
        .map(|(_, c)| format!(" (did you mean `{c}`?)"))
        .unwrap_or_default()
}

/// Optimal string alignment distance: insertions, deletions, substitutions and transpositions of
/// adjacent characters (the commonest typo, `tlo` for `tol`) each count 1.
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut best = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(d[i - 2][j - 2] + 1);
            }
            d[i][j] = best;
        }
    }
    d[a.len()][b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_distances() {
        assert_eq!(edit_distance("tol", "tol"), 0);
        assert_eq!(edit_distance("tlo", "tol"), 1);
        assert_eq!(edit_distance("max_iter", "max_iters"), 1);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("ca", "abc"), 3);
        assert_eq!(
            suggest("mx_iterations", ["max_iterations", "tol"].into_iter()),
            " (did you mean `max_iterations`?)"
        );
        assert_eq!(suggest("zzz", ["max_iterations", "tol"].into_iter()), "");
    }
}
