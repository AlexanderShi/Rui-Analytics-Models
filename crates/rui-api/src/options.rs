//! Options: what users give ([`Options`]), what solvers declare ([`OptionSpec`]) and the validated
//! values a solver receives ([`ResolvedOptions`]).
//!
//! Users give key/value pairs in code or as text (`"tol=1e-8 method=nlp"`, the AMPL / Pyomo form).
//! A key is plain (`tol`) or scoped to one solver (`nlp.mu_init`: used only when that solver solves the
//! problem). Rui Analytics Models resolves the keys against the chosen solver's declarations; unknown
//! names, values of the wrong type and values out of range are errors, never silently ignored.

use crate::Error;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// An option value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(b) => f.write_str(if *b { "yes" } else { "no" }),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(x) => write!(f, "{x:?}"),
            Value::Text(s) => f.write_str(s),
        }
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}
impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Int(v.into())
    }
}
impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::Int(v.into())
    }
}
impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Value::Int(i64::try_from(v).unwrap_or(i64::MAX))
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Text(v.to_string())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Text(v)
    }
}

/// Type and admissible values of an option.
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Bool,
    /// Integer in `[min, max]`.
    Int {
        min: i64,
        max: i64,
    },
    /// Number in `[min, max]`, or `(min, max]` when `min_exclusive`.
    Float {
        min: f64,
        max: f64,
        min_exclusive: bool,
    },
    /// One of the listed words (case-insensitive).
    Choice(Vec<&'static str>),
    Text,
}

impl Kind {
    pub fn int_at_least(min: i64) -> Kind {
        Kind::Int { min, max: i64::MAX }
    }
    pub fn int_range(min: i64, max: i64) -> Kind {
        Kind::Int { min, max }
    }
    pub fn float_at_least(min: f64) -> Kind {
        Kind::Float { min, max: f64::INFINITY, min_exclusive: false }
    }
    pub fn positive_float() -> Kind {
        Kind::Float { min: 0.0, max: f64::INFINITY, min_exclusive: true }
    }
    pub fn float_range(min: f64, max: f64) -> Kind {
        Kind::Float { min, max, min_exclusive: false }
    }
    pub fn choice(words: &[&'static str]) -> Kind {
        Kind::Choice(words.to_vec())
    }

    /// `an integer >= 0`, `a number > 0`, `one of a, b`, ...
    pub fn describe(&self) -> String {
        match self {
            Kind::Bool => "yes or no".to_string(),
            Kind::Int { min, max } if *max == i64::MAX => format!("an integer >= {min}"),
            Kind::Int { min, max } => format!("an integer in [{min}, {max}]"),
            Kind::Float { min, max, min_exclusive } => {
                let op = if *min_exclusive { ">" } else { ">=" };
                match (min.is_finite(), max.is_finite()) {
                    (false, false) => "a number".to_string(),
                    (true, false) => format!("a number {op} {min:?}"),
                    (false, true) => format!("a number <= {max:?}"),
                    (true, true) => format!("a number in {}{min:?}, {max:?}]", if *min_exclusive { "(" } else { "[" }),
                }
            }
            Kind::Choice(words) => format!("one of {}", words.join(", ")),
            Kind::Text => "text".to_string(),
        }
    }

    /// Converts a user value (often text) into a value of this kind, or explains why it cannot.
    pub fn coerce(&self, value: &Value) -> Result<Value, String> {
        let wrong = || format!("expected {}, got `{value}`", self.describe());
        match self {
            Kind::Bool => match value {
                Value::Bool(b) => Ok(Value::Bool(*b)),
                Value::Int(0) => Ok(Value::Bool(false)),
                Value::Int(1) => Ok(Value::Bool(true)),
                Value::Text(s) => match s.trim().to_ascii_lowercase().as_str() {
                    "" | "yes" | "true" | "on" | "1" => Ok(Value::Bool(true)),
                    "no" | "false" | "off" | "0" => Ok(Value::Bool(false)),
                    _ => Err(wrong()),
                },
                _ => Err(wrong()),
            },
            Kind::Int { min, max } => {
                let integral = |x: f64| (x.fract() == 0.0 && x.abs() < 9.0e15).then_some(x as i64);
                let i = match value {
                    Value::Int(i) => Some(*i),
                    Value::Float(x) => integral(*x),
                    Value::Text(s) => {
                        s.trim().parse::<i64>().ok().or_else(|| s.trim().parse::<f64>().ok().and_then(integral))
                    }
                    Value::Bool(_) => None,
                }
                .ok_or_else(wrong)?;
                if i < *min || i > *max {
                    return Err(format!("{i} is not {}", self.describe()));
                }
                Ok(Value::Int(i))
            }
            Kind::Float { min, max, min_exclusive } => {
                let x = match value {
                    Value::Float(x) => *x,
                    Value::Int(i) => *i as f64,
                    Value::Text(s) => s.trim().parse::<f64>().map_err(|_| wrong())?,
                    Value::Bool(_) => return Err(wrong()),
                };
                let below = if *min_exclusive { x <= *min } else { x < *min };
                if x.is_nan() || below || x > *max {
                    return Err(format!("{x:?} is not {}", self.describe()));
                }
                Ok(Value::Float(x))
            }
            Kind::Choice(words) => match value {
                Value::Text(s) => words
                    .iter()
                    .find(|w| w.eq_ignore_ascii_case(s.trim()))
                    .map(|w| Value::Text(w.to_string()))
                    .ok_or_else(wrong),
                _ => Err(wrong()),
            },
            Kind::Text => Ok(Value::Text(value.to_string())),
        }
    }
}

/// Declaration of one option of a solver.
#[derive(Clone, Debug, PartialEq)]
pub struct OptionSpec {
    /// Lowercase, `[a-z0-9_]`, no `.`.
    pub name: &'static str,
    /// Other accepted names (for example the name another tool uses for the same thing).
    pub aliases: Vec<&'static str>,
    pub kind: Kind,
    /// Value used when the user gives none; `None` means "unset".
    pub default: Option<Value>,
    pub help: &'static str,
}

impl OptionSpec {
    pub fn new(name: &'static str, kind: Kind, help: &'static str) -> Self {
        Self { name, aliases: Vec::new(), kind, default: None, help }
    }
    pub fn with_default(mut self, value: impl Into<Value>) -> Self {
        self.default = Some(value.into());
        self
    }
    pub fn alias(mut self, name: &'static str) -> Self {
        self.aliases.push(name);
        self
    }
    /// Whether `name` is this option's name or one of its aliases.
    pub fn answers_to(&self, name: &str) -> bool {
        self.name == name || self.aliases.contains(&name)
    }
}

/// Options with the same meaning for every solver.
///
/// The *universal* ones are accepted for every solve: `method` is consumed by dispatch, `time_limit`
/// and `verbosity` are applied through the [`Context`](crate::Context) (every solver must honor them).
/// The *optional* ones are honored by a solver when it lists them in
/// [`Solver::options`](crate::Solver::options), built with the functions below so that names, types and
/// help texts agree across solvers; when the chosen solver does not list one, it is ignored with a notice.
pub mod common {
    use super::{Kind, OptionSpec};

    pub const METHOD: &str = "method";
    pub const TIME_LIMIT: &str = "time_limit";
    pub const VERBOSITY: &str = "verbosity";
    pub const MAX_ITERATIONS: &str = "max_iterations";
    pub const MAX_EVALUATIONS: &str = "max_evaluations";
    pub const TOL: &str = "tol";
    pub const SEED: &str = "seed";

    /// The optional common options.
    pub const OPTIONAL: &[&str] = &[MAX_ITERATIONS, MAX_EVALUATIONS, TOL, SEED];

    pub fn universal() -> Vec<OptionSpec> {
        vec![
            OptionSpec::new(METHOD, Kind::Text, "solver id; absent or `auto`: chosen from the problem"),
            OptionSpec::new(TIME_LIMIT, Kind::float_at_least(0.0), "wall-clock limit in seconds")
                .with_default(f64::INFINITY),
            OptionSpec::new(
                VERBOSITY,
                Kind::int_range(0, 3),
                "0 silent, 1 summary, 2 one line per iteration or evaluation, 3 details",
            )
            .with_default(1i64),
        ]
    }

    pub fn max_iterations(default: i64) -> OptionSpec {
        OptionSpec::new(MAX_ITERATIONS, Kind::int_at_least(0), "iteration limit").with_default(default)
    }
    pub fn max_evaluations(default: i64) -> OptionSpec {
        OptionSpec::new(MAX_EVALUATIONS, Kind::int_at_least(1), "evaluation budget").with_default(default)
    }
    pub fn tol(default: f64) -> OptionSpec {
        OptionSpec::new(TOL, Kind::positive_float(), "convergence tolerance").with_default(default)
    }
    pub fn seed(default: i64) -> OptionSpec {
        OptionSpec::new(SEED, Kind::int_at_least(0), "random seed").with_default(default)
    }
}

/// Options as given by a user: ordered key/value pairs. Keys are stored in lowercase.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    entries: Vec<(String, Value)>,
}

impl Options {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder form of [`set`](Self::set).
    pub fn with(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.set(key, value);
        self
    }

    /// Sets `key`, replacing an earlier value of the same key.
    pub fn set(&mut self, key: &str, value: impl Into<Value>) {
        let key = key.trim().to_ascii_lowercase();
        let value = value.into();
        match self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = value,
            None => self.entries.push((key, value)),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        let key = key.to_ascii_lowercase();
        self.entries.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Sets every entry of `other`, which takes precedence.
    pub fn extend(&mut self, other: &Options) {
        for (k, v) in other.iter() {
            self.set(k, v.clone());
        }
    }

    /// Parses whitespace-separated `key=value` tokens; a value may be quoted (`"..."` or `'...'`) to
    /// contain spaces, and a bare `key` means `key=yes`. Values stay text until resolved against a
    /// solver's declarations.
    pub fn parse(text: &str) -> Result<Options, Error> {
        let mut out = Options::new();
        let mut it = text.chars().peekable();
        loop {
            while it.peek().is_some_and(|c| c.is_whitespace()) {
                it.next();
            }
            if it.peek().is_none() {
                return Ok(out);
            }
            let mut key = String::new();
            while let Some(&c) = it.peek() {
                if c.is_whitespace() || c == '=' {
                    break;
                }
                key.push(c);
                it.next();
            }
            if key.is_empty() {
                return Err(Error::InvalidOption(format!("expected an option name before `=` in `{text}`")));
            }
            if it.peek() != Some(&'=') {
                out.set(&key, Value::Bool(true));
                continue;
            }
            it.next();
            let mut value = String::new();
            match it.peek().copied() {
                Some(quote @ ('"' | '\'')) => {
                    it.next();
                    loop {
                        match it.next() {
                            Some(c) if c == quote => break,
                            Some(c) => value.push(c),
                            None => {
                                return Err(Error::InvalidOption(format!("unterminated quote in the value of `{key}`")))
                            }
                        }
                    }
                }
                _ => {
                    while let Some(&c) = it.peek() {
                        if c.is_whitespace() {
                            break;
                        }
                        value.push(c);
                        it.next();
                    }
                }
            }
            out.set(&key, Value::Text(value));
        }
    }
}

/// Validated option values a solver receives: every value has the type its declaration gives, and
/// every declared option with a default has a value.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResolvedOptions {
    values: BTreeMap<String, Value>,
    given: BTreeSet<String>,
}

impl ResolvedOptions {
    /// The defaults of `specs` (for calling a solver directly, e.g. in its own tests).
    pub fn defaults(specs: &[OptionSpec]) -> Self {
        let mut out = Self::default();
        for s in specs {
            if let Some(d) = &s.default {
                out.set(s.name, d.clone(), false);
            }
        }
        out
    }

    /// `given`: the user set it (as opposed to a default).
    pub fn set(&mut self, name: &str, value: Value, given: bool) {
        self.values.insert(name.to_string(), value);
        if given {
            self.given.insert(name.to_string());
        } else {
            self.given.remove(name);
        }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    /// Whether the user set `name` (rather than it holding a default).
    pub fn is_given(&self, name: &str) -> bool {
        self.given.contains(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.values.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Value of a declared number option. Panics if `name` has no number value: a solver reading an
    /// option it did not declare (or declared without default) is a bug in the solver.
    pub fn f64(&self, name: &str) -> f64 {
        match self.values.get(name) {
            Some(Value::Float(x)) => *x,
            Some(Value::Int(i)) => *i as f64,
            other => panic!("option `{name}` has no number value ({other:?})"),
        }
    }

    /// Value of a declared integer option (panics like [`f64`](Self::f64)).
    pub fn i64(&self, name: &str) -> i64 {
        match self.values.get(name) {
            Some(Value::Int(i)) => *i,
            other => panic!("option `{name}` has no integer value ({other:?})"),
        }
    }

    /// Value of a declared integer option as `usize` (negative values give 0).
    pub fn usize(&self, name: &str) -> usize {
        usize::try_from(self.i64(name).max(0)).unwrap_or(usize::MAX)
    }

    /// Value of a declared yes/no option (panics like [`f64`](Self::f64)).
    pub fn bool(&self, name: &str) -> bool {
        match self.values.get(name) {
            Some(Value::Bool(b)) => *b,
            other => panic!("option `{name}` has no yes/no value ({other:?})"),
        }
    }

    /// Value of a declared text or choice option (panics like [`f64`](Self::f64)).
    pub fn text(&self, name: &str) -> &str {
        match self.values.get(name) {
            Some(Value::Text(s)) => s,
            other => panic!("option `{name}` has no text value ({other:?})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_text_options() {
        let o = Options::parse("  tol=1e-8 Method=nlp wantsol  name=\"a b\" empty= x='y z'").unwrap();
        let pairs: Vec<(String, Value)> = o.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        assert_eq!(
            pairs,
            vec![
                ("tol".into(), Value::Text("1e-8".into())),
                ("method".into(), Value::Text("nlp".into())),
                ("wantsol".into(), Value::Bool(true)),
                ("name".into(), Value::Text("a b".into())),
                ("empty".into(), Value::Text("".into())),
                ("x".into(), Value::Text("y z".into())),
            ]
        );
        assert!(Options::parse("tol = 1").is_err());
        assert!(Options::parse("a=\"open").is_err());
        assert_eq!(Options::parse("tol=1 tol=2").unwrap().get("TOL"), Some(&Value::Text("2".into())));
    }

    #[test]
    fn coerce_by_kind() {
        let int = Kind::int_at_least(0);
        assert_eq!(int.coerce(&"12".into()), Ok(Value::Int(12)));
        assert_eq!(int.coerce(&"1e3".into()), Ok(Value::Int(1000)));
        assert_eq!(int.coerce(&Value::Float(3.0)), Ok(Value::Int(3)));
        assert!(int.coerce(&Value::Float(3.5)).is_err());
        assert_eq!(int.coerce(&Value::Int(-1)), Err("-1 is not an integer >= 0".to_string()));

        let tol = Kind::positive_float();
        assert_eq!(tol.coerce(&"1e-8".into()), Ok(Value::Float(1e-8)));
        assert_eq!(tol.coerce(&Value::Int(1)), Ok(Value::Float(1.0)));
        assert_eq!(tol.coerce(&Value::Float(0.0)), Err("0.0 is not a number > 0.0".to_string()));
        assert!(tol.coerce(&"nan".into()).is_err());
        assert_eq!(Kind::float_at_least(0.0).coerce(&"inf".into()), Ok(Value::Float(f64::INFINITY)));

        assert_eq!(Kind::Bool.coerce(&"No".into()), Ok(Value::Bool(false)));
        assert_eq!(Kind::Bool.coerce(&Value::Bool(true)), Ok(Value::Bool(true)));
        assert!(Kind::Bool.coerce(&"maybe".into()).is_err());

        let choice = Kind::choice(&["logei", "ucb"]);
        assert_eq!(choice.coerce(&"LogEI".into()), Ok(Value::Text("logei".into())));
        assert_eq!(choice.coerce(&"pi".into()), Err("expected one of logei, ucb, got `pi`".to_string()));
    }

    #[test]
    fn resolved_getters_and_defaults() {
        let specs = vec![common::tol(1e-6), common::max_iterations(50)];
        let mut r = ResolvedOptions::defaults(&specs);
        assert_eq!((r.f64("tol"), r.i64("max_iterations"), r.usize("max_iterations")), (1e-6, 50, 50));
        assert!(!r.is_given("tol"));
        r.set("tol", Value::Float(1e-9), true);
        assert!(r.is_given("tol"));
        assert_eq!(r.f64("tol"), 1e-9);
    }
}
