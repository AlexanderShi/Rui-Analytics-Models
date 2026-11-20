//! xoshiro256++ (Blackman & Vigna), seeded through SplitMix64.

pub(crate) struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `0..n` (`n >= 1`), without modulo bias.
    pub fn below(&mut self, n: u64) -> u64 {
        let limit = u64::MAX - u64::MAX % n;
        loop {
            let v = self.next_u64();
            if v < limit {
                return v % n;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut r = Rng::new(1);
        let mut counts = [0usize; 3];
        for _ in 0..30_000 {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
            counts[r.below(3) as usize] += 1;
        }
        // Each bucket expects 10000 with standard deviation about 82.
        assert!(counts.iter().all(|&c| (9_600..10_400).contains(&c)), "{counts:?}");
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }
}
