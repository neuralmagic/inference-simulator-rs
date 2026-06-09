//! Token-generation strategy for the engine.
//!
//! The `TokenSource` trait decouples *what tokens a request emits* from the engine loop
//! that paces them. The default `RandomTokens` reproduces the original behavior: uniform
//! random draws from `0..vocab_size` using the per-request seeded rng.

use rand::Rng as _;
use rand::rngs::StdRng;

/// Context passed to `TokenSource::next_tokens` so implementations can condition on
/// request state without holding a reference to the full `ActiveRequest`.
#[allow(dead_code)]
pub(crate) struct TokenCtx<'a> {
    pub request_id: &'a str,
    pub prompt_token_ids: &'a [u32],
    pub num_generated: usize,
}

/// Strategy for producing output tokens for a request.
///
/// `rng` is the per-request seeded rng; implementations that do not need randomness must
/// still not draw from it (callers rely on a fixed draw order for determinism).
pub(crate) trait TokenSource: Send {
    fn next_tokens(&mut self, ctx: &TokenCtx<'_>, n: usize, rng: &mut StdRng) -> Vec<u32>;

    /// Called when a request finishes or is aborted, so stateful sources can drop any
    /// per-request bookkeeping. The default is a no-op.
    fn on_request_finished(&mut self, _request_id: &str) {}
}

/// The original token strategy: each token is drawn uniformly from `0..vocab_size`.
pub(crate) struct RandomTokens {
    pub vocab_size: u32,
}

impl TokenSource for RandomTokens {
    fn next_tokens(&mut self, _ctx: &TokenCtx<'_>, n: usize, rng: &mut StdRng) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(n);
        for _ in 0..n {
            tokens.push(rng.random_range(0..self.vocab_size));
        }
        tokens
    }
}

/// Replays the request's prompt tokens as output, cycling from the start when
/// `max_tokens` exceeds the prompt length. Draws nothing from the rng.
#[cfg(test)]
pub(crate) struct EchoTokens;

#[cfg(test)]
impl TokenSource for EchoTokens {
    fn next_tokens(&mut self, ctx: &TokenCtx<'_>, n: usize, _rng: &mut StdRng) -> Vec<u32> {
        let prompt = ctx.prompt_token_ids;
        if prompt.is_empty() {
            return vec![0; n];
        }
        (0..n)
            .map(|i| {
                let idx = (ctx.num_generated + i) % prompt.len();
                prompt[idx]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng as _;

    use crate::tokens::{RandomTokens, TokenCtx, TokenSource};

    #[test]
    fn random_tokens_draws_correct_count() {
        let mut src = RandomTokens { vocab_size: 100 };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let ctx = TokenCtx {
            request_id: "t1",
            prompt_token_ids: &[0, 1, 2],
            num_generated: 0,
        };
        let tokens = src.next_tokens(&ctx, 5, &mut rng);
        assert_eq!(tokens.len(), 5);
        assert!(tokens.iter().all(|&t| t < 100));
    }

    #[test]
    fn random_tokens_deterministic_with_same_seed() {
        let make = || {
            let mut src = RandomTokens { vocab_size: 32000 };
            let mut rng = rand::rngs::StdRng::seed_from_u64(7);
            let ctx = TokenCtx {
                request_id: "det",
                prompt_token_ids: &[0],
                num_generated: 0,
            };
            src.next_tokens(&ctx, 10, &mut rng)
        };
        assert_eq!(make(), make());
    }

    #[test]
    fn on_request_finished_default_is_noop() {
        let mut src = RandomTokens { vocab_size: 10 };
        // Should not panic.
        src.on_request_finished("anything");
    }

    #[test]
    fn echo_tokens_replays_prompt_ids() {
        use crate::tokens::EchoTokens;
        let mut src = EchoTokens;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let prompt: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let ctx = TokenCtx {
            request_id: "echo",
            prompt_token_ids: &prompt,
            num_generated: 0,
        };
        let tokens = src.next_tokens(&ctx, 8, &mut rng);
        assert_eq!(tokens, prompt, "first 8 tokens echo the prompt exactly");
    }

    #[test]
    fn echo_tokens_cycles_when_exceeding_prompt_len() {
        use crate::tokens::EchoTokens;
        let mut src = EchoTokens;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let prompt: Vec<u32> = vec![1, 2, 3];
        // Already generated 3 (one full cycle), now generating 4 more: should cycle.
        let ctx = TokenCtx {
            request_id: "cycle",
            prompt_token_ids: &prompt,
            num_generated: 3,
        };
        let tokens = src.next_tokens(&ctx, 4, &mut rng);
        assert_eq!(tokens, vec![1, 2, 3, 1]);
    }

    #[test]
    fn echo_tokens_empty_prompt_returns_zeros() {
        use crate::tokens::EchoTokens;
        let mut src = EchoTokens;
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let ctx = TokenCtx {
            request_id: "empty",
            prompt_token_ids: &[],
            num_generated: 0,
        };
        let tokens = src.next_tokens(&ctx, 3, &mut rng);
        assert_eq!(tokens, vec![0, 0, 0]);
    }
}
