//! Waiting-queue scheduling policy for the engine.
//!
//! The `Scheduler` trait picks which waiting request to admit next, given an admissibility
//! filter (the engine enforces LoRA slot caps and token budget externally). Two impls
//! reproduce the existing `next_admissible_index` logic from `engine.rs`: FCFS (arrival
//! order) and priority (smallest `(priority, arrival_time)` among admissible).

use std::cmp::Ordering;
use std::collections::VecDeque;

use vllm_engine_core_client::protocol::EngineCoreRequest;

/// Strategy for picking the next waiting request to admit into the running batch.
pub(crate) trait Scheduler: Send {
    /// Return the index of the next waiting request to admit. Only requests for which
    /// `admissible(request)` returns true are considered. Returns `None` when nothing
    /// qualifies.
    fn next_admissible(
        &self,
        waiting: &VecDeque<Box<EngineCoreRequest>>,
        admissible: &dyn Fn(&EngineCoreRequest) -> bool,
    ) -> Option<usize>;
}

/// First-come, first-served: the first admissible request in arrival order.
pub(crate) struct Fcfs;

impl Scheduler for Fcfs {
    fn next_admissible(
        &self,
        waiting: &VecDeque<Box<EngineCoreRequest>>,
        admissible: &dyn Fn(&EngineCoreRequest) -> bool,
    ) -> Option<usize> {
        waiting.iter().position(|r| admissible(r))
    }
}

/// Priority: the admissible request with the smallest `(priority, arrival_time)`.
pub(crate) struct Priority;

impl Scheduler for Priority {
    fn next_admissible(
        &self,
        waiting: &VecDeque<Box<EngineCoreRequest>>,
        admissible: &dyn Fn(&EngineCoreRequest) -> bool,
    ) -> Option<usize> {
        waiting
            .iter()
            .enumerate()
            .filter(|(_, r)| admissible(r))
            .min_by(|(_, a), (_, b)| {
                a.priority.cmp(&b.priority).then_with(|| {
                    a.arrival_time
                        .partial_cmp(&b.arrival_time)
                        .unwrap_or(Ordering::Equal)
                })
            })
            .map(|(i, _)| i)
    }
}

/// Admits the admissible waiting request with the fewest prompt tokens. Ties broken
/// by arrival order (lower index wins).
#[cfg(test)]
pub(crate) struct ShortestPromptFirst;

#[cfg(test)]
impl Scheduler for ShortestPromptFirst {
    fn next_admissible(
        &self,
        waiting: &VecDeque<Box<EngineCoreRequest>>,
        admissible: &dyn Fn(&EngineCoreRequest) -> bool,
    ) -> Option<usize> {
        waiting
            .iter()
            .enumerate()
            .filter(|(_, r)| admissible(r))
            .min_by(|(i_a, a), (i_b, b)| {
                let len_a = a.prompt_token_ids.as_ref().map(Vec::len).unwrap_or(0);
                let len_b = b.prompt_token_ids.as_ref().map(Vec::len).unwrap_or(0);
                len_a.cmp(&len_b).then(i_a.cmp(i_b))
            })
            .map(|(i, _)| i)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use vllm_engine_core_client::protocol::{EngineCoreRequest, EngineCoreSamplingParams};

    use crate::sched::{Fcfs, Priority, Scheduler};

    fn req(id: &str, priority: i32) -> Box<EngineCoreRequest> {
        let mut r = EngineCoreRequest {
            request_id: id.to_string(),
            prompt_token_ids: Some(vec![0; 4]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 10,
                ..EngineCoreSamplingParams::for_test()
            }),
            priority,
            ..Default::default()
        };
        r.arrival_time = priority as f64;
        Box::new(r)
    }

    #[test]
    fn fcfs_picks_first_admissible() {
        let mut q = VecDeque::new();
        q.push_back(req("a", 5));
        q.push_back(req("b", 1));
        q.push_back(req("c", 3));

        // All admissible: picks index 0.
        let idx = Fcfs.next_admissible(&q, &|_| true);
        assert_eq!(idx, Some(0));

        // First blocked: skips to index 1.
        let idx = Fcfs.next_admissible(&q, &|r| r.request_id != "a");
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn fcfs_returns_none_when_all_blocked() {
        let mut q = VecDeque::new();
        q.push_back(req("x", 0));
        assert_eq!(Fcfs.next_admissible(&q, &|_| false), None);
    }

    #[test]
    fn priority_picks_smallest_priority() {
        let mut q = VecDeque::new();
        q.push_back(req("a", 10));
        q.push_back(req("b", 1));
        q.push_back(req("c", 5));

        let idx = Priority.next_admissible(&q, &|_| true);
        assert_eq!(idx, Some(1)); // "b" has priority 1
    }

    #[test]
    fn priority_skips_blocked() {
        let mut q = VecDeque::new();
        q.push_back(req("a", 10));
        q.push_back(req("b", 1)); // best but blocked
        q.push_back(req("c", 5));

        let idx = Priority.next_admissible(&q, &|r| r.request_id != "b");
        assert_eq!(idx, Some(2)); // "c" has priority 5, next best
    }

    #[test]
    fn priority_returns_none_on_empty() {
        let q = VecDeque::new();
        assert_eq!(Priority.next_admissible(&q, &|_| true), None);
    }

    fn req_with_prompt(id: &str, prompt_len: usize) -> Box<EngineCoreRequest> {
        Box::new(EngineCoreRequest {
            request_id: id.to_string(),
            prompt_token_ids: Some(vec![0; prompt_len]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 10,
                ..EngineCoreSamplingParams::for_test()
            }),
            ..Default::default()
        })
    }

    #[test]
    fn shortest_prompt_first_picks_shortest() {
        use crate::sched::ShortestPromptFirst;
        let mut q = VecDeque::new();
        q.push_back(req_with_prompt("long", 100));
        q.push_back(req_with_prompt("short", 5));
        q.push_back(req_with_prompt("mid", 50));

        let idx = ShortestPromptFirst.next_admissible(&q, &|_| true);
        assert_eq!(idx, Some(1)); // "short" has 5 tokens
    }

    #[test]
    fn shortest_prompt_first_breaks_ties_by_arrival_order() {
        use crate::sched::ShortestPromptFirst;
        let mut q = VecDeque::new();
        q.push_back(req_with_prompt("first", 10));
        q.push_back(req_with_prompt("second", 10));
        q.push_back(req_with_prompt("third", 10));

        let idx = ShortestPromptFirst.next_admissible(&q, &|_| true);
        assert_eq!(idx, Some(0)); // tied, so first arrival wins
    }

    #[test]
    fn shortest_prompt_first_skips_blocked() {
        use crate::sched::ShortestPromptFirst;
        let mut q = VecDeque::new();
        q.push_back(req_with_prompt("long", 100));
        q.push_back(req_with_prompt("short", 5)); // shortest but blocked
        q.push_back(req_with_prompt("mid", 50));

        let idx = ShortestPromptFirst.next_admissible(&q, &|r| r.request_id != "short");
        assert_eq!(idx, Some(2)); // "mid" is next shortest admissible
    }

    #[test]
    fn shortest_prompt_first_returns_none_on_empty() {
        use crate::sched::ShortestPromptFirst;
        let q = VecDeque::new();
        assert_eq!(ShortestPromptFirst.next_admissible(&q, &|_| true), None);
    }
}
