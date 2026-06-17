//! Canonical deployment-config fingerprint.
//!
//! The CI profile-once/replay-many cache and the conformance gate both need a
//! stable answer to "was this trace captured under a config I can replay
//! against?". That answer is a hash of the deployment inputs that actually
//! change replay behavior. Computing it from a typed struct (rather than
//! passing an opaque `--config-hash` string by hand) means capture and replay
//! derive the same value from the same inputs, so a mismatch is always a real
//! config difference, never a typo.
//!
//! The canonical form is versioned (`SCHEME`). If the input set ever changes,
//! bump the scheme: old hashes then deliberately stop matching, because a trace
//! captured under the old input set is no longer interchangeable.

use sha2::{Digest, Sha256};

/// Canonical-form scheme tag. Bump when the fingerprint inputs change.
///
/// v3 stops enumerating individual knobs (model/tp/block_size/max_num_seqs/
/// prefix_caching/speculative — the piecemeal trap) and folds the whole deployed
/// *behavioral* engine config into one `engine_config` digest, supplied by the
/// capture driver (which owns the engine flags). The rule is "hardware + version +
/// deployed behavioral flags": defaults aren't enumerated because `vllm_tag` pins
/// them, and transport/addressing flags are excluded by the driver. Older goldens
/// keep their own scheme's hashes and stay valid (the sim compares the stamped
/// hash, it never recomputes), so the bump only affects new captures.
const SCHEME: &str = "config-fingerprint-v3";

/// The deployment inputs that determine whether a captured trace is valid to
/// replay. Two deployments with the same fingerprint produce interchangeable
/// traces; any difference means the trace must not be replayed against the other
/// config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigFingerprint {
    /// GPU type, e.g. `"H200"`. Hardware, not an engine flag; affects absolute
    /// latencies, so it is fingerprinted explicitly.
    pub gpu: String,
    /// vLLM line/tag the engine spoke, e.g. `"v0.23.0"`. Pins the version (and thus
    /// every engine default not overridden by a flag below).
    pub vllm_tag: String,
    /// Canonical, order-stable string of the *behavioral* engine flags the capture
    /// deployed (model, tensor-parallel, gpu-mem-util, max-model-len, max-num-seqs,
    /// block-size, enforce-eager, prefix-caching, speculative config, ...), as
    /// produced by the capture driver (`gen-capture-jobs.py`). Transport/addressing
    /// flags are excluded. This subsumes the per-knob fields older schemes enumerated.
    pub engine_config: String,
}

impl ConfigFingerprint {
    /// The exact, order-fixed string that gets hashed. Never reorder fields:
    /// the order is part of the contract.
    fn canonical(&self) -> String {
        format!(
            "{SCHEME}\n\
             gpu={}\n\
             vllm_tag={}\n\
             engine_config={}\n",
            self.gpu, self.vllm_tag, self.engine_config,
        )
    }

    /// Lowercase-hex SHA-256 of the canonical form. This is the value stamped
    /// into `TraceMeta.config_hash` at capture and checked via
    /// `--expect-config-hash` at replay.
    pub fn hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical().as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use crate::config_hash::ConfigFingerprint;

    fn sample() -> ConfigFingerprint {
        ConfigFingerprint {
            gpu: "H200".to_string(),
            vllm_tag: "v0.23.0".to_string(),
            engine_config: "block_size=16;enforce_eager=true;enable_prefix_caching=true;\
                            gpu_memory_utilization=0.9;max_model_len=16384;max_num_seqs=256;\
                            model=Qwen/Qwen3-8B;speculative=none;tensor_parallel=1"
                .to_string(),
        }
    }

    #[test]
    fn hash_is_deterministic() {
        assert_eq!(sample().hash(), sample().hash());
        // 64 hex chars = 32-byte sha256.
        assert_eq!(sample().hash().len(), 64);
        assert!(sample().hash().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn every_field_changes_the_hash() {
        let base = sample().hash();
        let mutated = [
            ConfigFingerprint {
                gpu: "A100".to_string(),
                ..sample()
            },
            ConfigFingerprint {
                vllm_tag: "v0.22.1".to_string(),
                ..sample()
            },
            // A different deployed config (here: prefix caching off) -> different digest.
            ConfigFingerprint {
                engine_config: sample()
                    .engine_config
                    .replace("enable_prefix_caching=true", "enable_prefix_caching=false"),
                ..sample()
            },
        ];
        for m in mutated {
            assert_ne!(base, m.hash(), "changing a field must change the hash");
        }
    }

    #[test]
    fn field_labels_prevent_value_bleed() {
        // gpu="H200" + tag="v0.23.0" must not collide with gpu="H200v0.23.0" + tag="":
        // the labels + newlines in the canonical form keep field values from merging.
        let a = ConfigFingerprint {
            gpu: "H200".to_string(),
            vllm_tag: "v0.23.0".to_string(),
            engine_config: String::new(),
        };
        let b = ConfigFingerprint {
            gpu: "H200v0.23.0".to_string(),
            vllm_tag: String::new(),
            engine_config: String::new(),
        };
        assert_ne!(a.hash(), b.hash());
    }
}
