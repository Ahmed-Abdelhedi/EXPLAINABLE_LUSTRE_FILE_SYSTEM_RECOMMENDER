from preference_extractor.evaluation.layer2_fallback_common import PROMPT_POLICY, PROMPT_VERSION, build_prompt, prompt_policy_sha256

def test_v3_version(): assert PROMPT_VERSION == "layer2_qwen_fallback_v3_compact_20260825"
def test_no_signal_very_low(): assert "NO_SIGNAL is NOT VERY_LOW" in PROMPT_POLICY and "can be largely ignored" in PROMPT_POLICY
def test_comparison(): assert "PURE COMPARISON => RELATIVE_ONLY" in PROMPT_POLICY
def test_scale(): assert "weakly prioritized -> LOW, not VERY_LOW" in PROMPT_POLICY and "strong importance -> HIGH, not VERY_HIGH" in PROMPT_POLICY
def test_technical_adjective(): assert "TECHNICAL ADJECTIVE != PRIORITY LEVEL" in PROMPT_POLICY
def test_unrequested(): assert "no unrequested dimensions" in PROMPT_POLICY.lower()
def test_build(): assert '["performance", "cost"]' in build_prompt("Performance is more important than cost.",["performance","cost"])
def test_hash(): assert len(prompt_policy_sha256()) == 64
