"""promptfoo Python provider: runs one scenario through the agent variant given in config."""
from voice_agent_audit.llm import get_backend
from voice_agent_audit.prompts import build_prompt
import os


def call_api(prompt, options, context):
    cfg = options.get("config", {})
    v = context["vars"]
    account = v["account"]
    system = build_prompt(cfg["variant"], account, v["borrower_text"])
    model = os.environ.get("VAA_MODEL", "mock")
    res = get_backend().complete(system, [{"role": "user", "content": v["borrower_text"]}], model)
    return {
        "output": res.text,
        "tokenUsage": {"prompt": res.input_tokens, "completion": res.output_tokens,
                       "total": res.input_tokens + res.output_tokens},
        "metadata": {"token_source": res.token_source, "latency_s": res.latency_s},
    }
