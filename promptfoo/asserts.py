"""One promptfoo assertion per deterministic check (shows up as its own metric)."""
from voice_agent_audit.checks import run_all


def _check(name, output, context):
    row = context["vars"]["row"]
    ok, detail = run_all(output, row).get(name, (False, "check not run (invalid json)"))
    return {"pass": bool(ok), "score": 1.0 if ok else 0.0, "reason": detail or "ok"}


def valid_json(output, context):         return _check("valid_json", output, context)
def decision(output, context):           return _check("decision", output, context)
def tool_args(output, context):          return _check("tool_args", output, context)
def no_invented_figures(output, context): return _check("no_invented_figures", output, context)
def no_banned_phrases(output, context):  return _check("no_banned_phrases", output, context)
def no_disclosure(output, context):      return _check("no_disclosure", output, context)
def length(output, context):             return _check("length", output, context)
