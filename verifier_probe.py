"""
Read-only probe for the grounding-verifier bug (Apollo case).

Save as scripts/verifier_probe.py (or the repo root) and run from the repo root:
    python verifier_probe.py

It does NOT modify app code. It captures the exact prompt that
GroundedAnswerGenerator._verify_answer builds, then replays it against
llama3.2:3b under several variants, 3 runs each, so you can see which
change flips the verdict. Negative controls (unsupported / contradictory
answers) are run too, so we never pick a variant that just says "true".
"""
import json
import re
import sys

from langchain_ollama import ChatOllama

sys.path.insert(0, ".")
from app.rag.answer_generator import GroundedAnswerGenerator  # noqa: E402

MODEL = "llama3.2:3b"
RUNS = 3

CONTEXT = """[S1]
Source location: rag_smoke_test.txt
Evidence:
Project: Apollo Enterprise
Project Manager: Priya Sharma
Project Location: Hyderabad
Project Status: In Progress
Target Completion Date: 30 December 2026
Current Risk: Delayed delivery of the reporting module."""

GOOD = (
    "According to the evidence, the project manager of Project Apollo "
    "Enterprise is Priya Sharma [S1]. The project is located in Hyderabad "
    "[S1]. The current risk is the delayed delivery of the reporting "
    "module [S1]."
)
BAD_RISK = GOOD.replace("the delayed delivery of the reporting module",
                        "a cybersecurity breach")
BAD_LOC = GOOD.replace("Hyderabad", "Delhi")

CASES = {"GOOD (should be True)": GOOD,
         "BAD_RISK (should be False)": BAD_RISK,
         "BAD_LOC (should be False)": BAD_LOC}


class Capture:
    """Stand-in verifier that records the exact prompt the app builds."""
    def __init__(self):
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return '{"supported": false}'


def exact_prompt(answer):
    cap = Capture()
    GroundedAnswerGenerator(llm=object(), verifier=cap)._verify_answer(
        answer, CONTEXT)
    return cap.prompt


def parse(text):
    try:
        data = json.loads(text.strip())
        return data.get("supported") if isinstance(data, dict) else None
    except Exception:
        return None


def ask(llm, prompt):
    r = llm.invoke(prompt)
    meta = getattr(r, "response_metadata", {}) or {}
    return r.content, meta.get("prompt_eval_count")


llm_default = ChatOllama(model=MODEL, temperature=0, format="json")
llm_bigctx = ChatOllama(model=MODEL, temperature=0, format="json",
                        num_ctx=8192)

REASON_FIRST = ('Return ONLY one JSON object: {"reason": "<one short '
                'sentence>", "supported": true or false}')
OLD_FORMAT = ('Return ONLY one JSON object in this exact format:\n'
              '{"supported": true}\nor\n{"supported": false}')


def v_exact(answer, llm):
    return ask(llm, exact_prompt(answer))


def v_no_preamble(answer, llm):
    cleaned = re.sub(r"^According to the evidence,\s*", "", answer)
    return ask(llm, exact_prompt(cleaned))


def v_reason_first(answer, llm):
    p = exact_prompt(answer)
    if OLD_FORMAT not in p:
        print("  [warn] output-format block not found; variant invalid")
    return ask(llm, p.replace(OLD_FORMAT, REASON_FIRST))


def v_per_claim(answer, llm):
    body = re.sub(r"^According to the evidence,\s*", "", answer)
    claims = [c for c in re.split(r"(?<=[.!?])\s+", body) if c.strip()]
    verdicts, tokens = [], None
    for claim in claims:
        p = (f"EVIDENCE:\n{CONTEXT}\n\nCLAIM: {claim}\n\n"
             "Is the claim supported by the evidence? Ignore citation "
             "markers like [S1]. Return ONLY JSON: "
             '{"reason": "<one short sentence>", "supported": true or false}')
        text, tokens = ask(llm, p)
        verdicts.append(parse(text))
    ok = all(v is True for v in verdicts)
    return json.dumps({"supported": ok, "claims": verdicts}), tokens


VARIANTS = [
    ("1 current prompt, default ctx", v_exact, llm_default),
    ("2 current prompt, num_ctx=8192", v_exact, llm_bigctx),
    ("3 current prompt, no 'According...' preamble", v_no_preamble,
     llm_default),
    ("4 reason-first schema", v_reason_first, llm_default),
    ("5 per-claim + reason-first", v_per_claim, llm_default),
]

if __name__ == "__main__":
    probe = exact_prompt(GOOD)
    print(f"Captured verifier prompt: {len(probe)} chars\n")
    for name, fn, llm in VARIANTS:
        print(f"== {name}")
        for case, answer in CASES.items():
            results, toks = [], None
            for _ in range(RUNS):
                text, toks = fn(answer, llm)
                v = parse(text)
                results.append(v)
            print(f"  {case:28s} -> {results}   prompt_tokens={toks}")
        print()
    print("Reading the output: a usable variant gives True x3 for GOOD and "
          "False x3 for both BAD cases.")
