"""Cold-start interview prompts collected from the user via stdin."""
from __future__ import annotations

import re

PROMPTS = [
    "What's something you've been obsessed with lately?",
    "Drop 3 papers / songs / books that hit different recently.",
    "What are 3 topics you wish you had a smart friend deep into?",
    "What's a question you've been chewing on for weeks?",
    "Name a person whose mind you'd want to inhabit for a day.",
    "What's a small thing that delighted you in the last week?",
]


def interview(input_fn=input, output_fn=print) -> list[dict]:
    """Run the cold-start interview; returns a list of {text, source} dicts."""
    answers: list[dict] = []
    output_fn(
        "DMN — taste interview. Press Enter to skip a prompt; Ctrl-C to stop early."
    )
    for i, q in enumerate(PROMPTS, 1):
        try:
            ans = input_fn(f"\n  [{i}/{len(PROMPTS)}] {q}\n    ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not ans:
            continue
        pieces = _split_answer(ans)
        if pieces:
            output_fn(f"    noted: {'; '.join(pieces)}")
        else:
            output_fn("    (too short to use — skipped)")
        for piece in pieces:
            answers.append({"text": piece, "source": "manual"})
    return answers


def _split_answer(s: str) -> list[str]:
    """Split a multi-item answer on commas/semicolons/newlines but keep short answers whole."""
    parts = re.split(r"[,;\n]+", s)
    return [p.strip() for p in parts if len(p.strip()) > 1]
