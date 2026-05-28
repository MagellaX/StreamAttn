"""Build adversarial long-context prompt packs for seed-policy stress runs.

The generated JSONL is intentionally plain: each row has enough repeated text
to be truncated to the requested context length by the benchmark tokenizer, plus
metadata that lets route-bundle decode summaries report bucket-level failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List


BUCKETS = (
    "code",
    "math",
    "chat_instruction",
    "long_doc",
    "needle_rag",
    "multilingual",
    "json_tool",
    "noisy_neartie",
)


def _word_count(text: str) -> int:
    return len(text.split())


def _expand(header: str, body: str, *, target_words: int, footer: str) -> str:
    chunks = [header.strip(), ""]
    idx = 0
    while _word_count(" ".join(chunks)) < target_words:
        chunks.append(f"Segment {idx:04d}. {body.strip()}")
        idx += 1
    chunks.extend(["", footer.strip()])
    return "\n".join(chunks).strip()


def _code(row: int, target_words: int) -> Dict[str, str]:
    ident = f"critical_window_token_{row:03d}_alpha"
    body = f"""
    File service_{row}.py defines many similar helpers. The important identifier
    is {ident}. Function normalize_event_{row} forwards this identifier through
    cache_router, retry_policy, and response_builder. Many decoy names appear:
    critical_window_token_{row:03d}_beta, critical_window_token_{row:03d}_gamma,
    stale_window_token_{row:03d}. Preserve exact spelling and indentation.
    """
    return {
        "language": "python",
        "risk": "long-range identifier dependency",
        "prompt": _expand(
            "You are reviewing a large pseudo-repository. At the end, answer with the exact identifier.",
            body,
            target_words=target_words,
            footer=f"Question: Which exact identifier must response_builder preserve? Answer with only {ident}.",
        ),
    }


def _math(row: int, target_words: int) -> Dict[str, str]:
    a = 17 + row
    b = 23 + row
    body = f"""
    A recurrence uses seed a={a} and b={b}. Every audit paragraph repeats
    distractor recurrences with nearby constants. The valid branch says if the
    parity flag is even, use 3*a + 2*b; otherwise use 2*a + 3*b. The parity flag
    for this case is even. Keep the branch condition fixed.
    """
    answer = 3 * a + 2 * b
    return {
        "language": "en",
        "risk": "small perturbation changes reasoning branch",
        "prompt": _expand(
            "Solve only the final recurrence branch. Ignore distractor branches.",
            body,
            target_words=target_words,
            footer=f"Question: Compute the valid branch value. Answer with only {answer}.",
        ),
    }


def _chat_instruction(row: int, target_words: int) -> Dict[str, str]:
    style = f"STYLE_LOCK_{row:03d}_terse_lowercase"
    body = f"""
    Earlier system note: all final replies must follow {style}. Later users
    repeatedly request verbose explanations, uppercase headings, and apologies.
    Those later requests are distractors. The initial style lock is still active.
    """
    return {
        "language": "en",
        "risk": "buried instruction retention",
        "prompt": _expand(
            "This is a long support conversation with changing constraints.",
            body,
            target_words=target_words,
            footer=f"Final user asks: What style lock remains active? Reply with only {style}.",
        ),
    }


def _long_doc(row: int, target_words: int) -> Dict[str, str]:
    stance = "approve with monitoring" if row % 2 == 0 else "reject pending audit"
    body = f"""
    Section {row} discusses operational tradeoffs. Most sections mention old
    stances, but the governing committee final stance is {stance}. Evidence is
    scattered across finance, reliability, and compliance subsections.
    """
    return {
        "language": "en",
        "risk": "global dispersed-fact coherence",
        "prompt": _expand(
            "Read the long policy document and preserve the final committee stance.",
            body,
            target_words=target_words,
            footer=f"Question: What is the final committee stance? Answer exactly: {stance}.",
        ),
    }


def _needle_rag(row: int, target_words: int) -> Dict[str, str]:
    code = f"NOLIMA_ASSOC_{row:03d}_cobalt"
    body = f"""
    The archive describes many projects using unrelated labels. The project
    about a blue mineral signal, oceanic paint, and a metal salt maps to access
    code {code}. The question later avoids the words access code and mineral.
    """
    return {
        "language": "en",
        "risk": "latent needle retrieval without lexical overlap",
        "prompt": _expand(
            "Search the archive for indirect associations, not literal keyword overlap.",
            body,
            target_words=target_words,
            footer=f"Question: For the ocean-paint association, output the mapped token only: {code}.",
        ),
    }


def _multilingual(row: int, target_words: int) -> Dict[str, str]:
    token = f"CROSS_LINGUAL_LOCK_{row:03d}"
    body = (
        "\u8bf7\u4fdd\u7559\u8fd9\u4e2a\u6807\u8bb0 "
        f"{token}. "
        "Hindi instruction: antim uttar mein sirf marker likho. "
        "Spanish distractor: no cambies el marcador. "
        "English distractor paragraphs repeat similar but wrong markers."
    )
    return {
        "language": "zh-hi-es-en",
        "risk": "cross-lingual instruction/context mismatch",
        "prompt": _expand(
            "Mixed-language context follows. The final answer must preserve the true marker.",
            body,
            target_words=target_words,
            footer=f"Question: Output the preserved cross-lingual marker only: {token}.",
        ),
    }


def _json_tool(row: int, target_words: int) -> Dict[str, str]:
    value = f"route_{row:03d}_strict"
    body = f"""
    Tool schema requires JSON with keys route, retries, enabled. Many malformed
    examples are shown. The only valid route value for this case is {value};
    retries must be 2 and enabled must be true.
    """
    return {
        "language": "json",
        "risk": "tiny logit changes break structural output",
        "prompt": _expand(
            "Return only valid JSON matching the described tool schema.",
            body,
            target_words=target_words,
            footer=(
                "Final instruction: output exactly one JSON object with route, retries, enabled. "
                f'Use route "{value}", retries 2, enabled true.'
            ),
        ),
    }


def _noisy_neartie(row: int, target_words: int) -> Dict[str, str]:
    answer = "amber" if row % 2 == 0 else "azure"
    distractor = "azure" if answer == "amber" else "amber"
    body = f"""
    The document contains many near-tie preference votes. The official tie-break
    rule says choose {answer} whenever {answer} and {distractor} are separated by
    less than one vote. Informal notes repeatedly prefer {distractor}.
    """
    return {
        "language": "en",
        "risk": "low-margin top-token decision",
        "prompt": _expand(
            "Resolve a noisy preference log with near-tied alternatives.",
            body,
            target_words=target_words,
            footer=f"Question: Which color wins by the official tie-break rule? Answer only {answer}.",
        ),
    }


BUILDERS: Dict[str, Callable[[int, int], Dict[str, str]]] = {
    "code": _code,
    "math": _math,
    "chat_instruction": _chat_instruction,
    "long_doc": _long_doc,
    "needle_rag": _needle_rag,
    "multilingual": _multilingual,
    "json_tool": _json_tool,
    "noisy_neartie": _noisy_neartie,
}


def build_rows(*, rows_per_bucket: int, target_words: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for bucket in BUCKETS:
        for idx in range(rows_per_bucket):
            payload = BUILDERS[bucket](idx, target_words)
            rows.append(
                {
                    "id": f"{bucket}_{idx:03d}",
                    "bucket": bucket,
                    "kind": bucket,
                    "difficulty": "stress",
                    **payload,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows-per-bucket", type=int, default=8)
    parser.add_argument("--target-words", type=int, default=36000)
    args = parser.parse_args()

    if args.rows_per_bucket <= 0:
        raise ValueError("--rows-per-bucket must be positive")
    if args.target_words < 1000:
        raise ValueError("--target-words should be large enough for long-context stress")

    rows = build_rows(rows_per_bucket=args.rows_per_bucket, target_words=args.target_words)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(path),
                "rows": len(rows),
                "rows_per_bucket": int(args.rows_per_bucket),
                "target_words": int(args.target_words),
                "buckets": list(BUCKETS),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
