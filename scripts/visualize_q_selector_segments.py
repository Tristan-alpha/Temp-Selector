#!/usr/bin/env python3
"""Render Prefix-Q selector segment-temperature traces as a standalone HTML file."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _temperature_color(temp: float, min_temp: float = 0.1, max_temp: float = 1.5) -> str:
    """Map low temperature to green and high temperature to red."""
    span = max(max_temp - min_temp, 1e-9)
    ratio = min(1.0, max(0.0, (float(temp) - min_temp) / span))
    hue = 120.0 * (1.0 - ratio)
    return f"hsl({hue:.1f} 74% 43%)"


def _decision_title(decision: Dict[str, Any]) -> str:
    parts = [
        f"vote={decision.get('vote')}",
        f"segment={decision.get('segment_index')}",
        f"stage={decision.get('stage', 'unknown')}",
        f"temperature={float(decision.get('temperature', 0.0)):.1f}",
    ]
    if "prefix_value" in decision and decision["prefix_value"] is not None:
        parts.append(f"phi={float(decision['prefix_value']):.3f}")
    if "selected_q" in decision:
        parts.append(f"selected_q={float(decision['selected_q']):.3f}")
    if "margin_to_second" in decision:
        parts.append(f"margin={float(decision['margin_to_second']):.3f}")
    if decision.get("source") == "first_segment":
        parts.append("source=first_segment")
    return " | ".join(parts)


def _group_by_vote(prediction: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for decision in prediction.get("q_decisions", []):
        vote = int(decision.get("vote", 0))
        grouped[vote].append(decision)
    return {
        vote: sorted(items, key=lambda item: int(item.get("segment_index", 0)))
        for vote, items in sorted(grouped.items())
    }


def _unique_dynamic_temperatures(prediction: Dict[str, Any]) -> int:
    temps = {
        float(decision.get("temperature", 0.0))
        for decision in prediction.get("q_decisions", [])
        if decision.get("source") != "first_segment"
    }
    return len(temps)


def _pick_predictions(predictions: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Pick compact traces with visible temperature variation and mixed outcomes."""
    candidates = []
    for index, prediction in enumerate(predictions):
        total_segments = len(prediction.get("q_decisions", []))
        if total_segments < 25:
            continue
        if total_segments > 190:
            continue
        candidates.append({
            "index": index,
            "prediction": prediction,
            "unique_dynamic_temps": _unique_dynamic_temperatures(prediction),
            "total_segments": total_segments,
            "correct": int(prediction.get("majority_correct", 0)),
            "confidence": float(prediction.get("sc_confidence", 0.0)),
        })

    candidates.sort(
        key=lambda item: (
            -item["unique_dynamic_temps"],
            item["correct"],
            abs(item["total_segments"] - 95),
            item["confidence"],
        )
    )

    selected: List[Dict[str, Any]] = []
    seen_outcomes = set()
    for candidate in candidates:
        outcome = candidate["correct"]
        if len(seen_outcomes) < 2 and outcome in seen_outcomes:
            continue
        selected.append(candidate["prediction"])
        seen_outcomes.add(outcome)
        if len(selected) >= n:
            return selected

    for candidate in candidates:
        prediction = candidate["prediction"]
        if prediction in selected:
            continue
        selected.append(prediction)
        if len(selected) >= n:
            break
    return selected


def _legend(temperatures: Iterable[float]) -> str:
    chips = []
    for temp in sorted(set(float(t) for t in temperatures)):
        chips.append(
            f'<span class="legend-chip">'
            f'<span class="swatch" style="background:{_temperature_color(temp)}"></span>'
            f'{temp:.1f}</span>'
        )
    return "\n".join(chips)


def _segment(decision: Dict[str, Any]) -> str:
    temp = float(decision.get("temperature", 0.0))
    title = html.escape(_decision_title(decision), quote=True)
    label = f"{temp:.1f}" if decision.get("segment_index", 0) == 0 else ""
    first_class = " first" if decision.get("source") == "first_segment" else ""
    return (
        f'<span class="segment{first_class}" title="{title}" '
        f'style="background:{_temperature_color(temp)}">{html.escape(label)}</span>'
    )


def _prediction_card(prediction: Dict[str, Any], rank: int) -> str:
    grouped = _group_by_vote(prediction)
    temp_counts = Counter(
        float(decision.get("temperature", 0.0))
        for decision in prediction.get("q_decisions", [])
    )
    dist = " ".join(
        f'<span class="dist-item">{temp:.1f}: {count}</span>'
        for temp, count in sorted(temp_counts.items())
    )
    correct = int(prediction.get("majority_correct", 0))
    badge = "correct" if correct else "wrong"
    rows = []
    individual = prediction.get("individual_correct", [])
    answers = prediction.get("extracted_answers", [])
    for vote, decisions in grouped.items():
        vote_correct = int(individual[vote]) if vote < len(individual) else 0
        answer = str(answers[vote]) if vote < len(answers) else ""
        rows.append(
            '<div class="vote-row">'
            f'<div class="vote-meta">v{vote}<span class="{ "ok" if vote_correct else "bad" }">'
            f'{"ok" if vote_correct else "err"}</span>'
            f'<small>{html.escape(answer)}</small></div>'
            f'<div class="segments">{"".join(_segment(item) for item in decisions)}</div>'
            '</div>'
        )
    return f"""
    <section class="card">
      <div class="card-header">
        <div>
          <h2>{rank}. {html.escape(str(prediction.get("problem_id", "")))}</h2>
          <p>majority answer <code>{html.escape(str(prediction.get("majority_answer", "")))}</code>,
             confidence {float(prediction.get("sc_confidence", 0.0)):.3f},
             answer entropy {float(prediction.get("answer_entropy", 0.0)):.3f}</p>
        </div>
        <span class="badge {badge}">{badge}</span>
      </div>
      <div class="distribution">{dist}</div>
      <div class="vote-grid">
        {''.join(rows)}
      </div>
    </section>
    """


def render_html(data: Dict[str, Any], selected: List[Dict[str, Any]], source: str) -> str:
    all_temps = data.get("allowed_temperatures", [])
    cards = "\n".join(_prediction_card(prediction, i + 1) for i, prediction in enumerate(selected))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prefix-Q Segment Temperature Traces</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5d6972;
      --line: #d7dde2;
      --panel: #ffffff;
      --bg: #f6f8fa;
      --ok: #17694b;
      --bad: #a3372a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px 24px 44px;
    }}
    header {{
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 4px;
      font-size: 16px;
      line-height: 1.35;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #263238;
      background: #eef2f5;
      padding: 1px 5px;
      border-radius: 4px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .metric b {{
      display: block;
      font-size: 18px;
      margin-bottom: 2px;
    }}
    .metric span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      align-items: center;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 18px;
    }}
    .legend-chip {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 13px;
      color: #263238;
    }}
    .swatch {{
      width: 18px;
      height: 12px;
      border-radius: 2px;
      border: 1px solid rgb(0 0 0 / 0.16);
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin: 14px 0;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
    }}
    .badge {{
      flex: 0 0 auto;
      min-width: 68px;
      text-align: center;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .badge.correct {{
      color: var(--ok);
      background: #e6f4ee;
    }}
    .badge.wrong {{
      color: var(--bad);
      background: #fae9e5;
    }}
    .distribution {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin: 12px 0 10px;
    }}
    .dist-item {{
      color: #3d4951;
      background: #f1f4f6;
      border: 1px solid #dce2e6;
      border-radius: 6px;
      padding: 3px 7px;
      font-size: 12px;
    }}
    .vote-grid {{
      display: grid;
      gap: 8px;
    }}
    .vote-row {{
      display: grid;
      grid-template-columns: 108px minmax(0, 1fr);
      align-items: center;
      gap: 8px;
      min-height: 28px;
    }}
    .vote-meta {{
      display: grid;
      grid-template-columns: 26px 32px minmax(0, 1fr);
      align-items: center;
      gap: 5px;
      color: #263238;
      font-size: 12px;
      min-width: 0;
    }}
    .vote-meta span {{
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .vote-meta small {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .ok {{ color: var(--ok); }}
    .bad {{ color: var(--bad); }}
    .segments {{
      display: flex;
      gap: 2px;
      align-items: center;
      min-width: 0;
      overflow-x: auto;
      padding: 2px 0;
    }}
    .segment {{
      display: inline-flex;
      width: 13px;
      height: 22px;
      flex: 0 0 13px;
      align-items: center;
      justify-content: center;
      border: 1px solid rgb(0 0 0 / 0.16);
      border-radius: 3px;
      color: white;
      font-size: 8px;
      font-weight: 700;
      text-shadow: 0 1px 1px rgb(0 0 0 / 0.28);
    }}
    .segment.first {{
      outline: 2px solid rgb(0 0 0 / 0.18);
      outline-offset: 1px;
    }}
    .note {{
      margin-top: 16px;
      font-size: 13px;
      color: var(--muted);
    }}
    @media (max-width: 780px) {{
      main {{ padding: 20px 14px 32px; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .vote-row {{ grid-template-columns: 1fr; align-items: start; }}
      .vote-meta {{ grid-template-columns: 26px 32px minmax(140px, 1fr); }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Prefix-Q Segment Temperature Traces</h1>
      <p>Source: <code>{html.escape(source)}</code>. Each row is one vote. Each block is one generated segment; green is lower temperature, red is higher temperature. The outlined first block is the forced first segment temperature.</p>
    </header>
    <section class="summary">
      <div class="metric"><b>{int(data.get("n_prompts", 0))}</b><span>prompts in source file</span></div>
      <div class="metric"><b>{float(data.get("majority_accuracy", 0.0)):.3f}</b><span>majority accuracy</span></div>
      <div class="metric"><b>{float(data.get("individual_accuracy", 0.0)):.3f}</b><span>individual accuracy</span></div>
      <div class="metric"><b>{len(selected)}</b><span>selected trajectories</span></div>
    </section>
    <section class="legend">
      {_legend(all_temps)}
    </section>
    {cards}
    <p class="note">The JSON stores temperature decisions and final answers, but not the full generated text for each segment. Tooltips on blocks show segment index, stage, temperature, phi, selected Q, and margin when available.</p>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/eval200_online_20260622_155253/q_selector_seed42.json",
    )
    parser.add_argument(
        "--output",
        default="results/eval200_online_20260622_155253/q_selector_seed42_temperature_traces.html",
    )
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    selected = _pick_predictions(data.get("predictions", []), args.n)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_html(data, selected, str(input_path)),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output_path),
        "selected_problem_ids": [item.get("problem_id") for item in selected],
        "selected_count": len(selected),
    }, indent=2))


if __name__ == "__main__":
    main()
