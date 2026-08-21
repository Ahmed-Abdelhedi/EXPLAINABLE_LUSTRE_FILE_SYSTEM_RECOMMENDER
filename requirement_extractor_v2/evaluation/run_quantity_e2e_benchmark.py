from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


def safe_div(a, b):
    return a / b if b else 0.0


def f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def same_value(a, b, tol=1e-6):
    try:
        af, bf = float(a), float(b)
        return abs(af - bf) <= tol * max(1.0, abs(bf))
    except (TypeError, ValueError):
        return a == b


def exact_item(g, p):
    return (
        g["field"] == p["field"]
        and same_value(g["value"], p["value"])
        and g["unit"] == p["unit"]
    )


def exact_multiset(gold, pred):
    if len(gold) != len(pred):
        return False
    used = set()
    for g in gold:
        found = False
        for i, p in enumerate(pred):
            if i in used:
                continue
            if exact_item(g, p):
                used.add(i)
                found = True
                break
        if not found:
            return False
    return True


def greedy_field_pairs(gold, pred):
    used = set()
    pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if pi in used:
                continue
            if g["field"] == p["field"]:
                used.add(pi)
                pairs.append((gi, pi))
                break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="qwen2.5-coder:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--disable-llm", action="store_true")
    args = ap.parse_args()

    # Configure Ollama before importing project modules.
    os.environ["ENABLE_LLM_FALLBACK"] = "false" if args.disable_llm else "true"
    os.environ["OLLAMA_MODEL"] = args.model
    os.environ["OLLAMA_HOST"] = args.host

    from requirement_extractor_v2.models import ParamName
    from requirement_extractor_v2.verified_pipeline import VerifiedRequirementPipeline

    with open(args.dataset, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    pipeline = VerifiedRequirementPipeline()

    field_tp = field_fp = field_fn = 0
    exact_outputs = 0
    value_correct = 0
    unit_correct = 0
    total_gold = 0
    complete_ok = 0
    scope_ok = 0
    ambiguity_total = ambiguity_ok = 0
    oos_total = oos_ok = 0
    safety_total = false_accept = 0
    llm_total = 0
    latencies = []
    routes = Counter()
    details = []
    per_cat = defaultdict(lambda: {
        "n": 0, "success": 0, "gold": 0, "exact": 0,
        "llm": 0, "lat": []
    })

    for idx, case in enumerate(cases, 1):
        prev_field = (
            None if case.get("previous_question_field") is None
            else ParamName(case["previous_question_field"])
        )

        llm = pipeline.cascade.llm_fallback
        before = getattr(llm, "call_count", 0)

        t0 = time.perf_counter()
        result = pipeline.process(
            case["text"],
            previous_question=case.get("previous_question"),
            previous_question_field=prev_field,
            requested_unit=case.get("requested_unit"),
        )
        latency = time.perf_counter() - t0
        latencies.append(latency)

        after = getattr(llm, "call_count", before)
        llm_calls = max(0, after - before)
        llm_total += llm_calls

        pred = [{
            "field": d.field.value,
            "value": d.value,
            "unit": d.unit,
            "role": None if d.role is None else d.role.value,
        } for d in result.verified if d.field is not None]

        gold = case["expected_outputs"]
        total_gold += len(gold)

        pairs = greedy_field_pairs(gold, pred)
        field_tp += len(pairs)
        field_fp += len(pred) - len(pairs)
        field_fn += len(gold) - len(pairs)

        case_exact = 0
        for gi, pi in pairs:
            g, p = gold[gi], pred[pi]
            value_correct += int(same_value(g["value"], p["value"]))
            unit_correct += int(g["unit"] == p["unit"])
            if exact_item(g, p):
                exact_outputs += 1
                case_exact += 1

        actual_scope = None if result.scope is None else result.scope.intent.value
        this_scope_ok = actual_scope == case["expected_scope"]
        scope_ok += int(this_scope_ok)

        safety = case.get("safety", "normal")
        this_ambiguity_ok = True
        this_oos_ok = True

        if safety == "ambiguity":
            ambiguity_total += 1
            has_safe_signal = any(
                d.status.value in ("AMBIGUOUS", "UNRESOLVED", "INVALID")
                for d in result.decisions
            )
            this_ambiguity_ok = len(pred) == 0 and has_safe_signal
            ambiguity_ok += int(this_ambiguity_ok)

        if safety == "out_of_scope":
            oos_total += 1
            this_oos_ok = (
                actual_scope == "OUT_OF_SCOPE"
                and len(pred) == 0
                and len(result.cascade.quantities) == 0
            )
            oos_ok += int(this_oos_ok)

        if safety in ("ambiguity", "out_of_scope"):
            safety_total += 1
            false_accept += int(bool(pred))

        complete = (
            exact_multiset(gold, pred)
            and this_scope_ok
            and this_ambiguity_ok
            and this_oos_ok
        )
        complete_ok += int(complete)

        for tr in result.cascade.traces.values():
            routes[tr.final_resolver or "unresolved"] += 1

        cat = per_cat[case["category"]]
        cat["n"] += 1
        cat["success"] += int(complete)
        cat["gold"] += len(gold)
        cat["exact"] += case_exact
        cat["llm"] += llm_calls
        cat["lat"].append(latency)

        details.append({
            "id": case["id"],
            "category": case["category"],
            "language": case["language"],
            "text": case["text"],
            "expected_scope": case["expected_scope"],
            "actual_scope": actual_scope,
            "gold_outputs": gold,
            "predicted_verified_outputs": pred,
            "decisions": [d.to_dict() for d in result.decisions],
            "llm_calls": llm_calls,
            "latency_s": latency,
            "complete_success": complete,
            "safety": safety,
        })

        print(
            f"[{idx:03d}/{len(cases)}] {case['id']} "
            f"{case['category']:<25} llm={llm_calls} "
            f"lat={latency:7.3f}s {'PASS' if complete else 'FAIL'}"
        )

    p = safe_div(field_tp, field_tp + field_fp)
    r = safe_div(field_tp, field_tp + field_fn)

    category_metrics = {}
    for name, s in per_cat.items():
        category_metrics[name] = {
            "n": s["n"],
            "complete_message_success": safe_div(s["success"], s["n"]),
            "exact_output_recall": (
                safe_div(s["exact"], s["gold"]) if s["gold"] else None
            ),
            "average_llm_calls_per_message": safe_div(s["llm"], s["n"]),
            "mean_latency_s": statistics.mean(s["lat"]),
            "p95_latency_s": percentile(s["lat"], 0.95),
        }

    metrics = {
        "benchmark": "quantity_e2e_holdout_v1",
        "n_messages": len(cases),
        "field_tp": field_tp,
        "field_fp": field_fp,
        "field_fn": field_fn,
        "field_precision": p,
        "field_recall": r,
        "field_f1": f1(p, r),
        "gold_outputs": total_gold,
        "exact_output_correct": exact_outputs,
        "exact_output_recall": safe_div(exact_outputs, total_gold),
        "exact_value_accuracy_on_correct_field": safe_div(value_correct, field_tp),
        "unit_accuracy_on_correct_field": safe_div(unit_correct, field_tp),
        "complete_message_success": safe_div(complete_ok, len(cases)),
        "scope_accuracy": safe_div(scope_ok, len(cases)),
        "ambiguity_cases": ambiguity_total,
        "correct_ambiguity_detection": safe_div(ambiguity_ok, ambiguity_total),
        "out_of_scope_cases": oos_total,
        "correct_out_of_scope_detection": safe_div(oos_ok, oos_total),
        "safety_negative_cases": safety_total,
        "false_automatic_acceptance_count": false_accept,
        "false_automatic_acceptance_rate": safe_div(false_accept, safety_total),
        "total_llm_calls": llm_total,
        "average_llm_calls_per_message": safe_div(llm_total, len(cases)),
        "mean_latency_s": statistics.mean(latencies),
        "median_latency_s": statistics.median(latencies),
        "p95_latency_s": percentile(latencies, 0.95),
        "max_latency_s": max(latencies),
        "route_counts": dict(routes),
        "per_category": category_metrics,
    }

    failures = [x for x in details if not x["complete_success"]]

    Path(args.output).write_text(
        json.dumps({
            "metrics": metrics,
            "details": details,
            "benchmark_note": (
                "First-run quantity-only end-to-end holdout. Preserve the first "
                "result unchanged. If implementation is modified after seeing "
                "these failures, later runs are regression results."
            ),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("QUANTITY-ONLY END-TO-END METRICS")
    print("=" * 88)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nFAILURES: {len(failures)} / {len(cases)}")

    if failures:
        print("=" * 88)
        for d in failures:
            print(
                f"\n{d['id']} | {d['category']} | {d['language']}\n"
                f"TEXT     : {d['text']}\n"
                f"SCOPE    : expected={d['expected_scope']} actual={d['actual_scope']}\n"
                f"GOLD     : {d['gold_outputs']}\n"
                f"PREDICTED: {d['predicted_verified_outputs']}\n"
                f"LLM CALLS: {d['llm_calls']}\n"
                f"LATENCY  : {d['latency_s']:.3f}s"
            )


if __name__ == "__main__":
    main()