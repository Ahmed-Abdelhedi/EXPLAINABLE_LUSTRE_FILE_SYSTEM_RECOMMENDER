from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path


CONFIG_LABELS = {
    "A": "Scanner + LLM fallback only",
    "B": "Scanner + Explicit Resolver",
    "C": "B + Semantic Linker",
    "D": "C + LLM Fallback",
    "E": "Full V2 + Deterministic Verifier",
}


def safe_div(a, b):
    return a / b if b else 0.0



def get_expected_outputs(case):
    """
    Compatibility layer between legacy ablation datasets and V2 E2E benchmark.

    Legacy format:
        expected_outputs

    Current V2 benchmark format:
        gold_quantities
    """
    if "expected_outputs" in case:
        return case["expected_outputs"]

    if "gold_quantities" in case:
        return [
            {
                "field": q.get("field"),
                "value": q.get("value"),
                "unit": q.get("unit"),
                "role": q.get("role"),
            }
            for q in case["gold_quantities"]
        ]

    return []


def f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def same_value(a, b, tol=1e-6):
    try:
        af = float(a)
        bf = float(b)
        return abs(af - bf) <= tol * max(1.0, abs(bf))
    except (TypeError, ValueError):
        return a == b


def exact_item(gold, pred):
    return (
        gold["field"] == pred["field"]
        and same_value(gold["value"], pred["value"])
        and gold["unit"] == pred["unit"]
    )


def exact_multiset(gold, pred):
    if len(gold) != len(pred):
        return False

    used = set()

    for gold_item in gold:
        found = False

        for index, pred_item in enumerate(pred):
            if index in used:
                continue

            if exact_item(gold_item, pred_item):
                used.add(index)
                found = True
                break

        if not found:
            return False

    return True


def greedy_field_pairs(gold, pred):
    used = set()
    pairs = []

    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(pred):
            if pred_index in used:
                continue

            if gold_item["field"] == pred_item["field"]:
                used.add(pred_index)
                pairs.append(
                    (gold_index, pred_index)
                )
                break

    return pairs


def warmup_ollama(host, model):
    """
    Warm the Ollama model before measured benchmark messages.

    This is intentionally outside the timed benchmark so configurations
    that use the LLM are compared on inference/runtime cost rather than
    one-time model loading.
    """
    try:
        from ollama import Client

        client = Client(host=host)

        client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return JSON only: "
                        '{"status":"ok"}'
                    ),
                }
            ],
            format="json",
            options={
                "temperature": 0,
                "num_predict": 8,
            },
            stream=False,
        )

        print(
            f"[WARMUP] Ollama model ready: {model}"
        )

    except Exception as exc:
        print(
            "[WARMUP WARNING] Ollama warmup failed: "
            f"{type(exc).__name__}: {exc}"
        )


def build_common_components(config, host, model):
    from requirement_extractor_v2.conversation_scope_resolver import (
        ConversationScopeResolver,
    )
    from requirement_extractor_v2.explicit_pattern_resolver import (
        ExplicitPatternResolver,
    )
    from requirement_extractor_v2.llm_fallback_extractor import (
        LLMFallbackExtractor,
    )
    from requirement_extractor_v2.quantity_scanner import (
        QuantityScanner,
    )

    components = {
        "scope": ConversationScopeResolver(),
        "scanner": QuantityScanner(),
        "explicit": None,
        "semantic": None,
        "llm": None,
        "pipeline": None,
    }

    if config in {"B", "C", "D"}:
        components["explicit"] = (
            ExplicitPatternResolver()
        )

    if config in {"C", "D"}:
        from requirement_extractor_v2.semantic_linker.runtime import (
            SemanticLinkerRuntime,
        )

        # V3.3 compatible runtime loading:
        # SemanticLinkerRuntime now loads hierarchical FIELD->ROLE
        # calibration internally from manifest.json/thresholds.json.
        # No legacy global confidence_threshold access is required here.
        components["semantic"] = (
            SemanticLinkerRuntime()
        )

    if config in {"A", "D"}:
        components["llm"] = (
            LLMFallbackExtractor(
                enabled=True,
                host=host,
                model=model,
            )
        )

    if config == "E":
        from requirement_extractor_v2.verified_pipeline import (
            VerifiedRequirementPipeline,
        )

        components["pipeline"] = (
            VerifiedRequirementPipeline()
        )

    return components


def contextual_contract():
    from requirement_extractor_v2.models import (
        ParamName,
        QuantityDimension,
        SemanticRole,
    )

    role_by_field = {
        ParamName.requested_usable_capacity_tib:
            SemanticRole.TARGET,

        ParamName.client_count:
            SemanticRole.TOTAL_COUNT,

        ParamName.average_file_size_gb:
            SemanticRole.AVERAGE_VALUE,

        ParamName.max_file_size_gb:
            SemanticRole.MAXIMUM_LIMIT,

        ParamName.total_file_count:
            SemanticRole.TOTAL_COUNT,

        ParamName.read_write_ratio:
            SemanticRole.RATIO_COMPONENT,

        ParamName.target_read_gbps:
            SemanticRole.TARGET,

        ParamName.target_write_gbps:
            SemanticRole.TARGET,

        ParamName.max_budget_usd:
            SemanticRole.MAXIMUM_LIMIT,

        ParamName.max_power_w:
            SemanticRole.MAXIMUM_LIMIT,

        ParamName.annual_growth_percent:
            SemanticRole.GROWTH_RATE,
    }

    dimension_by_field = {
        ParamName.requested_usable_capacity_tib:
            QuantityDimension.CAPACITY,

        ParamName.client_count:
            QuantityDimension.UNKNOWN,

        ParamName.average_file_size_gb:
            QuantityDimension.FILE_SIZE,

        ParamName.max_file_size_gb:
            QuantityDimension.FILE_SIZE,

        ParamName.total_file_count:
            QuantityDimension.UNKNOWN,

        ParamName.read_write_ratio:
            QuantityDimension.PERCENT,

        ParamName.target_read_gbps:
            QuantityDimension.THROUGHPUT,

        ParamName.target_write_gbps:
            QuantityDimension.THROUGHPUT,

        ParamName.max_budget_usd:
            QuantityDimension.MONEY,

        ParamName.max_power_w:
            QuantityDimension.POWER,

        ParamName.annual_growth_percent:
            QuantityDimension.PERCENT,
    }

    return role_by_field, dimension_by_field


def apply_context_to_quantities(
    quantities,
    target_field,
    inherited_unit,
):
    if (
        target_field is None
        or inherited_unit is None
    ):
        return quantities

    _, dimension_by_field = contextual_contract()

    expected_dimension = (
        dimension_by_field.get(
            target_field
        )
    )

    if expected_dimension is None:
        return quantities

    updated = []

    for quantity in quantities:
        if quantity.unit is not None:
            updated.append(quantity)
            continue

        updated.append(
            replace(
                quantity,
                unit=inherited_unit,
                dimension=expected_dimension,
            )
        )

    return updated


def contextual_links(
    quantities,
    target_field,
):
    from requirement_extractor_v2.models import (
        SemanticLink,
    )

    role_by_field, _ = (
        contextual_contract()
    )

    role = role_by_field.get(
        target_field
    )

    if role is None:
        return []

    return [
        SemanticLink(
            quantity_id=quantity.id,
            field=target_field,
            role=role,
            evidence=quantity.raw,
            resolver="conversation_scope",
        )
        for quantity in quantities
    ]


def normalize_candidate_output(
    quantity,
    link,
):
    """
    Normalize A-D candidate outputs without applying final verification.

    If normalization fails, preserve the raw candidate instead of silently
    dropping it. This prevents ablations without the verifier from receiving
    an artificial safety advantage.
    """
    from requirement_extractor_v2.models import (
        ParamName,
    )
    from requirement_extractor_v2.unit_normalizer import (
        normalize_unit_value,
    )

    if link.field is None:
        return None

    normalization_error = None

    try:
        if (
            link.field
            == ParamName.read_write_ratio
        ):
            value = quantity.value
            unit = "%"
        else:
            value, unit = (
                normalize_unit_value(
                    field=link.field,
                    value=quantity.value,
                    unit=quantity.unit,
                )
            )

    except Exception as exc:
        value = quantity.value
        unit = quantity.unit
        normalization_error = (
            f"{type(exc).__name__}: {exc}"
        )

    return {
        "field": link.field.value,
        "value": value,
        "unit": unit,
        "role": link.role.value,
        "resolver": link.resolver,
        "normalization_error":
            normalization_error,
    }


def run_nonverified_config(
    *,
    config,
    case,
    components,
):
    from requirement_extractor_v2.models import (
        ParamName,
        ScopeIntent,
    )

    text = case["text"]

    previous_field = (
        None
        if case.get(
            "previous_question_field"
        ) is None
        else ParamName(
            case[
                "previous_question_field"
            ]
        )
    )

    scope = components[
        "scope"
    ].resolve(
        user_text=text,
        previous_question_field=
            previous_field,
        requested_unit=
            case.get(
                "requested_unit"
            ),
        previous_question=
            case.get(
                "previous_question"
            ),
    )

    # -------------------------------------------------------------
    # Common scope layer
    # -------------------------------------------------------------

    if (
        scope.intent
        == ScopeIntent.OUT_OF_SCOPE
    ):
        return {
            "scope":
                scope.intent.value,

            "accepted_outputs": [],

            "routes": [],

            "llm_calls": 0,
        }

    scanner = components[
        "scanner"
    ]

    quantities = scanner.scan(
        text
    )

    # -------------------------------------------------------------
    # Contextual short answer:
    # Scope is held constant across all five configurations.
    # -------------------------------------------------------------

    if (
        scope.intent
        == ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
    ):
        quantities = (
            apply_context_to_quantities(
                quantities=quantities,
                target_field=
                    scope.target_field,
                inherited_unit=
                    scope.inherited_unit,
            )
        )

        links = contextual_links(
            quantities=quantities,
            target_field=
                scope.target_field,
        )

        quantity_by_id = {
            q.id: q
            for q in quantities
        }

        outputs = [
            normalize_candidate_output(
                quantity_by_id[
                    link.quantity_id
                ],
                link,
            )
            for link in links
            if (
                link.quantity_id
                in quantity_by_id
            )
        ]

        return {
            "scope":
                scope.intent.value,

            "accepted_outputs": [
                item
                for item in outputs
                if item is not None
            ],

            "routes": [
                "conversation_scope"
                for _ in links
            ],

            "llm_calls": 0,
        }

    # NEW_REQUIREMENT / CORRECTION
    quantity_by_id = {
        q.id: q
        for q in quantities
    }

    if not quantities:
        return {
            "scope":
                scope.intent.value,
            "accepted_outputs": [],
            "routes": [],
            "llm_calls": 0,
        }

    links = []
    routes = []
    llm_before = 0
    llm_after = 0

    # -------------------------------------------------------------
    # A — Scanner + LLM only
    # -------------------------------------------------------------

    if config == "A":
        llm = components["llm"]
        llm_before = llm.call_count

        for quantity in quantities:
            link = (
                llm.resolve_quantity(
                    user_text=text,
                    quantity=quantity,
                    previous_question=None,
                )
            )

            if link is None:
                routes.append(
                    "unresolved"
                )
                continue

            links.append(link)
            routes.append(
                "llm_fallback"
            )

        llm_after = llm.call_count

    # -------------------------------------------------------------
    # B/C/D — Explicit first
    # -------------------------------------------------------------

    else:
        explicit = components[
            "explicit"
        ]

        explicit_result = (
            explicit.resolve(
                text,
                quantities,
            )
        )

        links.extend(
            explicit_result.links
        )

        explicit_ids = {
            link.quantity_id
            for link
            in explicit_result.links
        }

        route_by_qid = {
            qid: "explicit_pattern"
            for qid
            in explicit_ids
        }

        unresolved_ids = list(
            explicit_result
            .unresolved_quantity_ids
        )

        # B ends here.
        if config == "B":
            for qid in unresolved_ids:
                route_by_qid[
                    qid
                ] = "unresolved"

        # C / D add Semantic Linker.
        else:
            semantic = components[
                "semantic"
            ]

            llm = components.get(
                "llm"
            )

            if llm is not None:
                llm_before = (
                    llm.call_count
                )

            for qid in unresolved_ids:
                quantity = (
                    quantity_by_id[
                        qid
                    ]
                )

                prediction = (
                    semantic.predict(
                        text=text,
                        quantity=quantity,
                        previous_question=None,
                    )
                )

                if (
                    prediction.accepted
                    and
                    prediction.link
                    is not None
                ):
                    links.append(
                        prediction.link
                    )

                    route_by_qid[
                        qid
                    ] = (
                        "semantic_linker_xlmr"
                    )

                    continue

                # C stops after semantic abstention.
                if config == "C":
                    route_by_qid[
                        qid
                    ] = "unresolved"
                    continue

                # D adds the LLM fallback.
                llm_link = (
                    llm.resolve_quantity(
                        user_text=text,
                        quantity=quantity,
                        previous_question=None,
                    )
                )

                if llm_link is None:
                    route_by_qid[
                        qid
                    ] = "unresolved"
                    continue

                links.append(
                    llm_link
                )

                route_by_qid[
                    qid
                ] = "llm_fallback"

            if llm is not None:
                llm_after = (
                    llm.call_count
                )

        routes = [
            route_by_qid.get(
                quantity.id,
                "unresolved",
            )
            for quantity in quantities
        ]

    outputs = []

    for link in links:
        quantity = (
            quantity_by_id.get(
                link.quantity_id
            )
        )

        if quantity is None:
            continue

        item = (
            normalize_candidate_output(
                quantity,
                link,
            )
        )

        if item is not None:
            outputs.append(
                item
            )

    return {
        "scope":
            scope.intent.value,

        "accepted_outputs":
            outputs,

        "routes":
            routes,

        "llm_calls":
            max(
                0,
                llm_after
                - llm_before,
            ),
    }


def run_full_config_e(
    *,
    case,
    components,
):
    from requirement_extractor_v2.models import (
        ParamName,
    )

    previous_field = (
        None
        if case.get(
            "previous_question_field"
        ) is None
        else ParamName(
            case[
                "previous_question_field"
            ]
        )
    )

    pipeline = components[
        "pipeline"
    ]

    llm = pipeline.cascade.llm_fallback

    before = getattr(
        llm,
        "call_count",
        0,
    )

    result = pipeline.process(
        text=case["text"],
        previous_question=
            case.get(
                "previous_question"
            ),
        previous_question_field=
            previous_field,
        requested_unit=
            case.get(
                "requested_unit"
            ),
    )

    after = getattr(
        llm,
        "call_count",
        before,
    )

    outputs = [
        {
            "field":
                decision.field.value,

            "value":
                decision.value,

            "unit":
                decision.unit,

            "role": (
                None
                if decision.role is None
                else decision.role.value
            ),

            "resolver":
                result.cascade
                .traces[
                    decision.quantity_id
                ]
                .final_resolver
                if (
                    decision.quantity_id
                    in result.cascade.traces
                )
                else None,

            "normalization_error":
                None,
        }
        for decision
        in result.verified
        if decision.field is not None
    ]

    routes = [
        trace.final_resolver
        or "unresolved"
        for trace
        in result.cascade.traces.values()
    ]

    return {
        "scope": (
            None
            if result.scope is None
            else result.scope.intent.value
        ),

        "accepted_outputs":
            outputs,

        "routes":
            routes,

        "llm_calls":
            max(
                0,
                after - before,
            ),

        "decisions": [
            decision.to_dict()
            for decision
            in result.decisions
        ],
    }


def evaluate(
    *,
    config,
    dataset_path,
    output_path,
    host,
    model,
):
    with Path(
        dataset_path
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        cases = [
            json.loads(line)
            for line in handle
            if line.strip()
        ]

    # Warmup only when the measured configuration can call the LLM.
    if config in {
        "A",
        "D",
        "E",
    }:
        warmup_ollama(
            host=host,
            model=model,
        )

    print()
    print("=" * 88)
    print(
        f"ABLATION {config} — "
        f"{CONFIG_LABELS[config]}"
    )
    print("=" * 88)

    components = (
        build_common_components(
            config=config,
            host=host,
            model=model,
        )
    )

    field_tp = 0
    field_fp = 0
    field_fn = 0

    exact_outputs = 0
    exact_value_correct = 0
    unit_correct = 0
    gold_outputs_total = 0

    complete_success = 0
    scope_correct = 0

    safety_negative_count = 0
    false_acceptance_count = 0

    ambiguity_count = 0
    ambiguity_safe_count = 0

    out_of_scope_count = 0
    out_of_scope_correct = 0

    llm_calls_total = 0

    latencies = []
    routes = Counter()

    per_category = defaultdict(
        lambda: {
            "n": 0,
            "complete": 0,
            "gold": 0,
            "exact": 0,
            "llm": 0,
            "latencies": [],
        }
    )

    details = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        start = time.perf_counter()

        if config == "E":
            run = run_full_config_e(
                case=case,
                components=components,
            )
        else:
            run = run_nonverified_config(
                config=config,
                case=case,
                components=components,
            )

        latency = (
            time.perf_counter()
            - start
        )

        latencies.append(
            latency
        )

        llm_calls = run[
            "llm_calls"
        ]

        llm_calls_total += (
            llm_calls
        )

        routes.update(
            run["routes"]
        )

        predicted = (
            run[
                "accepted_outputs"
            ]
        )

        gold = list(
            get_expected_outputs(case)
        )

        gold_outputs_total += (
            len(gold)
        )

        pairs = (
            greedy_field_pairs(
                gold,
                predicted,
            )
        )

        field_tp += len(
            pairs
        )

        field_fp += (
            len(predicted)
            - len(pairs)
        )

        field_fn += (
            len(gold)
            - len(pairs)
        )

        case_exact = 0

        for gold_index, pred_index in pairs:
            gold_item = (
                gold[
                    gold_index
                ]
            )

            pred_item = (
                predicted[
                    pred_index
                ]
            )

            if same_value(
                gold_item["value"],
                pred_item["value"],
            ):
                exact_value_correct += 1

            if (
                gold_item["unit"]
                == pred_item["unit"]
            ):
                unit_correct += 1

            if exact_item(
                gold_item,
                pred_item,
            ):
                exact_outputs += 1
                case_exact += 1

        actual_scope = (
            run["scope"]
        )

        expected_scope = (
            case[
                "expected_scope"
            ]
        )

        this_scope_correct = (
            actual_scope
            == expected_scope
        )

        scope_correct += int(
            this_scope_correct
        )

        safety = case.get(
            "safety",
            "normal",
        )

        if safety in {
            "ambiguity",
            "out_of_scope",
        }:
            safety_negative_count += 1

            if predicted:
                false_acceptance_count += 1

        if safety == "ambiguity":
            ambiguity_count += 1

            if not predicted:
                ambiguity_safe_count += 1

        if safety == "out_of_scope":
            out_of_scope_count += 1

            if (
                actual_scope
                == "OUT_OF_SCOPE"
                and
                not predicted
            ):
                out_of_scope_correct += 1

        complete = (
            exact_multiset(
                gold,
                predicted,
            )
            and
            this_scope_correct
        )

        complete_success += int(
            complete
        )

        category = per_category[
            case["category"]
        ]

        category["n"] += 1
        category["complete"] += int(
            complete
        )
        category["gold"] += len(
            gold
        )
        category["exact"] += (
            case_exact
        )
        category["llm"] += (
            llm_calls
        )
        category[
            "latencies"
        ].append(
            latency
        )

        detail = {
            "id":
                case["id"],

            "category":
                case["category"],

            "language":
                case["language"],

            "text":
                case["text"],

            "expected_scope":
                expected_scope,

            "actual_scope":
                actual_scope,

            "gold_outputs":
                gold,

            "accepted_outputs":
                predicted,

            "llm_calls":
                llm_calls,

            "routes":
                run["routes"],

            "latency_s":
                latency,

            "complete_success":
                complete,

            "safety":
                safety,
        }

        if "decisions" in run:
            detail[
                "verification_decisions"
            ] = run[
                "decisions"
            ]

        details.append(
            detail
        )

        print(
            f"[{index:03d}/{len(cases)}] "
            f"{case['id']} "
            f"{case['category']:<25} "
            f"llm={llm_calls:<2} "
            f"lat={latency:7.3f}s "
            f"{'PASS' if complete else 'FAIL'}"
        )

    precision = safe_div(
        field_tp,
        field_tp + field_fp,
    )

    recall = safe_div(
        field_tp,
        field_tp + field_fn,
    )

    category_metrics = {}

    for name, values in (
        per_category.items()
    ):
        category_metrics[
            name
        ] = {
            "n":
                values["n"],

            "complete_message_success":
                safe_div(
                    values["complete"],
                    values["n"],
                ),

            "exact_output_recall": (
                safe_div(
                    values["exact"],
                    values["gold"],
                )
                if values["gold"]
                else None
            ),

            "average_llm_calls_per_message":
                safe_div(
                    values["llm"],
                    values["n"],
                ),

            "mean_latency_s":
                statistics.mean(
                    values[
                        "latencies"
                    ]
                ),

            "p95_latency_s":
                percentile(
                    values[
                        "latencies"
                    ],
                    0.95,
                ),
        }

    metrics = {
        "config":
            config,

        "configuration":
            CONFIG_LABELS[
                config
            ],

        "n_messages":
            len(cases),

        "common_layers_held_constant": [
            "ConversationScopeResolver",
            "QuantityScanner",
            "unit normalization for scoring",
        ],

        "field_tp":
            field_tp,

        "field_fp":
            field_fp,

        "field_fn":
            field_fn,

        "field_precision":
            precision,

        "field_recall":
            recall,

        "field_f1":
            f1(
                precision,
                recall,
            ),

        "gold_outputs":
            gold_outputs_total,

        "exact_output_correct":
            exact_outputs,

        "exact_output_recall":
            safe_div(
                exact_outputs,
                gold_outputs_total,
            ),

        "exact_value_accuracy_on_correct_field":
            safe_div(
                exact_value_correct,
                field_tp,
            ),

        "unit_accuracy_on_correct_field":
            safe_div(
                unit_correct,
                field_tp,
            ),

        "complete_message_success":
            safe_div(
                complete_success,
                len(cases),
            ),

        "scope_accuracy":
            safe_div(
                scope_correct,
                len(cases),
            ),

        "ambiguity_cases":
            ambiguity_count,

        "ambiguity_safe_handling_rate":
            safe_div(
                ambiguity_safe_count,
                ambiguity_count,
            ),

        "out_of_scope_cases":
            out_of_scope_count,

        "correct_out_of_scope_rate":
            safe_div(
                out_of_scope_correct,
                out_of_scope_count,
            ),

        "safety_negative_cases":
            safety_negative_count,

        "false_automatic_acceptance_count":
            false_acceptance_count,

        "false_automatic_acceptance_rate":
            safe_div(
                false_acceptance_count,
                safety_negative_count,
            ),

        "total_llm_calls":
            llm_calls_total,

        "average_llm_calls_per_message":
            safe_div(
                llm_calls_total,
                len(cases),
            ),

        "mean_latency_s":
            statistics.mean(
                latencies
            ),

        "median_latency_s":
            statistics.median(
                latencies
            ),

        "p95_latency_s":
            percentile(
                latencies,
                0.95,
            ),

        "max_latency_s":
            max(
                latencies
            ),

        "route_counts":
            dict(routes),

        "per_category":
            category_metrics,
    }

    output = {
        "metrics":
            metrics,

        "details":
            details,

        "methodology_note": (
            "Ablation uses the fixed 96-message quantity-only E2E benchmark "
            "without modifying implementation after the first E2E run. "
            "ConversationScopeResolver and QuantityScanner are held constant "
            "because the current LLM fallback API consumes already-detected "
            "Quantity objects. Configurations A-D expose accepted candidate "
            "links after canonical unit normalization but without final "
            "DeterministicVerifier rejection. Configuration E uses the real "
            "VerifiedRequirementPipeline and exposes only VERIFIED outputs."
        ),
    }

    Path(
        output_path
    ).write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print(
        f"ABLATION {config} METRICS"
    )
    print("=" * 88)
    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    failures = [
        detail
        for detail in details
        if not detail[
            "complete_success"
        ]
    ]

    print()
    print(
        f"FAILURES: "
        f"{len(failures)} / "
        f"{len(cases)}"
    )

    # Explicitly release large model objects before process exit.
    del components
    gc.collect()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        choices=[
            "A",
            "B",
            "C",
            "D",
            "E",
        ],
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
    )

    parser.add_argument(
        "--host",
        default="http://localhost:11434",
    )

    args = parser.parse_args()

    evaluate(
        config=args.config,
        dataset_path=args.dataset,
        output_path=args.output,
        host=args.host,
        model=args.model,
    )
if __name__ == "__main__":
    main()