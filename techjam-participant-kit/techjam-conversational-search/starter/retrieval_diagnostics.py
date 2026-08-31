from __future__ import annotations

import argparse
import json
import random
import uuid
from collections import defaultdict
from pathlib import Path

from starter.agent import Agent
from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)


def rank_of(target: str, results) -> int | None:
    """Return 1-based target rank from (asin, score) pairs or ASIN strings."""
    for rank, item in enumerate(results, start=1):
        asin = item[0] if isinstance(item, tuple) else item
        if asin == target:
            return rank
    return None


def summarize(records: list[dict]) -> dict:
    total = len(records)
    if not total:
        return {}

    def rate(key: str) -> float:
        return round(sum(r[key] is not None for r in records) / total, 4)

    return {
        "turns_analyzed": total,
        "bm25_recall": rate("bm25_rank"),
        "faiss_recall": rate("faiss_rank"),
        "union_recall": round(
            sum(
                r["bm25_rank"] is not None or r["faiss_rank"] is not None
                for r in records
            ) / total,
            4,
        ),
        "fusion_recall": rate("fusion_rank"),
        "top10_recall": rate("top10_rank"),
    }


def evaluate_diagnostics(
    agent: Agent,
    samples: list[dict],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    retrieval_k: int = 250,
    fusion_k: int = 50,
) -> dict:
    records = []

    for sample in samples:
        session_id = f"diag_{uuid.uuid4().hex}"
        agent.reset(session_id, sample["user_profile"])

        target = str(sample["ground_truth"]["parent_asin"])

        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {
            **sample,
            "intent_card": intent_card,
            "behavior": behavior,
        }

        disclosed = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"

        user_message = initial_message(
            effective_sample,
            coarse_category(categories.get(target, [])),
            disclosed,
        )

        for turn in range(1, MAX_TURNS + 1):
            # ----------------------------------------------------------
            # Normal agent state/query construction.
            # Ground truth is NOT supplied to any of these operations.
            # ----------------------------------------------------------
            active_slots, _, _ = agent.state_tracker.update_state(
                user_message, turn
            )

            intent = agent._detect_intent(user_message, active_slots)

            category_list = active_slots.get("category", [])
            category = category_list[0] if category_list else ""
            slot_query = agent.state_tracker.build_search_query(category)

            if slot_query and slot_query.lower() not in user_message.lower():
                search_query = f"{user_message} {slot_query}".strip()
            else:
                search_query = user_message

            # ----------------------------------------------------------
            # Freeze independent retrieval outputs FIRST.
            # ----------------------------------------------------------
            bm25_results = agent._search_bm25(
                search_query, top_k=retrieval_k
            )
            faiss_results = agent.vector_indexer.search(
                search_query, top_k=retrieval_k
            )

            if intent == "Buying":
                weights = (2.5, 0.8)
            else:
                weights = (0.8, 1.8)

            fused_results = agent._combmnz_fusion(
                bm25_results,
                faiss_results,
                weights=weights,
            )[:fusion_k]

            top10 = fused_results[:TOP_K]

            # ----------------------------------------------------------
            # ONLY NOW compare the frozen outputs with ground truth.
            # Nothing below this point affects retrieval or ranking.
            # ----------------------------------------------------------
            bm25_rank = rank_of(target, bm25_results)
            faiss_rank = rank_of(target, faiss_results)
            fusion_rank = rank_of(target, fused_results)
            top10_rank = rank_of(target, top10)

            records.append({
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "turn": turn,
                "intent": intent,
                "query": search_query,
                "target": target,
                "bm25_rank": bm25_rank,
                "faiss_rank": faiss_rank,
                "fusion_rank": fusion_rank,
                "top10_rank": top10_rank,
            })

            # Match evaluator termination behavior.
            if override_applied and top10_rank is not None:
                break

            if turn == MAX_TURNS:
                break

            # Ask the next attribute exactly as the real agent would.
            ask_attr = agent.state_tracker.get_next_ask_attribute()

            override = effective_sample.get("behavior", {}).get("override") or {}

            if (
                not override_applied
                and turn + 1 == int(override.get("turn", 3))
            ):
                override_applied = True
                new_value = str(override.get("new_value", ""))

                if new_value:
                    disclosed.add(new_value)

                user_message = str(
                    override.get(
                        "message",
                        "Actually, please ignore my earlier preference.",
                    )
                )
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample,
                    ask_attr,
                    disclosed,
                    boundary_used,
                )

    # --------------------------------------------------------------
    # Overall and scenario summaries
    # --------------------------------------------------------------
    grouped = defaultdict(list)
    for record in records:
        grouped[record["scenario_type"]].append(record)

    # Analyze each session's BEST retrieval opportunity as well.
    by_session = defaultdict(list)
    for record in records:
        by_session[record["sample_id"]].append(record)

    sessions = []

    for sample_id, turns in by_session.items():
        def best_rank(field: str) -> int | None:
            ranks = [r[field] for r in turns if r[field] is not None]
            return min(ranks) if ranks else None

        bm25_best = best_rank("bm25_rank")
        faiss_best = best_rank("faiss_rank")
        fusion_best = best_rank("fusion_rank")
        top10_best = best_rank("top10_rank")

        sessions.append({
            "sample_id": sample_id,
            "scenario_type": turns[0]["scenario_type"],
            "bm25_best_rank": bm25_best,
            "faiss_best_rank": faiss_best,
            "union_hit": bm25_best is not None or faiss_best is not None,
            "fusion_best_rank": fusion_best,
            "top10_best_rank": top10_best,
        })

    total_sessions = len(sessions)

    session_summary = {
        "sessions": total_sessions,
        "bm25_hit_at_retrieval_k": round(
            sum(s["bm25_best_rank"] is not None for s in sessions)
            / total_sessions,
            4,
        ),
        "faiss_hit_at_retrieval_k": round(
            sum(s["faiss_best_rank"] is not None for s in sessions)
            / total_sessions,
            4,
        ),
        "union_hit_at_retrieval_k": round(
            sum(s["union_hit"] for s in sessions) / total_sessions,
            4,
        ),
        "fusion_hit_at_k": round(
            sum(s["fusion_best_rank"] is not None for s in sessions)
            / total_sessions,
            4,
        ),
        "top10_hit": round(
            sum(s["top10_best_rank"] is not None for s in sessions)
            / total_sessions,
            4,
        ),
    }

    # Most useful failure decomposition.
    failure_analysis = {
        "retrieval_miss": 0,
        "bm25_only": 0,
        "faiss_only": 0,
        "both_retrievers": 0,
        "retrieved_but_lost_before_fusion_k": 0,
        "fusion_k_but_not_top10": 0,
        "top10_success": 0,
    }

    for s in sessions:
        bm25 = s["bm25_best_rank"] is not None
        faiss = s["faiss_best_rank"] is not None
        fusion = s["fusion_best_rank"] is not None
        top10 = s["top10_best_rank"] is not None

        if not bm25 and not faiss:
            failure_analysis["retrieval_miss"] += 1
        else:
            if bm25 and faiss:
                failure_analysis["both_retrievers"] += 1
            elif bm25:
                failure_analysis["bm25_only"] += 1
            else:
                failure_analysis["faiss_only"] += 1

            if not fusion:
                failure_analysis["retrieved_but_lost_before_fusion_k"] += 1
            elif not top10:
                failure_analysis["fusion_k_but_not_top10"] += 1
            else:
                failure_analysis["top10_success"] += 1

    return {
        "config": {
            "retrieval_k": retrieval_k,
            "fusion_k": fusion_k,
            "top_k": TOP_K,
        },
        "session_summary": session_summary,
        "failure_analysis": failure_analysis,
        "turn_summary": summarize(records),
        "scenario_turn_summary": {
            name: summarize(group)
            for name, group in sorted(grouped.items())
        },
        "sessions": sessions,
        "turns": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline retrieval diagnostics for TechJam agent"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="retrieval_diagnostics.json")
    parser.add_argument("--retrieval-k", type=int, default=250)
    parser.add_argument("--fusion-k", type=int, default=50)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    _, categories, products = catalog_index(args.catalog)

    agent = Agent(args.catalog)

    result = evaluate_diagnostics(
        agent,
        samples,
        categories,
        products,
        retrieval_k=args.retrieval_k,
        fusion_k=args.fusion_k,
    )

    Path(args.output).write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== Retrieval Diagnostics ===")
    print(json.dumps({
        "config": result["config"],
        "session_summary": result["session_summary"],
        "failure_analysis": result["failure_analysis"],
        "scenario_turn_summary": result["scenario_turn_summary"],
    }, indent=2))

    print(f"\nFull diagnostics saved to: {args.output}")


if __name__ == "__main__":
    main()