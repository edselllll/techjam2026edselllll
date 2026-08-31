from __future__ import annotations

import argparse
import json
import random
import uuid
from collections import defaultdict
from pathlib import Path

from starter.agent import Agent, _terms, _clean_text
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


def analyze_overrides(records: list[dict]) -> dict:
    """Analyze how quickly and where the agent recovers after an intent override."""
    by_session = defaultdict(list)

    for record in records:
        if record["scenario_type"] == "intent_override":
            by_session[record["sample_id"]].append(record)

    sessions = []

    for sample_id, turns in by_session.items():
        turns = sorted(turns, key=lambda r: r["turn"])
        override_turn = next((r["override_turn"] for r in turns if r.get("override_turn") is not None), None)

        if override_turn is None:
            continue

        post_override = [r for r in turns if r["turn"] >= override_turn]
        override_record = next((r for r in post_override if r["turn"] == override_turn), None)
        first_hit = next((r["turn"] for r in post_override if r["top10_rank"] is not None), None)
        recovery_delay = first_hit - override_turn if first_hit is not None else None

        sessions.append({
            "sample_id": sample_id,
            "override_turn": override_turn,
            "first_top10_turn": first_hit,
            "recovery_delay": recovery_delay,
            "override_bm25_rank": override_record["bm25_rank"] if override_record else None,
            "override_faiss_rank": override_record["faiss_rank"] if override_record else None,
            "override_candidate_rank": override_record["candidate_rank"] if override_record else None,
            "override_rerank50_rank": override_record["rerank50_rank"] if override_record else None,
            "override_rerank20_rank": override_record["rerank20_rank"] if override_record else None,
            "override_top10_rank": override_record["top10_rank"] if override_record else None,
            "override_active_slots": override_record["active_slots"] if override_record else {},
            "override_search_query": override_record["query"] if override_record else "",
            "override_slot_query": override_record["slot_query"] if override_record else "",
            "override_rerank_query": override_record["rerank_query"] if override_record else "",
            "override_user_message": override_record["user_message"] if override_record else "",
        })

    total = len(sessions)
    successful = [s for s in sessions if s["recovery_delay"] is not None]

    if not total:
        return {"sessions": 0, "session_details": []}

    def rate(condition) -> float:
        return round(sum(condition(s) for s in sessions) / total, 4)

    return {
        "sessions": total,
        "immediate_recovery_rate": rate(lambda s: s["recovery_delay"] == 0),
        "within_1_turn_rate": rate(lambda s: s["recovery_delay"] is not None and s["recovery_delay"] <= 1),
        "within_2_turn_rate": rate(lambda s: s["recovery_delay"] is not None and s["recovery_delay"] <= 2),
        "eventual_recovery_rate": round(len(successful) / total, 4),
        "mean_recovery_delay_successes": round(
            sum(s["recovery_delay"] for s in successful) / len(successful), 3
        ) if successful else None,
        "target_in_bm25_on_override": rate(lambda s: s["override_bm25_rank"] is not None),
        "target_in_faiss_on_override": rate(lambda s: s["override_faiss_rank"] is not None),
        "target_in_candidate_pool_on_override": rate(lambda s: s["override_candidate_rank"] is not None),
        "target_in_rerank50_on_override": rate(lambda s: s["override_rerank50_rank"] is not None),
        "target_in_rerank20_on_override": rate(lambda s: s["override_rerank20_rank"] is not None),
        "target_in_top10_on_override": rate(lambda s: s["override_top10_rank"] is not None),
        "session_details": sessions,
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
        effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}

        disclosed = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        override_turn = None

        user_message = initial_message(
            effective_sample,
            coarse_category(categories.get(target, [])),
            disclosed,
        )

        for turn in range(1, MAX_TURNS + 1):
            active_slots, _, _ = agent.state_tracker.update_state(user_message, turn)
            intent = agent._detect_intent(user_message, active_slots)

            category_list = active_slots.get("category", [])
            category = category_list[0] if category_list else ""
            slot_query = agent.state_tracker.build_search_query(category)

            if slot_query and slot_query.lower() not in user_message.lower():
                search_query = f"{user_message} {slot_query}".strip()
            else:
                search_query = user_message

            bm25_results = agent._search_bm25(search_query, top_k=retrieval_k)
            faiss_results = agent.vector_indexer.search(search_query, top_k=retrieval_k)

            # IMPORTANT: match current agent weights.
            weights = (2.5, 0.8) if intent == "Buying" else (1.4, 1.0)

            fused_all = agent._combmnz_fusion(
                bm25_results,
                faiss_results,
                weights=weights,
            )
            fused_results = fused_all[:fusion_k]

            # IMPORTANT: match current widened candidate pool.
            candidates = agent._candidate_union(
                fused_all,
                bm25_results,
                faiss_results,
                fusion_k=100,
                bm25_k=150,
                faiss_k=75,
            )

            rerank_query = slot_query if len(_terms(slot_query)) >= 2 else search_query

            reranked = agent._rerank_candidates(
                rerank_query,
                candidates,
                bm25_results,
                faiss_results,
                intent=intent,
            )

            reranked_50 = reranked[:50]
            reranked_20 = reranked[:20]
            top10 = reranked[:TOP_K]

            # Ground truth comparison happens only AFTER all rankings are frozen.
            bm25_rank = rank_of(target, bm25_results)
            faiss_rank = rank_of(target, faiss_results)
            fusion_rank = rank_of(target, fused_results)
            candidate_rank = rank_of(target, candidates)
            rerank50_rank = rank_of(target, reranked_50)
            rerank20_rank = rank_of(target, reranked_20)
            top10_rank = rank_of(target, top10)

            records.append({
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "turn": turn,
                "intent": intent,
                "target": target,
                "user_message": user_message,
                "query": search_query,
                "slot_query": slot_query,
                "rerank_query": rerank_query,
                "active_slots": {key: list(values) for key, values in active_slots.items()},
                "override_turn": override_turn,
                "is_post_override": override_applied,
                "bm25_rank": bm25_rank,
                "faiss_rank": faiss_rank,
                "fusion_rank": fusion_rank,
                "candidate_rank": candidate_rank,
                "rerank50_rank": rerank50_rank,
                "rerank20_rank": rerank20_rank,
                "top10_rank": top10_rank,
            })

            if override_applied and top10_rank is not None:
                break

            if turn == MAX_TURNS:
                break

            ask_attr = agent.state_tracker.get_next_ask_attribute()
            override = effective_sample.get("behavior", {}).get("override") or {}

            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                override_turn = turn + 1

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

    grouped = defaultdict(list)
    by_session = defaultdict(list)

    for record in records:
        grouped[record["scenario_type"]].append(record)
        by_session[record["sample_id"]].append(record)

    sessions = []

    for sample_id, turns in by_session.items():
        def best_rank(field: str) -> int | None:
            ranks = [r[field] for r in turns if r[field] is not None]
            return min(ranks) if ranks else None

        bm25_best = best_rank("bm25_rank")
        faiss_best = best_rank("faiss_rank")
        fusion_best = best_rank("fusion_rank")
        candidate_best = best_rank("candidate_rank")
        rerank50_best = best_rank("rerank50_rank")
        rerank20_best = best_rank("rerank20_rank")
        top10_best = best_rank("top10_rank")

        sessions.append({
            "sample_id": sample_id,
            "scenario_type": turns[0]["scenario_type"],
            "bm25_best_rank": bm25_best,
            "faiss_best_rank": faiss_best,
            "union_hit": bm25_best is not None or faiss_best is not None,
            "fusion_best_rank": fusion_best,
            "candidate_best_rank": candidate_best,
            "rerank50_best_rank": rerank50_best,
            "rerank20_best_rank": rerank20_best,
            "top10_best_rank": top10_best,
        })

    total_sessions = len(sessions)

    session_summary = {
        "sessions": total_sessions,
        "bm25_hit_at_250": round(sum(s["bm25_best_rank"] is not None for s in sessions) / total_sessions, 4),
        "faiss_hit_at_250": round(sum(s["faiss_best_rank"] is not None for s in sessions) / total_sessions, 4),
        "union_hit_at_250": round(sum(s["union_hit"] for s in sessions) / total_sessions, 4),
        "fusion_hit_at_50": round(sum(s["fusion_best_rank"] is not None for s in sessions) / total_sessions, 4),
        "candidate_pool_hit": round(sum(s["candidate_best_rank"] is not None for s in sessions) / total_sessions, 4),
        "reranked_hit_at_50": round(sum(s["rerank50_best_rank"] is not None for s in sessions) / total_sessions, 4),
        "reranked_hit_at_20": round(sum(s["rerank20_best_rank"] is not None for s in sessions) / total_sessions, 4),
        "reranked_hit_at_10": round(sum(s["top10_best_rank"] is not None for s in sessions) / total_sessions, 4),
    }

    failure_analysis = {
        "retrieval_miss": 0,
        "retrieved_but_not_preserved": 0,
        "preserved_but_below_rerank50": 0,
        "rerank50_but_below_20": 0,
        "rerank20_but_below_10": 0,
        "top10_success": 0,
    }

    for s in sessions:
        retrieved = s["union_hit"]
        preserved = s["candidate_best_rank"] is not None
        r50 = s["rerank50_best_rank"] is not None
        r20 = s["rerank20_best_rank"] is not None
        r10 = s["top10_best_rank"] is not None

        if not retrieved:
            failure_analysis["retrieval_miss"] += 1
        elif not preserved:
            failure_analysis["retrieved_but_not_preserved"] += 1
        elif not r50:
            failure_analysis["preserved_but_below_rerank50"] += 1
        elif not r20:
            failure_analysis["rerank50_but_below_20"] += 1
        elif not r10:
            failure_analysis["rerank20_but_below_10"] += 1
        else:
            failure_analysis["top10_success"] += 1

    override_analysis = analyze_overrides(records)

    return {
        "config": {
            "retrieval_k": retrieval_k,
            "fusion_k": fusion_k,
            "top_k": TOP_K,
        },
        "session_summary": session_summary,
        "failure_analysis": failure_analysis,
        "override_analysis": override_analysis,
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
    print("\n=== Intent Override Diagnostics ===")
    print(json.dumps({
        key: value
        for key, value in result["override_analysis"].items()
        if key != "session_details"
    }, indent=2))

    print(f"\nFull diagnostics saved to: {args.output}")


    print("\n=== Failed Override Sessions ===")

    failed = [
        s for s in result["override_analysis"]["session_details"]
        if s["recovery_delay"] is None
    ]

    for s in failed:
        print(f"\n--- {s['sample_id']} ---")
        print(f"Override turn: {s['override_turn']}")
        print(f"User message: {s['override_user_message']}")
        print(f"Active slots: {json.dumps(s['override_active_slots'], indent=2)}")
        print(f"Search query: {s['override_search_query']}")
        print(f"Slot query: {s['override_slot_query']}")
        print(f"Rerank query: {s['override_rerank_query']}")
        print(
            f"Ranks: BM25={s['override_bm25_rank']} | "
            f"FAISS={s['override_faiss_rank']} | "
            f"Candidate={s['override_candidate_rank']} | "
            f"R50={s['override_rerank50_rank']} | "
            f"R20={s['override_rerank20_rank']} | "
            f"Top10={s['override_top10_rank']}"
        )

if __name__ == "__main__":
    main()