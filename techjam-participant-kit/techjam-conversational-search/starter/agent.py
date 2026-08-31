import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

from sentence_transformers import SentenceTransformer
from starter.indexer import CatalogIndexer  # Import modular FAISS indexer
from starter.indexer import _clean_text  # Import text cleaning utility
from starter.state_tracker_rulebased import DialogueStateTracker  # Import rule-based dialogue state tracker

# Allowed attributes defined by the evaluator
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", 
    "brand", "budget", "feature", "use_case", "other"
}

# Regex Signals for Zero-Latency Intent Classification
PRICE_PATTERN = re.compile(r"(\$|under|less than|below|over|budget|around)\s*\d+", re.I)
SIZE_PATTERN = re.compile(r"\b(size|sz)\s*(\d+|xs|s|m|l|xl|xxl)\b", re.I)
DEPARTMENT_PATTERN = re.compile(r"\b(men'?s|women'?s|kids'|unisex|baby)\b", re.I)

COLOR_VOCAB = {"black", "white", "blue", "red", "green", "yellow", "brown", "pink", "purple", "grey", "gray", "silver", "gold"}
MATERIAL_VOCAB = {"leather", "cotton", "wool", "polyester", "silk", "nylon", "spandex", "denim", "canvas", "mesh"}
BUYING_VERBS = {"buy", "purchase", "order", "need", "find", "get", "checkout", "add"}
BROWSING_SIGNALS = {"looking for", "exploring", "suggestions", "ideas", "recommend", "what should i", "something for", "gift", "options"}

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    def __init__(
        self, 
        catalog_path: str | Path = "data/catalog.jsonl",
        embedding_model: str = "all-MiniLM-L6-v2"
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self.parent_asins: List[str] = []
        self.products_by_asin: Dict[str, Dict[str, Any]] = {}
        self.state_tracker = DialogueStateTracker()

        # 1. Load Model Encoder
        self.encoder = SentenceTransformer(
            embedding_model, 
            token=hf_token if hf_token else None
        )

        # 2. Build Sparse BM25 Index in SQLite FTS5
        self._build_bm25_index()
        

        # 3. Instantiate Modular FAISS Indexer
        self.vector_indexer = CatalogIndexer(
            catalog_path=self.catalog_path,
            encoder=self.encoder
        )

    def _build_bm25_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        bm25_batch: List[Tuple[str, str, str, str, str, str, str]] = []
        # documents_to_embed: List[str] = []
        
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                self.products_by_asin[asin] = product
                
                title = _clean_text(product.get("title"))
                cats = _clean_text(product.get("categories"))
                feats = _clean_text(product.get("features"))
                details = _clean_text(product.get("details"))
                description = _clean_text(product.get("description"))
                store = _clean_text(product.get("store"))

                embed_text = (
                    f"Product: {title}. "
                    f"Category: {cats}. "
                    f"Brand: {store}. "
                    f"Features: {feats}. "
                    f"Details: {details}. "
                    f"Description: {description}."
                ).strip()

                self.parent_asins.append(asin)
                bm25_batch.append((asin, title, cats, feats, details, store, description))
                
                # embed_text = f"{title}. Category: {cats}. Features: {feats}".strip()
                # documents_to_embed.append(embed_text)

                if len(bm25_batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", bm25_batch)
                    bm25_batch.clear()

        if bm25_batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", bm25_batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: Optional[dict] = None) -> None:
        """Resets agent state per session and loads user profile preferences."""
        self.state_tracker.reset(user_profile)
        
    def _detect_intent(self, user_message: str, active_slots: Dict[str, Any]) -> str:
        """
        Dynamically detects user turn intent ('Buying' vs 'Browsing').
        Evaluates regex rules, keyword vocabs, and slot-state density on every turn.
        """
        text = user_message.lower()

        # Count total active constraints currently present in state
        filled_slots_count = sum(1 for k, v in active_slots.items() if v)

        # Detect specific regex attributes in the current message
        message_attribute_signals = 0
        if PRICE_PATTERN.search(text):
            message_attribute_signals += 1
        if SIZE_PATTERN.search(text):
            message_attribute_signals += 1
        if DEPARTMENT_PATTERN.search(text):
            message_attribute_signals += 1

        tokens = set(re.findall(r"\w+", text))
        if tokens & COLOR_VOCAB:
            message_attribute_signals += 1
        if tokens & MATERIAL_VOCAB:
            message_attribute_signals += 1

        has_browsing_phrases = any(phrase in text for phrase in BROWSING_SIGNALS)
        has_buying_verbs = any(verb in tokens for verb in BUYING_VERBS)

        # 1. Override: Browsing phrases explicitly indicate discovery unless strong constraints exist
        if has_browsing_phrases and (filled_slots_count + message_attribute_signals) < 3:
            return "Browsing"

        # 2. Buying Criteria: Direct buying verbs or high cumulative attribute density (>=2 attributes)
        if (
            has_buying_verbs 
            or (filled_slots_count + message_attribute_signals) >= 2 
            or message_attribute_signals >= 2
        ):
            return "Buying"

        return "Browsing"

    def _search_bm25(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Executes BM25 search returning (asin, raw_bm25_score) pairs."""
        terms = list(dict.fromkeys(_terms(query)))[:40]
        if not terms:
            return []
        expression = " OR ".join(f'"{t}"' for t in terms)
        
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? "
            "ORDER BY score LIMIT ?",
            (expression, top_k),
        ).fetchall()

        return [(str(r[0]), -float(r[1])) for r in rows]

    def _min_max_scale(self, scores_dict: Dict[str, float]) -> Dict[str, float]:
        """Scales raw retrieval scores to [0, 1] range."""
        if not scores_dict:
            return {}
        vals = np.array(list(scores_dict.values()))
        min_v, max_v = vals.min(), vals.max()
        if max_v == min_v:
            return {k: 1.0 for k in scores_dict}
        return {k: float((v - min_v) / (max_v - min_v + 1e-8)) for k, v in scores_dict.items()}

    def _combmnz_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        faiss_results: List[Tuple[str, float]],
        weights: Tuple[float, float] = (1.0, 1.0),
    ) -> List[str]:
        """Combines sparse and dense candidate pools using CombMNZ."""
        bm25_dict = dict(bm25_results)
        faiss_dict = dict(faiss_results)

        norm_bm25 = self._min_max_scale(bm25_dict)
        norm_faiss = self._min_max_scale(faiss_dict)

        w_bm25, w_faiss = weights
        all_asins = set(bm25_dict.keys()) | set(faiss_dict.keys())
        combmnz_scores: Dict[str, float] = {}

        for asin in all_asins:
            matches_count = 0
            comb_sum = 0.0

            if asin in norm_bm25:
                matches_count += 1
                comb_sum += w_bm25 * norm_bm25[asin]

            if asin in norm_faiss:
                matches_count += 1
                comb_sum += w_faiss * norm_faiss[asin]

            combmnz_scores[asin] = comb_sum * matches_count

        sorted_candidates = sorted(
            combmnz_scores.items(), 
            key=lambda item: item[1], 
            reverse=True
        )
        return [asin for asin, _ in sorted_candidates]

    def _candidate_union(self, fused: List[str], bm25_results: List[Tuple[str, float]], faiss_results: List[Tuple[str, float]], fusion_k: int = 50, bm25_k: int = 50, faiss_k: int = 25,) -> List[str]:
        """Preserves strong candidates from fusion and individual retrievers."""
        candidates, seen = [], set()

        sources = [
            fused[:fusion_k],
            [asin for asin, _ in bm25_results[:bm25_k]],
            [asin for asin, _ in faiss_results[:faiss_k]],
        ]

        for source in sources:
            for asin in source:
                if asin not in seen:
                    seen.add(asin)
                    candidates.append(asin)

        return candidates


    def _field_terms(self, product: dict, field: str) -> set[str]:
        return set(_terms(_clean_text(product.get(field))))


    def _rerank_candidates(
        self,
        query: str,
        candidates: List[str],
        bm25_results: List[Tuple[str, float]],
        faiss_results: List[Tuple[str, float]],
        intent: str,
    ) -> List[str]:
        """Reranks retrieved candidates using retrieval and constraint-coverage signals."""
        if not candidates:
            return []

        query_terms = set(_terms(query))
        bm25_scores = dict(bm25_results)
        faiss_scores = dict(faiss_results)

        norm_bm25 = self._min_max_scale(bm25_scores)
        norm_faiss = self._min_max_scale(faiss_scores)

        bm25_ranks = {asin: rank for rank, (asin, _) in enumerate(bm25_results, 1)}
        faiss_ranks = {asin: rank for rank, (asin, _) in enumerate(faiss_results, 1)}

        scored = []

        for asin in candidates:
            product = self.products_by_asin.get(asin, {})

            title_terms = self._field_terms(product, "title")
            category_terms = self._field_terms(product, "categories")
            feature_terms = self._field_terms(product, "features")
            detail_terms = self._field_terms(product, "details")
            description_terms = self._field_terms(product, "description")
            store_terms = self._field_terms(product, "store")

            all_terms = (
                title_terms | category_terms | feature_terms |
                detail_terms | description_terms | store_terms
            )

            if query_terms:
                overall_coverage = len(query_terms & all_terms) / len(query_terms)
                title_coverage = len(query_terms & title_terms) / len(query_terms)
                category_coverage = len(query_terms & category_terms) / len(query_terms)
                feature_coverage = len(query_terms & feature_terms) / len(query_terms)
            else:
                overall_coverage = title_coverage = category_coverage = feature_coverage = 0.0

            bm25_score = norm_bm25.get(asin, 0.0)
            dense_score = norm_faiss.get(asin, 0.0)

            # Smooth rank signals in [0, 1].
            bm25_rank_score = 1.0 / bm25_ranks[asin] if asin in bm25_ranks else 0.0
            dense_rank_score = 1.0 / faiss_ranks[asin] if asin in faiss_ranks else 0.0

            both_bonus = 1.0 if asin in bm25_scores and asin in faiss_scores else 0.0

            if intent == "Buying":
                score = (
                    2.20 * bm25_score + 0.65 * dense_score + 1.40 * overall_coverage + 0.70 * title_coverage + 0.45 * category_coverage + 0.60 * feature_coverage + 0.20 * bm25_rank_score + 0.05 * dense_rank_score + 0.15 * both_bonus)
            else:
                score = (
                    1.30 * bm25_score+ 1.00 * dense_score + 1.20 * overall_coverage + 0.55 * title_coverage + 0.35 * category_coverage + 0.55 * feature_coverage + 0.10 * bm25_rank_score + 0.10 * dense_rank_score + 0.15 * both_bonus)

            scored.append((asin, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [asin for asin, _ in scored]

    def _buying_pipeline(self, query: str, top_k: int, rerank_query: str = "") -> List[str]:
        candidate_pool_size = max(top_k * 5, 50)
        bm25_results = self._search_bm25(query, top_k=candidate_pool_size)
        faiss_results = self.vector_indexer.search(query, top_k=candidate_pool_size)

        fused = self._combmnz_fusion(bm25_results,faiss_results,weights=(2.5, 0.8),)

        candidates = self._candidate_union(fused,bm25_results,faiss_results,fusion_k=50,bm25_k=50,faiss_k=25,)

        return self._rerank_candidates(rerank_query or query,  candidates, bm25_results, faiss_results, intent="Buying",)[:top_k]

    def _browsing_pipeline(self, query: str, top_k: int, rerank_query: str = "",) -> List[str]:
        candidate_pool_size = max(top_k * 5, 50)
        bm25_results = self._search_bm25(query, top_k=candidate_pool_size)
        faiss_results = self.vector_indexer.search(query, top_k=candidate_pool_size)

        fused = self._combmnz_fusion(bm25_results,faiss_results,weights=(1.4, 1.0),)

        candidates = self._candidate_union(fused,bm25_results,faiss_results,fusion_k=50,bm25_k=50,faiss_k=25,)

        return self._rerank_candidates(rerank_query or query, candidates, bm25_results, faiss_results, intent="Browsing",)[:top_k]

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int = 10) -> Dict[str, Any]:
        """
        Handles dialogue turn cleanly:
        1. Updates slot state via state tracker.
        2. Re-detects turn intent (Buying vs Browsing) dynamically.
        3. Formulates structured search query.
        4. Routes query to either Buying or Browsing retrieval pipeline.
        5. Validates candidate parent ASINs and returns response dictionary.
        """
        # 1. Update slots via DialogueStateTracker
        active_slots, p_tokens, c_tokens = self.state_tracker.update_state(user_message, turn)

        # 2. Re-detect intent dynamically based on updated active_slots and turn message
        intent = self._detect_intent(user_message, active_slots)

        # 3. Formulate search query
        category_list = active_slots.get("category", [])
        category = category_list[0] if category_list else ""
        
        slot_query = self.state_tracker.build_search_query(category)

        # Combine user message and slot query safely
        if slot_query and slot_query.lower() not in user_message.lower():
            search_query = f"{user_message} {slot_query}".strip()
        else:
            search_query = user_message

        # 4. Route candidate retrieval based on detected turn intent
        rerank_query = slot_query if len(_terms(slot_query)) >= 2 else search_query
        if intent == "Buying":
            raw_recommendations = self._buying_pipeline(search_query, top_k=top_k * 5, rerank_query=rerank_query)
        else:
            raw_recommendations = self._browsing_pipeline(search_query, top_k=top_k * 5, rerank_query=rerank_query)

        # 5. Strict order-preserving deduplication & catalog validation
        seen = set()
        recommendations = []
        for asin in raw_recommendations:
            if asin not in seen and asin in self.parent_asins:
                seen.add(asin)
                recommendations.append(asin)
                if len(recommendations) == top_k:
                    break

        # 6. Select next unasked attribute for evaluator clarification loop
        ask_attr = self.state_tracker.get_next_ask_attribute()
        if ask_attr:
            message = f"Got it. What are your preferences for {ask_attr}?"
        else:
            message = "Got it. These are my best matches based on your preferences."

        # 7. Return payload expected by evaluator
        return {
            "message": message,
            "ask_attribute": ask_attr,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
            },
        }