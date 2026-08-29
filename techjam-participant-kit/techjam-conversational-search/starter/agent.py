from __future__ import annotations

import os
import re
import json
import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN", "").strip()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

from sentence_transformers import SentenceTransformer
from starter.indexer import CatalogIndexer  # Import modular FAISS indexer


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
BUYING_VERBS = {"buy", "purchase", "order", "need", "find", "get"}
BROWSING_SIGNALS = {"looking for", "exploring", "suggestions", "ideas", "recommend", "what should i", "something for", "gift", "options"}

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items() if v)
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    return str(value)


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
        documents_to_embed: List[str] = []
        
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                
                title = _text(product.get("title"))
                cats = _text(product.get("categories"))
                feats = _text(product.get("features"))
                dets = _text(product.get("details"))
                store = _text(product.get("store"))
                desc = _text(product.get("description"))

                self.parent_asins.append(asin)
                bm25_batch.append((asin, title, cats, feats, dets, store, desc))
                
                # Compose text document for dense vector embedding
                embed_text = f"{title}. Category: {cats}. Features: {feats}".strip()
                documents_to_embed.append(embed_text)

                if len(bm25_batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", bm25_batch)
                    bm25_batch.clear()

        if bm25_batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", bm25_batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Resets session context for a new evaluation run."""
        self._sessions[session_id] = {
            "history": [],
            "slots": {},
            "profile": user_profile or {}
        }

    def _detect_intent(self, user_message: str, active_slots: Dict[str, Any]) -> str:
        text = user_message.lower()
        detected_attributes = len(active_slots)

        if PRICE_PATTERN.search(text):
            detected_attributes += 1
        if SIZE_PATTERN.search(text):
            detected_attributes += 1
        if DEPARTMENT_PATTERN.search(text):
            detected_attributes += 1

        tokens = set(re.findall(r"\w+", text))
        if tokens & COLOR_VOCAB:
            detected_attributes += 1
        if tokens & MATERIAL_VOCAB:
            detected_attributes += 1

        has_browsing_phrases = any(phrase in text for phrase in BROWSING_SIGNALS)
        has_buying_verbs = any(verb in tokens for verb in BUYING_VERBS)

        if has_browsing_phrases and detected_attributes < 2:
            return "Browsing"
        if detected_attributes >= 2 or (has_buying_verbs and detected_attributes >= 1):
            return "Buying"

        return "Browsing"

    def _search_bm25(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Executes BM25 search returning (asin, raw_bm25_score) pairs."""
        terms = list(dict.fromkeys(_terms(query)))[:40]
        if not terms:
            return []
        expression = " OR ".join(f'"{t}"' for t in terms)
        
        # Select parent_asin and BM25 rank score
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? "
            "ORDER BY score LIMIT ?",
            (expression, top_k),
        ).fetchall()

        # SQLite FTS5 bm25() returns negative values (lower is better match),
        # so we negate it (-r[1]) so higher scores represent better matches.
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

        # 1. Normalize scores independently to [0, 1]
        norm_bm25 = self._min_max_scale(bm25_dict)
        norm_faiss = self._min_max_scale(faiss_dict)

        w_bm25, w_faiss = weights
        all_asins = set(bm25_dict.keys()) | set(faiss_dict.keys())
        combmnz_scores: Dict[str, float] = {}

        # 2. Compute CombSUM & multiply by match count (N_retrievers)
        for asin in all_asins:
            matches_count = 0
            comb_sum = 0.0

            if asin in norm_bm25:
                matches_count += 1
                comb_sum += w_bm25 * norm_bm25[asin]

            if asin in norm_faiss:
                matches_count += 1
                comb_sum += w_faiss * norm_faiss[asin]

            # CombMNZ = CombSUM * N_retrievers
            combmnz_scores[asin] = comb_sum * matches_count

        # 3. Sort candidates by final score descending
        sorted_candidates = sorted(
            combmnz_scores.items(), 
            key=lambda item: item[1], 
            reverse=True
        )
        return [asin for asin, _ in sorted_candidates]

    def _select_ask_attribute(self, user_message: str, active_slots: Dict[str, Any]) -> str | None:
        """Selects an unfulfilled attribute to ask the user to clarify."""
        text = user_message.lower()
        attribute_priority = ["color", "size", "material", "brand", "style", "use_case", "category"]
        
        for attr in attribute_priority:
            if attr not in text and attr not in active_slots:
                if attr in ALLOWED_ATTRIBUTES:
                    return attr
        return "other" if "other" in ALLOWED_ATTRIBUTES else None

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        session = self._sessions[session_id]
        session["history"].append(user_message)
        combined_query = " ".join(session["history"])
        
        # 1. Detect Intent Track using session slots dictionary
        intent_track = self._detect_intent(user_message, session["slots"])

        # 2. Multi-Route Candidate Retrieval (Top-100 each for candidate overlap)
        bm25_candidates = self._search_bm25(combined_query, top_k=100)
        faiss_candidates = self.vector_indexer.search(combined_query, top_k=100)

        # 3. Dynamic CombMNZ Weighting
        # Fusion weights = (BM25 weight, FAISS weight)
        if intent_track == "Buying":
            # Keyword precision prioritized
            fusion_weights = (2.5, 1.0) 
        else:
            # Semantic scenario similarity prioritized
            fusion_weights = (0.5, 2.0)

        # 4. Fuse Candidate Pools using CombMNZ
        fused_asins = self._combmnz_fusion(
            bm25_candidates, 
            faiss_candidates, 
            weights=fusion_weights
        )[:top_k]

        recommendations = [{"parent_asin": asin} for asin in fused_asins]
        ask_attr = self._select_ask_attribute(user_message, session["slots"])

        return {
            "message": f"Retrieved top matches using {intent_track} CombMNZ track.",
            "ask_attribute": ask_attr,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }