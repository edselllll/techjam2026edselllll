from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional


ATTRIBUTE_ORDER = [
    "feature", "material", "color", "style",
    "use_case", "brand", "size", "budget"
]

MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|"
    r"fleece|canvas|denim|linen|acrylic|suede|rubber|mesh)\b", re.I
)

COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|"
    r"yellow|orange|silver|gold|navy|beige|tan|khaki|cream)\b", re.I
)

STYLE_RE = re.compile(
    r"\b(casual|formal|vintage|modern|slim fit|regular fit|relaxed fit|"
    r"athletic|boho|streetwear|classic|retro|oversized|fitted)\b", re.I
)

USE_CASE_RE = re.compile(
    r"\b(hiking|running|gym|winter|summer|outdoor|work|party|travel|"
    r"yoga|swimming|wedding|training|walking|camping|skiing)\b", re.I
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items() if v)
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    return str(value)


class ClarificationPolicy:
    """
    Candidate-aware proactive clarification policy.

    It answers two questions:
    1. Is the current search still too general?
    2. Which missing attribute would best divide the candidate pool?
    """

    def __init__(
        self,
        min_turn: int = 1,
        max_clarification_turn: int = 7,
        broad_pool_threshold: int = 150,
        min_attribute_coverage: float = 0.15,
    ) -> None:
        self.min_turn = min_turn
        self.max_clarification_turn = max_clarification_turn
        self.broad_pool_threshold = broad_pool_threshold
        self.min_attribute_coverage = min_attribute_coverage

    def _product_text(self, product: Dict[str, Any]) -> str:
        return " ".join([
            _text(product.get("title")),
            _text(product.get("categories")),
            _text(product.get("features")),
            _text(product.get("details")),
            _text(product.get("description")),
        ]).lower()

    def _extract_values(self, product: Dict[str, Any], attribute: str) -> List[str]:
        text = self._product_text(product)

        if attribute == "material":
            return list(dict.fromkeys(m.group(0).lower() for m in MATERIAL_RE.finditer(text)))

        if attribute == "color":
            return list(dict.fromkeys(m.group(0).lower() for m in COLOR_RE.finditer(text)))

        if attribute == "style":
            return list(dict.fromkeys(m.group(0).lower() for m in STYLE_RE.finditer(text)))

        if attribute == "use_case":
            return list(dict.fromkeys(m.group(0).lower() for m in USE_CASE_RE.finditer(text)))

        if attribute == "brand":
            store = _text(product.get("store")).lower().strip()
            return [store] if store else []

        return []

    def _attribute_information(
        self,
        products: List[Dict[str, Any]],
        attribute: str,
    ) -> float:
        """Normalized entropy × catalog coverage for an attribute."""
        values = []
        populated = 0

        for product in products:
            extracted = self._extract_values(product, attribute)
            if not extracted:
                continue

            populated += 1
            values.extend(set(extracted))

        if not products or not values:
            return 0.0

        coverage = populated / len(products)
        if coverage < self.min_attribute_coverage:
            return 0.0

        counts = Counter(values)
        total = sum(counts.values())

        entropy = -sum(
            (count / total) * math.log2(count / total)
            for count in counts.values()
        )

        max_entropy = math.log2(max(2, len(counts)))
        normalized_entropy = entropy / max_entropy if max_entropy else 0.0

        return normalized_entropy * coverage

    def is_over_general(
        self,
        turn: int,
        active_slots: Dict[str, List[str]],
        candidates: List[str],
        bm25_results,
        faiss_results,
    ) -> bool:
        """
        Detect candidate-pool overload using both state specificity and
        retrieval ambiguity.
        """
        if turn < self.min_turn or turn > self.max_clarification_turn:
            return False

        filled = sum(bool(values) for values in active_slots.values())

        # Already strongly constrained: don't keep interrogating the user.
        if filled >= 4:
            return False

        # Large surviving candidate space is the first overload signal.
        broad_pool = len(candidates) >= self.broad_pool_threshold

        # Check whether BM25 has a decisive winner.
        bm25_ambiguous = True
        if len(bm25_results) >= 2:
            first = bm25_results[0][1]
            second = bm25_results[1][1]
            denominator = max(abs(first), 1e-8)
            bm25_ambiguous = abs(first - second) / denominator < 0.20

        # Same idea for dense similarity.
        dense_ambiguous = True
        if len(faiss_results) >= 2:
            first = faiss_results[0][1]
            second = faiss_results[1][1]
            dense_ambiguous = abs(first - second) < 0.05

        # Very little explicit information is inherently ambiguous.
        low_specificity = filled <= 1

        return broad_pool and low_specificity and (
            bm25_ambiguous or dense_ambiguous
        )

    def choose_attribute(
        self,
        products: List[Dict[str, Any]],
        active_slots: Dict[str, List[str]],
        asked_attributes: set[str],
        no_preference_attributes: set[str],
    ) -> Optional[str]:

        eligible = [
            attr for attr in ATTRIBUTE_ORDER
            if attr not in active_slots
            and attr not in asked_attributes
            and attr not in no_preference_attributes
        ]

        if not eligible:
            return None

        default_attr = eligible[0]

        best_attr = None
        best_information = 0.0

        for attr in eligible:
            if attr == "feature":
                continue

            information = self._attribute_information(products, attr)

            if information > best_information:
                best_information = information
                best_attr = attr

        # Only override the high-value default question when candidate evidence
        # is genuinely strong.
        if best_attr and best_information >= 0.65:
            return best_attr

        return default_attr

    def decide(
        self,
        turn: int,
        active_slots: Dict[str, List[str]],
        candidates: List[str],
        products_by_asin: Dict[str, Dict[str, Any]],
        bm25_results,
        faiss_results,
        asked_attributes: set[str],
        no_preference_attributes: set[str],
    ) -> tuple[bool, Optional[str]]:
        """Return (should_clarify, attribute)."""
        if not self.is_over_general(
            turn, active_slots, candidates, bm25_results, faiss_results
        ):
            return False, None

        # Use the highest-ranked candidates to estimate uncertainty.
        products = [
            products_by_asin[asin]
            for asin in candidates[:75]
            if asin in products_by_asin
        ]

        attribute = self.choose_attribute(
            products,
            active_slots,
            asked_attributes,
            no_preference_attributes,
        )

        return attribute is not None, attribute