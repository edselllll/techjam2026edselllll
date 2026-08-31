from __future__ import annotations

from typing import Dict, List, Tuple


class RetrievalOrchestrator:
    """
    Runtime orchestration for candidate truncation.

    Retrieval depth stays fixed for stable recall and score normalization.
    Only the number of candidates passed to the reranker changes according
    to current session specificity and intent.
    """

    SLOT_WEIGHTS = {
        "category": 0.8,
        "material": 1.0,
        "color": 0.7,
        "size": 0.7,
        "style": 0.8,
        "brand": 1.2,
        "budget": 1.0,
        "feature": 1.3,
        "use_case": 1.0,
    }

    def specificity_score(self, active_slots: Dict[str, List[str]]) -> float:
        score = 0.0

        for slot, values in active_slots.items():
            if not values:
                continue

            score += self.SLOT_WEIGHTS.get(slot, 0.5)

            # Multiple constraints in one slot add some specificity,
            # but shouldn't dominate the score.
            if len(values) > 1:
                score += min(0.3, 0.1 * (len(values) - 1))

        return score

    def candidate_depth(
        self,
        intent: str,
        active_slots: Dict[str, List[str]],
    ) -> Tuple[int, int, int]:
        """
        Returns:
            fusion_k, bm25_k, faiss_k

        Broad queries preserve the full candidate pool.
        Specific queries can use a smaller reranking pool.
        """
        specificity = self.specificity_score(active_slots)

        # Broad / exploratory
        if specificity < 1.5:
            return 100, 150, 75

        # Moderately constrained
        if specificity < 3.0:
            return 95, 145, 70

        # Highly constrained buying request
        if intent == "Buying":
            return 90, 135, 65

        # Specific but still browsing
        return 95, 140, 70