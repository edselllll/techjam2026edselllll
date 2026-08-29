from __future__ import annotations
import re
from typing import Dict, Any, Optional, Set

# Allowed attributes strictly per competition guidelines
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", 
    "brand", "budget", "feature", "use_case", "other"
}

class DialogueState:
    def __init__(self) -> None:
        self.slots: Dict[str, str] = {}
        self.disliked_asins: Set[str] = set()

    def update_from_input(self, user_message: str) -> bool:
        """
        Parses input, handles incremental accumulation AND slot overrides (intent pivots).
        Returns True if an explicit Intent Override/Pivot was detected.
        """
        text = user_message.lower()
        intent_override_detected = False

        # 1. Detect Intent Override / Pivot signals (e.g., "instead", "actually", "change my mind")
        pivot_keywords = {"instead", "actually", "nevermind", "change", "forget", "rather than", "switch"}
        if any(kw in text for kw in pivot_keywords):
            intent_override_detected = True

        # 2. Extract new slot candidates
        new_slots = self._extract_slots(text)

        # 3. Apply state update rule: Override conflicting old slots or accumulate new ones
        for key, val in new_slots.items():
            if key in self.slots and self.slots[key] != val:
                # Slot rewriting / Override
                intent_override_detected = True
            self.slots[key] = val

        return intent_override_detected

    def _extract_slots(self, text: str) -> Dict[str, str]:
        extracted = {}

        # Colors
        colors = {"black", "white", "blue", "red", "green", "yellow", "brown", "pink", "purple", "grey", "gray", "leather"}
        for c in colors:
            if re.search(r'\b' + c + r'\b', text):
                extracted["color"] = c
                break

        # Materials
        materials = {"leather", "cotton", "wool", "polyester", "silk", "denim", "canvas", "mesh"}
        for m in materials:
            if re.search(r'\b' + m + r'\b', text):
                extracted["material"] = m
                break

        # Department / Audience
        departments = {"men", "mens", "women", "womens", "kids", "baby"}
        for d in departments:
            if re.search(r'\b' + d + r'\b', text):
                extracted["use_case"] = d
                break

        # Budget / Price limit
        price_match = re.search(r'(\$|under|less than|below)\s*(\d+)', text)
        if price_match:
            extracted["budget"] = f"under ${price_match.group(2)}"

        return extracted

    def is_over_general(self, history: list[str]) -> bool:
        """
        Proactive Guidance Rule:
        Triggers Over-Generality cutoff if candidate pool is at risk of overload
        (fewer than 2 active specific slots defined after turn 1).
        """
        # Exclude budget/use_case if they are too broad alone
        specific_slots = [k for k in self.slots if k in {"color", "material", "category", "brand", "style"}]
        
        # If early in turn and active specific slots < 2 -> Over-general
        if len(history) <= 2 and len(specific_slots) < 2:
            return True
        return False

    def select_proactive_guidance_attribute(self) -> str:
        """Selects the highest priority missing slot to ask for proactive convergence."""
        priority_order = ["category", "color", "material", "size", "budget", "style"]
        for attr in priority_order:
            if attr not in self.slots and attr in ALLOWED_ATTRIBUTES:
                return attr
        return "other"

    def build_search_query(self) -> str:
        """Synthesizes clean search query from accumulated slots."""
        return " ".join(self.slots.values())