from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Set
from huggingface_hub import InferenceClient

# Constants aligned with local_evaluator.py
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

MATERIALS = r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b"
COLORS = r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b"
SIZES = r"\b(small|medium|large|xlarge|x-large|xxl|xs|s|m|l|xl|size\s*\d+(\.\d+)?|\b\d{1,2}[xsml]\b)\b"
STYLES = r"\b(casual|formal|vintage|modern|slim fit|regular fit|loose|athletic|boho|streetwear)\b"
USE_CASES = r"\b(hiking|running|gym|winter|outdoor|work|party|travel|yoga|swimming|wedding)\b"

ATTRIBUTE_PRIORITY = ["category", "color", "size", "material", "style", "use_case", "budget", "brand", "feature"]

SLOT_PATTERNS = {
    "budget": re.compile(r"(?:under|below|less than|budget(?: of)?|around|\$)\s*(\$?\d+(?:\.\d{2})?)", re.I),
    "material": re.compile(MATERIALS, re.I),
    "color": re.compile(COLORS, re.I),
    "size": re.compile(SIZES, re.I),
    "style": re.compile(STYLES, re.I),
    "use_case": re.compile(USE_CASES, re.I),
}

PIVOT_TRIGGERS = re.compile(
    r"\b(ignore|instead|actually|change|rather|never mind|different|switch|replace|scratch that)\b", re.I
)


class DialogueStateTracker:
    def __init__(self, api_key: Optional[str] = None) -> None:
        hf_token = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.llm_model = "meta-llama/Llama-3.1-8B-Instruct"
        self.client = InferenceClient(model=self.llm_model, api_key=hf_token) if hf_token else None
        
        self.active_slots: Dict[str, List[str]] = {}
        self.asked_attributes: Set[str] = set()

    def reset(self, user_profile: Optional[dict] = None) -> None:
        """Resets dialogue state and seeds initial profile preferences."""
        self.active_slots = {}
        self.asked_attributes = set()
        
        if user_profile and isinstance(user_profile, dict):
            # Seed non-conflicting default preferences if present
            for key in ["size", "brand", "style"]:
                val = user_profile.get(key)
                if val:
                    self.active_slots[key] = [str(val).lower()]

    def extract_slots_regex(self, text: str) -> Dict[str, List[str]]:
        """Fast, zero-latency regex extraction for common slot attributes."""
        extracted: Dict[str, List[str]] = {}
        
        budget_match = SLOT_PATTERNS["budget"].search(text)
        if budget_match:
            val = budget_match.group(1).replace("$", "").strip()
            extracted["budget"] = [f"under ${val}"]
            
        for slot_name in ["material", "color", "size", "style", "use_case"]:
            matches = SLOT_PATTERNS[slot_name].findall(text)
            if matches:
                cleaned = [m[0] if isinstance(m, tuple) else m for m in matches]
                extracted[slot_name] = list(set(item.lower() for item in cleaned))
                
        return extracted

    def _run_llm_override(self, user_message: str) -> Dict[str, List[str]]:
        """Triggers the fast 8B LLM to resolve slot conflicts during intent pivots."""
        if not self.client:
            return self.active_slots

        prompt = f"""You are an intent override state updater for a shopping assistant.
Update the current JSON slot state based on the user's latest message.

RULES:
1. If the user cancels/overrides a preference (e.g., "ignore black, want red"), REMOVE the old value and REPLACE it.
2. Preserve unrelated active slots unless explicitly contradicted.
3. Output ONLY a valid JSON object.

Allowed slot keys: ["category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case"]

Current State: {json.dumps(self.active_slots)}
User Message: "{user_message}"
Output JSON:"""

        try:
            response = self.client.text_generation(
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.0,
                stop_sequences=["}"]
            )
            raw_json = response.strip()
            if not raw_json.endswith("}"):
                raw_json += "}"
            
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                return {k: [str(v)] if isinstance(v, str) else v for k, v in parsed.items() if k in ALLOWED_ATTRIBUTES}
        except Exception:
            pass  # Fallback to current slots on API timeout or parse error
            
        return self.active_slots

    def update_state(self, user_message: str, turn: int) -> Dict[str, List[str]]:
        """Updates slot state via fast regex or LLM override depending on pivot language."""
        has_pivot = bool(PIVOT_TRIGGERS.search(user_message))
        
        if has_pivot and turn > 1:
            # Route to LLM to clear/replace invalidated constraints
            self.active_slots = self._run_llm_override(user_message)
        else:
            # Happy path: Fast regex accumulation
            new_slots = self.extract_slots_regex(user_message)
            for key, val in new_slots.items():
                if key in self.active_slots:
                    self.active_slots[key] = list(set(self.active_slots[key] + val))
                else:
                    self.active_slots[key] = val

        return self.active_slots

    def get_next_ask_attribute(self) -> Optional[str]:
        """Determines the next unasked attribute for local_evaluator customer_reply alignment."""
        for attr in ATTRIBUTE_PRIORITY:
            if attr not in self.active_slots and attr not in self.asked_attributes:
                self.asked_attributes.add(attr)
                return attr
        return "feature"

    def build_search_query(self, category: str) -> str:
        """Flattens active slots into an optimized candidate search string."""
        terms = [category]
        for key, values in self.active_slots.items():
            if values:
                terms.extend(values)
        return " ".join(terms).strip()