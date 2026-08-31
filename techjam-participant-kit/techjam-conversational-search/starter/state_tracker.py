from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
from huggingface_hub import InferenceClient

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

# Semantic map connecting preference_tags -> priority slot attributes
TAG_TO_ATTRIBUTE_MAP = {
    "warmth": ["material", "use_case"],       # e.g., wool, fleece, winter
    "weather": ["use_case", "material"],      # e.g., outdoor, rain, waterproof
    "performance": ["use_case", "feature"],   # e.g., running, breathable
    "durability": ["material", "brand"],      # e.g., leather, heavy-duty
    "fit": ["size", "style"],                 # e.g., slim fit, medium
    "style": ["style", "color", "brand"],     # e.g., casual, vintage
    "comfort": ["material", "feature"],       # e.g., cotton, cushioned
    "general shopping": ["category", "budget"],
    "material": ["material"]
}

MATERIALS = r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b"
COLORS = r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b"
SIZES = r"\b(small|medium|large|xlarge|x-large|xxl|xs|s|m|l|xl|size\s*\d+(\.\d+)?|\b\d{1,2}[xsml]\b)\b"
STYLES = r"\b(casual|formal|vintage|modern|slim fit|regular fit|loose|athletic|boho|streetwear)\b"
USE_CASES = r"\b(hiking|running|gym|winter|outdoor|work|party|travel|yoga|swimming|wedding)\b"

ATTRIBUTE_PRIORITY = ["category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "other", None]

SLOT_PATTERNS = {
    "budget": re.compile(r"(?:under|below|less than|budget(?: of)?|around|\$)\s*(\$?\d+(?:\.\d{2})?)", re.I),
    "material": re.compile(MATERIALS, re.I),
    "color": re.compile(COLORS, re.I),
    "size": re.compile(SIZES, re.I),
    "style": re.compile(STYLES, re.I),
    "use_case": re.compile(USE_CASES, re.I),
}

PIVOT_TRIGGERS = re.compile(
    r"\b(ignore|instead|actually|change|changed|rather|never mind|nevermind|different|switch|replace|scratch|no|not|dont|don't|forget|prefer|would like)\b", re.I
)


class DialogueStateTracker:
    def __init__(self, api_key: Optional[str] = None) -> None:
        hf_token = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self.llm_model = "meta-llama/Llama-3.1-8B-Instruct"
        self.client = InferenceClient(model=self.llm_model, api_key=hf_token) if hf_token else None
        
        self.active_slots: Dict[str, List[str]] = {}
        self.asked_attributes: Set[str] = set()
        
        # Cumulative session trackers for logging/debugging
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    def reset(self, user_profile: Optional[dict] = None) -> None:
        """Resets dialogue state and seeds initial profile preferences."""
        self.active_slots = {}
        self.asked_attributes = set()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        if user_profile and isinstance(user_profile, dict):
            # Seed preference_tags if available in profile contract
            pref_tags = user_profile.get("preference_tags", [])
            if isinstance(pref_tags, list):
                for tag in pref_tags:
                    extracted = self.extract_slots_regex(str(tag))
                    for k, v in extracted.items():
                        self.active_slots[k] = v

            for key in ["size", "brand", "style"]:
                val = user_profile.get(key)
                if val and key not in self.active_slots:
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

    def _run_llm_override(self, user_message: str) -> Tuple[Dict[str, List[str]], int, int]:
        """Uses chat_completion for Novita API compatibility and captures turn token usage."""
        if not self.client:
            return self.active_slots, 0, 0

        system_prompt = (
            "You are a dialogue state tracking engine for an e-commerce assistant.\n"
            "Your task is to update the JSON slot state after a user changes their mind or alters search constraints.\n\n"
            "STRICT OVERRIDE RULES:\n"
            "1. IDENTIFY CONTRADICTIONS: When a user provides a new preference for an existing slot category (e.g., color, material, category, size), DELETE the old value and REPLACE it entirely with the new value.\n"
            "2. DELETE CANCELLED PREFERENCES: If the user says \"ignore X\", \"no X\", or \"forget X\", REMOVE the \"X\" key from the state completely.\n"
            "3. DO NOT MERGE CONTRADICTORY SLOTS: Never keep both old and new values for the same attribute (e.g., if old color is \"black\" and user says \"actually red\", output MUST be \"color\": [\"red\"], NOT [\"black\", \"red\"]).\n"
            "4. PRESERVE UNTOUCHED CONSTRAINTS: Keep existing slots ONLY if they do not conflict with the new request.\n"
            "5. STRICT JSON OUTPUT: Output ONLY a valid JSON dictionary mapping allowed keys to arrays of string values.\n\n"
            f"Allowed slot keys:\n{json.dumps(list(ALLOWED_ATTRIBUTES))}\n\n"
            "--- EXAMPLE 1: Value Override ---\n"
            "Current State: {\"category\": [\"jacket\"], \"color\": [\"black\"], \"material\": [\"leather\"]}\n"
            "User Message: \"Actually, ignore the leather preference. What I need is synthetic fabric.\"\n"
            "Output JSON:\n"
            "{\"category\": [\"jacket\"], \"color\": [\"black\"], \"material\": [\"synthetic fabric\"]}\n\n"
            "--- EXAMPLE 2: Category Pivot & Slot Deletion ---\n"
            "Current State: {\"category\": [\"running shoes\"], \"brand\": [\"nike\"], \"size\": [\"10\"], \"color\": [\"blue\"]}\n"
            "User Message: \"Instead of running shoes, I want hiking boots in brown.\"\n"
            "Output JSON:\n"
            "{\"category\": [\"hiking boots\"], \"brand\": [\"nike\"], \"size\": [\"10\"], \"color\": [\"brown\"]}\n\n"
            "--- EXAMPLE 3: Explicit Cancellation ---\n"
            "Current State: {\"category\": [\"dress\"], \"style\": [\"vintage\"], \"budget\": [\"under $50\"]}\n"
            "User Message: \"Never mind the budget limit, show me modern styles instead.\"\n"
            "Output JSON:\n"
            "{\"category\": [\"dress\"], \"style\": [\"modern\"]}"
        )

        user_content = (
            f"--- CURRENT TASK ---\n"
            f"Current State: {json.dumps(self.active_slots)}\n"
            f"User Message: \"{user_message}\"\n"
            f"Output JSON:"
        )

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=150,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            p_tokens = getattr(response.usage, "prompt_tokens", 0) if hasattr(response, "usage") else 0
            c_tokens = getattr(response.usage, "completion_tokens", 0) if hasattr(response, "usage") else 0

            self.total_prompt_tokens += p_tokens
            self.total_completion_tokens += c_tokens

            raw_json = response.choices[0].message.content.strip()
            parsed = json.loads(raw_json)
            
            if isinstance(parsed, dict):
                cleaned = {}
                for k, v in parsed.items():
                    if k in ALLOWED_ATTRIBUTES and v:
                        cleaned[k] = [str(x).lower() for x in v] if isinstance(v, list) else [str(v).lower()]
                return cleaned, p_tokens, c_tokens
                
        except Exception as e:
            print(f"[LLM ERROR] Exception during override call: {e}")
            
        return self.active_slots, 0, 0

    def update_state(self, user_message: str, turn: int) -> Tuple[Dict[str, List[str]], int, int]:
        """Updates slot state: LLM override on pivot keywords, accumulation on regular turns."""
        clean_msg = user_message.lower().strip()
        has_pivot = bool(PIVOT_TRIGGERS.search(clean_msg))
        
        # 1. Pivot Path (LLM resolves conflicts/deletions)
        if has_pivot and turn >= 2:
            new_slots, p_tokens, c_tokens = self._run_llm_override(user_message)
            self.active_slots = new_slots
            # NOTE: Removed self.asked_attributes overwrite to prevent locking out future clarification turns
            return self.active_slots, p_tokens, c_tokens

        # 2. Non-Pivot Path (Accumulate active slots without destructive assignment)
        new_slots = self.extract_slots_regex(clean_msg)
        for key, vals in new_slots.items():
            if key in self.active_slots:
                # Merge unique values to preserve history across turns
                merged = list(dict.fromkeys(self.active_slots[key] + vals))
                self.active_slots[key] = merged
            else:
                self.active_slots[key] = vals

        return self.active_slots, 0, 0

    def get_next_ask_attribute(self) -> Optional[str]:
        """Determines the next unasked attribute for local_evaluator customer_reply alignment."""
        for attr in ATTRIBUTE_PRIORITY:
            if attr not in self.active_slots and attr not in self.asked_attributes:
                self.asked_attributes.add(attr)
                return attr
        return None

    def build_search_query(self, category: str) -> str:
        """Constructs an optimized BM25 query prioritising high-precision attributes."""
        terms = [category] if category else []
        
        # High-precision core keys first
        for key in ["category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "other", None]:
            if key in self.active_slots:
                terms.extend(self.active_slots[key])
                
        query = " ".join(terms).strip()
        return query if query else category