"""
state_tracker_rulebased.py

A high-performance, zero-latency Dialogue State Tracker for e-commerce assistants.
Uses regex slot extraction, TF-IDF cosine similarity tag mapping, and robust intent-override handling.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ALLOWED_ATTRIBUTES = [
    "category", "material", "color", "size", "style", 
    "brand", "budget", "feature", "use_case"
]

SINGLE_VALUE_SLOTS = {"category", "budget", "size"}

MATERIALS = r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|fleece|canvas|denim|down|velvet)\b"
COLORS = r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|silver|gold|navy)\b"
SIZES = r"\b(small|medium|large|xlarge|x-large|xxl|xs|s|m|l|xl|size\s*\d+(?:\.\d+)?|\b\d{1,2}[xsml]\b)\b"
STYLES = r"\b(casual|formal|vintage|modern|slim fit|regular fit|loose|athletic|boho|streetwear|chic|classic)\b"
USE_CASES = r"\b(hiking|running|gym|winter|spring|autumn|fall|outdoor|work|party|travel|yoga|swimming|wedding|sports)\b"
CATEGORIES = r"\b(shoes?|boots?|sneakers?|jackets?|coats?|dresses?|pants|jeans|shirts?|t-shirts?|tops?|bottoms?|hoodies?|sweaters?|shorts|skirts?|socks|bags?|backpacks?|hats?)\b"

SLOT_PATTERNS = {
    "budget": re.compile(r"(?:under|below|less than|budget(?: of)?|around|\$)\s*(\$?\d+(?:\.\d{2})?)", re.I),
    "material": re.compile(MATERIALS, re.I),
    "color": re.compile(COLORS, re.I),
    "size": re.compile(SIZES, re.I),
    "style": re.compile(STYLES, re.I),
    "use_case": re.compile(USE_CASES, re.I),
    "category": re.compile(CATEGORIES, re.I),
}

ATTRIBUTE_PROFILES: Dict[str, str] = {
    "category": "product type clothing apparel item shoes boots jacket dress pants shirt top bottom outerwear accessories",
    "material": "fabric textile leather cotton wool polyester silk nylon spandex denim canvas fleece synthetic velvet down insulation",
    "color": "color shade hue tint black white blue red green yellow brown pink purple grey gray silver gold dark light navy",
    "size": "fit sizing dimensions small medium large xl xxl measurements waist length tight loose regular slim oversized",
    "style": "style look aesthetic casual formal vintage modern slim fit regular fit loose athletic boho streetwear chic classic retro",
    "brand": "manufacturer designer label company store make original logo trademark maker producer vendor Nike Adidas Puma Reebok Under Armour North Face Columbia Patagonia Levi's Gucci Prada Versace",
    "budget": "price cost money cheap expensive dollar payment discount sale value affordable premium luxury economy under price-range",
    "feature": "detail specification characteristic function pocket zipper waterproof breathable lightweight heavy durable flexible stretch",
    "use_case": "activity occasion purpose weather winter summer spring autumn fall running hiking gym work party travel indoor outdoor training sports"
}

RESET_ALL_PATTERN = re.compile(
    r"\b(start over|clear all|forget everything|scratch that|reset all)\b", 
    re.I
)

SHIFT_OVERRIDE_PATTERN = re.compile(
    r"\b(actually|instead|changed my mind|nevermind|never mind|forget about|switch to|rather have|no longer|change my budget)\b", 
    re.I
)

NEGATION_PATTERN = re.compile(
    r"\b(no|not|without|don't want|other than|except)\s+([a-z0-9\s]+)\b", 
    re.I
)


class PreferenceTagMapper:
    def __init__(self, attributes: List[str] = ALLOWED_ATTRIBUTES, profiles: Optional[Dict[str, str]] = None) -> None:
        self.attributes = attributes
        self.profiles = profiles if profiles is not None else ATTRIBUTE_PROFILES
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        
        profile_corpus = [self.profiles.get(attr, attr) for attr in self.attributes]
        self.attribute_matrix = self.vectorizer.fit_transform(profile_corpus)

    def map_tag_to_attribute(self, tag: str, threshold: float = 0.12) -> Optional[str]:
        tag_vector = self.vectorizer.transform([tag])
        similarities = cosine_similarity(tag_vector, self.attribute_matrix).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        return self.attributes[best_idx] if best_score >= threshold else None

    def prioritize_attributes(self, user_tags: List[str], threshold: float = 0.12) -> List[str]:
        ranked_attributes: List[str] = []
        for tag in user_tags:
            matched_attr = self.map_tag_to_attribute(str(tag).lower().strip(), threshold=threshold)
            if matched_attr and matched_attr not in ranked_attributes:
                ranked_attributes.append(matched_attr)
        return ranked_attributes


class DialogueStateTracker:
    def __init__(self) -> None:
        self.active_slots: Dict[str, List[str]] = {}
        self.asked_attributes: Set[str] = set()
        self.tag_mapper = PreferenceTagMapper()
        self.dynamic_attribute_priority: List[str] = []

    def reset(self, user_profile: Optional[dict] = None) -> None:
        self.active_slots = {}
        self.asked_attributes = set()
        self.dynamic_attribute_priority = []
        
        if not user_profile or not isinstance(user_profile, dict):
            self.dynamic_attribute_priority = list(ALLOWED_ATTRIBUTES)
            return

        pref_tags = user_profile.get("preference_tags", [])
        if isinstance(pref_tags, list):
            mapped_attrs = self.tag_mapper.prioritize_attributes(pref_tags)
            global_defaults = ["category", "color", "size", "material", "style", "use_case", "budget", "brand", "feature"]
            self.dynamic_attribute_priority = mapped_attrs + [a for a in global_defaults if a not in mapped_attrs]
        else:
            self.dynamic_attribute_priority = list(ALLOWED_ATTRIBUTES)

        for key in ["size", "brand", "style"]:
            val = user_profile.get(key)
            if val and key not in self.active_slots:
                self.active_slots[key] = [str(val).lower()]

    def extract_slots_regex(self, text: str) -> Dict[str, List[str]]:
        extracted: Dict[str, List[str]] = {}
        
        budget_match = SLOT_PATTERNS["budget"].search(text)
        if budget_match:
            val = budget_match.group(1).replace("$", "").strip()
            extracted["budget"] = [f"under ${val}"]
            
        for slot_name in ["category", "material", "color", "size", "style", "use_case"]:
            matches = SLOT_PATTERNS[slot_name].finditer(text)
            found = []
            for m in matches:
                word = m.group(0).lower()
                if slot_name == "category":
                    if word.endswith("s") and word not in ["jeans", "pants", "shorts", "boots", "shoes", "socks"]:
                        word = word[:-1]
                found.append(word)
            if found:
                extracted[slot_name] = list(dict.fromkeys(found))
                
        return extracted

    def _wipe_slot(self, slot_name: str) -> None:
        if slot_name in self.active_slots:
            del self.active_slots[slot_name]

    def _apply_negation_and_shift_wiping(self, text: str) -> Set[str]:
        negated_values: Set[str] = set()

        if RESET_ALL_PATTERN.search(text):
            self.active_slots.clear()
            return negated_values

        if SHIFT_OVERRIDE_PATTERN.search(text):
            category_match = SLOT_PATTERNS["category"].search(text)
            if category_match:
                self._wipe_slot("category")
                self._wipe_slot("size")
                self._wipe_slot("material")

        negation_matches = NEGATION_PATTERN.findall(text)
        slots_to_wipe = []
        
        for _, negated_phrase in negation_matches:
            negated_phrase = negated_phrase.strip().lower()
            for slot_name, vals in list(self.active_slots.items()):
                remaining = []
                for val in vals:
                    if val.lower() in negated_phrase or negated_phrase in val.lower():
                        negated_values.add(val.lower())
                    else:
                        remaining.append(val)
                
                if remaining:
                    self.active_slots[slot_name] = remaining
                else:
                    slots_to_wipe.append(slot_name)

            for token in negated_phrase.split():
                negated_values.add(token)

        for slot_name in slots_to_wipe:
            self._wipe_slot(slot_name)

        return negated_values

    def update_state(self, user_message: str, turn: int) -> Tuple[Dict[str, List[str]], int, int]:
        clean_msg = user_message.lower().strip()
        
        # Step 1: Wipe overridden or negated slots and capture negated terms
        negated_terms = self._apply_negation_and_shift_wiping(clean_msg)
        
        # Step 2: Extract new slot candidates
        new_slots = self.extract_slots_regex(clean_msg)

        # Step 3: Filter out any extracted terms that match negated values in this turn
        if negated_terms:
            filtered_new_slots = {}
            for slot, vals in new_slots.items():
                valid_vals = [v for v in vals if v.lower() not in negated_terms]
                if valid_vals:
                    filtered_new_slots[slot] = valid_vals
            new_slots = filtered_new_slots

        # Step 4: Merge slots
        is_shift = bool(SHIFT_OVERRIDE_PATTERN.search(clean_msg))

        for key, vals in new_slots.items():
            if not vals:
                continue

            if key in SINGLE_VALUE_SLOTS or is_shift:
                self.active_slots[key] = [vals[-1]]
            else:
                if key in self.active_slots:
                    merged = list(dict.fromkeys(self.active_slots[key] + vals))
                    self.active_slots[key] = merged
                else:
                    self.active_slots[key] = vals

        return self.active_slots, 0, 0

    def get_next_ask_attribute(self) -> Optional[str]:
        for attr in self.dynamic_attribute_priority:
            if attr not in self.active_slots and attr not in self.asked_attributes:
                self.asked_attributes.add(attr)
                return attr
        return None

    def build_search_query(self, category: Optional[str] = None) -> str:
        terms = []
        if category:
            terms.append(category)
        elif "category" in self.active_slots:
            terms.extend(self.active_slots["category"])

        for key in ["color", "material", "brand", "style", "use_case", "size", "feature"]:
            if key in self.active_slots:
                terms.extend(self.active_slots[key])
                
        query = " ".join(terms).strip()
        return query