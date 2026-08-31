"""
state_tracker_rulebased.py

Fast rule-based Dialogue State Tracker for the TechJam shopping agent.
Supports regex slot extraction, contextual clarification capture, intent overrides,
no-preference handling, profile-aware question ordering, and optional entropy selection.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ALLOWED_ATTRIBUTES = ["category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case"]
SINGLE_VALUE_SLOTS = {"category", "budget", "size"}

# Feature comes first because arbitrary Amazon features/details often carry the
# most discriminative information. Category is usually already given initially.
QUESTION_PRIORITY = ["feature", "material", "color", "style", "use_case", "size", "budget", "brand", "category"]

MATERIALS = r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|fleece|canvas|denim|down|velvet|linen|acrylic|suede)\b"
COLORS = r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|silver|gold|navy|beige|tan|khaki|cream)\b"
SIZES = r"\b(small|medium|large|xlarge|x-large|xxl|xxxl|xs|s|m|l|xl|size\s*\d+(?:\.\d+)?|\d{1,2}[xsml])\b"
STYLES = r"\b(casual|formal|vintage|modern|slim fit|regular fit|relaxed fit|loose|athletic|boho|streetwear|chic|classic|retro|oversized|fitted)\b"
USE_CASES = r"\b(hiking|running|gym|winter|summer|spring|autumn|fall|outdoor|work|party|travel|yoga|swimming|wedding|sports|training|walking|camping|skiing)\b"
CATEGORIES = r"\b(shoes?|boots?|sneakers?|jackets?|coats?|dresses?|pants|jeans|shirts?|t-shirts?|tops?|bottoms?|hoodies?|sweaters?|shorts|skirts?|socks|bags?|backpacks?|hats?|earrings?|jewelry|sandals?|slippers?|leggings?|vests?|gloves?|scarves?|belts?)\b"

SLOT_PATTERNS = {
    "budget": re.compile(r"(?:under|below|less than|budget(?: of)?|around|up to|\$)\s*(\$?\d+(?:\.\d{1,2})?)", re.I),
    "material": re.compile(MATERIALS, re.I),
    "color": re.compile(COLORS, re.I),
    "size": re.compile(SIZES, re.I),
    "style": re.compile(STYLES, re.I),
    "use_case": re.compile(USE_CASES, re.I),
    "category": re.compile(CATEGORIES, re.I),
}

ATTRIBUTE_PROFILES: Dict[str, str] = {
    "category": "product type clothing apparel shoes boots jacket dress pants shirt top outerwear accessories jewelry",
    "material": "fabric textile leather cotton wool polyester silk nylon spandex denim canvas fleece material",
    "color": "color shade hue black white blue red green yellow brown pink purple grey gray silver gold navy",
    "size": "fit sizing dimensions small medium large xl xxl measurements waist length tight loose regular slim oversized",
    "style": "style look aesthetic casual formal vintage modern slim regular loose athletic boho streetwear chic classic retro",
    "brand": "manufacturer designer label company store make logo trademark maker producer vendor brand",
    "budget": "price cost money cheap expensive dollar discount sale value affordable premium luxury budget",
    "feature": "detail specification characteristic feature comfort durability performance waterproof breathable lightweight flexible stretch pocket zipper quick drying moisture wicking warmth insulation",
    "use_case": "activity occasion purpose weather winter summer running hiking gym work party travel indoor outdoor training sports",
}

RESET_ALL_PATTERN = re.compile(r"\b(start over|clear all|forget everything|scratch that|reset all)\b", re.I)
HARD_OVERRIDE_PATTERN = re.compile(r"\b(ignore my earlier preference|ignore my previous preference|changed my mind|forget my earlier preference|forget my previous preference|switch to|rather have|no longer want)\b", re.I)
SOFT_SHIFT_PATTERN = re.compile(r"\b(actually|instead|rather|change my budget|nevermind|never mind)\b", re.I)
NO_PREFERENCE_PATTERN = re.compile(r"\b(no preference|don't have (?:an? )?(?:additional )?preference|do not have (?:an? )?(?:additional )?preference|doesn't matter|does not matter|any is fine|anything is fine|use your judgment|you decide)\b", re.I)
NEGATION_PATTERN = re.compile(r"\b(no|not|without|don't want|do not want|other than|except)\s+([a-z0-9][a-z0-9\s\-]+)", re.I)
CLARIFICATION_REPLY_PATTERN = re.compile(r"(?:for that,\s*)?what matters is:\s*(.+)", re.I)
OVERRIDE_VALUE_PATTERN = re.compile(r"what i need is:\s*(.+)",re.I)

class PreferenceTagMapper:
    def __init__(self, attributes: List[str] = ALLOWED_ATTRIBUTES, profiles: Optional[Dict[str, str]] = None) -> None:
        self.attributes = attributes
        self.profiles = profiles or ATTRIBUTE_PROFILES
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        self.attribute_matrix = self.vectorizer.fit_transform([self.profiles.get(attr, attr) for attr in attributes])

    def map_tag_to_attribute(self, tag: str, threshold: float = 0.12) -> Optional[str]:
        tag_vector = self.vectorizer.transform([str(tag).lower().strip()])
        similarities = cosine_similarity(tag_vector, self.attribute_matrix).flatten()
        best_idx = int(np.argmax(similarities))
        return self.attributes[best_idx] if float(similarities[best_idx]) >= threshold else None

    def prioritize_attributes(self, user_tags: List[str], threshold: float = 0.12) -> List[str]:
        ranked = []
        for tag in user_tags:
            attr = self.map_tag_to_attribute(tag, threshold)
            if attr and attr not in ranked:
                ranked.append(attr)
        return ranked


class DialogueStateTracker:
    def __init__(self, catalog: Optional[List[Dict[str, str]]] = None) -> None:
        self.active_slots: Dict[str, List[str]] = {}
        self.asked_attributes: Set[str] = set()
        self.no_preference_attributes: Set[str] = set()
        self.expected_attribute: Optional[str] = None
        self.profile_priority: List[str] = []
        self.tag_mapper = PreferenceTagMapper()
        self.catalog = catalog or []

    def reset(self, user_profile: Optional[dict] = None) -> None:
        self.active_slots, self.asked_attributes = {}, set()
        self.no_preference_attributes, self.expected_attribute = set(), None
        self.profile_priority = []

        if user_profile and isinstance(user_profile, dict):
            tags = user_profile.get("preference_tags", [])
            if isinstance(tags, list):
                self.profile_priority = self.tag_mapper.prioritize_attributes(tags)

    @staticmethod
    def _clean(value: str) -> str:
        return re.sub(r"\s+", " ", str(value).lower().strip()).strip(" \t\n.;,")

    def _wipe_slot(self, slot: str) -> None:
        self.active_slots.pop(slot, None)

    def _clear_preferences(self,preserve_category: bool = False,preserve_asked: bool = False,) -> None:
        category = self.active_slots.get("category") if preserve_category else None
        asked = set(self.asked_attributes) if preserve_asked else set()

        self.active_slots.clear()
        self.no_preference_attributes.clear()
        self.expected_attribute = None
        self.asked_attributes = asked

        if category:
            self.active_slots["category"] = category

    def _store(self, slot: str, values: List[str], replace: bool = False) -> None:
        values = list(dict.fromkeys(self._clean(v) for v in values if self._clean(v)))
        if not values:
            return

        if replace or slot in SINGLE_VALUE_SLOTS:
            self.active_slots[slot] = [values[-1]]
        else:
            self.active_slots[slot] = list(
                dict.fromkeys(self.active_slots.get(slot, []) + values)
            )

    def extract_slots_regex(self, text: str) -> Dict[str, List[str]]:
        extracted: Dict[str, List[str]] = {}

        budget = SLOT_PATTERNS["budget"].search(text)
        if budget:
            extracted["budget"] = [f"under ${budget.group(1).replace('$', '').strip()}"]

        for slot in ["category", "material", "color", "size", "style", "use_case"]:
            found = [m.group(0).lower().strip() for m in SLOT_PATTERNS[slot].finditer(text)]
            if slot == "category":
                preserve = {"jeans", "pants", "shorts", "boots", "shoes", "socks", "earrings", "leggings", "gloves"}
                found = [v[:-1] if v.endswith("s") and v not in preserve and len(v) > 3 else v for v in found]
            if found:
                extracted[slot] = list(dict.fromkeys(found))
        return extracted

    def _capture_expected_answer(self, text: str) -> bool:
        """Store free-form clarification replies under the attribute we asked for."""
        if not self.expected_attribute:
            return False

        expected = self.expected_attribute

        if NO_PREFERENCE_PATTERN.search(text):
            self.no_preference_attributes.add(expected)
            self.expected_attribute = None
            return True

        match = CLARIFICATION_REPLY_PATTERN.search(text)
        if not match:
            return False

        values = [self._clean(v) for v in match.group(1).split(";") if self._clean(v)]
        if values:
            self._store(expected, values, replace=expected in SINGLE_VALUE_SLOTS)
            self.no_preference_attributes.discard(expected)

        self.expected_attribute = None
        return True

    def _apply_negations(self, text: str) -> Set[str]:
        negated: Set[str] = set()

        for _, phrase in NEGATION_PATTERN.findall(text):
            phrase = self._clean(phrase)
            for slot, values in list(self.active_slots.items()):
                remaining = []
                for value in values:
                    if value.lower() in phrase or phrase in value.lower():
                        negated.add(value.lower())
                    else:
                        remaining.append(value)
                if remaining:
                    self.active_slots[slot] = remaining
                else:
                    self._wipe_slot(slot)

            negated.update(token for token in phrase.split() if len(token) > 1)

        return negated

    def update_state(self, user_message: str, turn: int) -> Tuple[Dict[str, List[str]], int, int]:
        text = user_message.lower().strip()
        if not text:
            return self.active_slots, 0, 0

        is_reset = bool(RESET_ALL_PATTERN.search(text))
        is_override = bool(HARD_OVERRIDE_PATTERN.search(text))

        if is_reset:
            self._clear_preferences()

        captured_answer = self._capture_expected_answer(text)
        new_slots = self.extract_slots_regex(text)

        # Intent override = rewrite conflicting values, not erase all context.
        if is_override:
            match = OVERRIDE_VALUE_PATTERN.search(text)

            if match:
                override_value = self._clean(match.group(1))
                override_slots = self.extract_slots_regex(override_value)

                if override_slots:
                    for slot, values in override_slots.items():
                        self._store(slot, values, replace=True)
                        new_slots.pop(slot, None)
                elif override_value:
                    self._store("feature", [override_value], replace=True)

        negated = self._apply_negations(text)

        if negated:
            new_slots = {
                slot: [v for v in values if v.lower() not in negated]
                for slot, values in new_slots.items()
            }
            new_slots = {
                slot: values
                for slot, values in new_slots.items()
                if values
            }

        # "Actually" inside a hard override must not trigger normal soft-shift logic.
        is_soft_shift = bool(SOFT_SHIFT_PATTERN.search(text)) and not is_override

        if is_soft_shift:
            for slot in new_slots:
                self._wipe_slot(slot)

        for slot, values in new_slots.items():
            self._store(
                slot,
                values,
                replace=slot in SINGLE_VALUE_SLOTS or is_soft_shift,
            )
            self.no_preference_attributes.discard(slot)

        if self.expected_attribute and not captured_answer and new_slots:
            self.expected_attribute = None

        return self.active_slots, 0, 0

    def get_filtered_candidates(self) -> List[Dict[str, str]]:
        if not self.catalog:
            return []

        candidates = []
        for item in self.catalog:
            matched = True
            for slot, values in self.active_slots.items():
                if slot == "feature":
                    item_text = " ".join(str(item.get(f, "")) for f in ["title", "features", "details", "description"]).lower()
                else:
                    item_text = str(item.get(slot, "")).lower()

                if not any(v.lower() in item_text for v in values):
                    matched = False
                    break

            if matched:
                candidates.append(item)

        return candidates

    def _eligible_attributes(self) -> List[str]:
        return [
            attr for attr in QUESTION_PRIORITY
            if attr not in self.active_slots
            and attr not in self.asked_attributes
            and attr not in self.no_preference_attributes
        ]

    def _fallback_priority(self, eligible: List[str]) -> List[str]:
        """Base strategy dominates; profile preferences provide only a small bonus."""
        base_rank = {attr: i for i, attr in enumerate(QUESTION_PRIORITY)}
        profile_rank = {attr: i for i, attr in enumerate(self.profile_priority)}

        def score(attr: str) -> float:
            value = float(base_rank[attr])
            if attr in profile_rank:
                value -= 0.2 / (profile_rank[attr] + 1)
            return value

        return sorted(eligible, key=score)

    def get_next_ask_attribute(self) -> Optional[str]:
        eligible = self._eligible_attributes()
        if not eligible:
            self.expected_attribute = None
            return None

        candidates = self.get_filtered_candidates()

        # Agent currently doesn't pass catalog data to the tracker, so this path
        # will normally use the high-information fallback priority.
        if not candidates or len(candidates) <= 2:
            best = self._fallback_priority(eligible)[0]
            self.asked_attributes.add(best)
            self.expected_attribute = best
            return best

        # Optional candidate-aware entropy strategy if catalog is supplied later.
        best_attr, best_score = None, -1.0
        total = len(candidates)

        for attr in eligible:
            counts: Dict[str, int] = {}
            populated = 0

            for item in candidates:
                value = item.get(attr)
                if value in (None, "", [], {}):
                    continue

                populated += 1
                if isinstance(value, dict):
                    value = " ".join(f"{k} {v}" for k, v in value.items())
                elif isinstance(value, list):
                    value = " ".join(str(v) for v in value)

                value = str(value).lower()
                counts[value] = counts.get(value, 0) + 1

            if not populated or populated / total < 0.10:
                continue

            entropy = -sum((count / populated) * math.log2(count / populated) for count in counts.values())
            max_entropy = math.log2(max(2, len(counts)))
            score = (entropy / max_entropy if max_entropy else 0.0) * (populated / total)

            # Small bonus to preserve our high-information question prior.
            score += (len(QUESTION_PRIORITY) - QUESTION_PRIORITY.index(attr)) * 0.01

            if score > best_score:
                best_attr, best_score = attr, score

        if best_attr is None:
            best_attr = self._fallback_priority(eligible)[0]

        self.asked_attributes.add(best_attr)
        self.expected_attribute = best_attr
        return best_attr

    def build_search_query(self, category: Optional[str] = None) -> str:
        terms: List[str] = []

        if category:
            terms.append(category)
        elif "category" in self.active_slots:
            terms.extend(self.active_slots["category"])

        # Put discriminative constraints earlier in the query.
        for slot in ["feature", "material", "color", "style", "use_case", "brand", "size", "budget"]:
            terms.extend(self.active_slots.get(slot, []))

        return " ".join(dict.fromkeys(term.strip() for term in terms if term.strip()))