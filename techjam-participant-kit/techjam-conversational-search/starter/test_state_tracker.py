import unittest
from starter.state_tracker_rulebased import DialogueStateTracker


class TestDialogueStateTracker(unittest.TestCase):

    def setUp(self) -> None:
        """Initialize a fresh DialogueStateTracker before each test."""
        self.tracker = DialogueStateTracker()
        self.tracker.reset()

    def test_multi_turn_slot_accumulation(self) -> None:
        """Tests that slots accumulate across turns without overriding prematurely."""
        # Turn 1
        slots, _, _ = self.tracker.update_state("Looking for black jacket", turn=1)
        self.assertEqual(slots.get("category"), ["jacket"])
        self.assertEqual(slots.get("color"), ["black"])

        # Turn 2: Add material constraint
        slots, _, _ = self.tracker.update_state("I prefer leather material", turn=2)
        self.assertEqual(slots.get("category"), ["jacket"])
        self.assertEqual(slots.get("color"), ["black"])
        self.assertEqual(slots.get("material"), ["leather"])

        # Turn 3: Add size constraint
        slots, _, _ = self.tracker.update_state("Size medium please", turn=3)
        self.assertEqual(slots.get("category"), ["jacket"])
        self.assertEqual(slots.get("color"), ["black"])
        self.assertEqual(slots.get("material"), ["leather"])
        self.assertEqual(slots.get("size"), ["medium"])

    def test_targeted_slot_negation(self) -> None:
        """Tests that 'no red' or 'without leather' removes specific slot values."""
        # Setup initial state
        self.tracker.update_state("Show me red leather jackets", turn=1)
        self.assertIn("red", self.tracker.active_slots.get("color", []))
        self.assertIn("leather", self.tracker.active_slots.get("material", []))

        # Turn 2: Negate color
        slots, _, _ = self.tracker.update_state("No red ones, show me blue", turn=2)
        self.assertNotIn("red", slots.get("color", []))
        self.assertIn("blue", slots.get("color", []))
        self.assertIn("leather", slots.get("material", []))

        # Turn 3: Negate material
        slots, _, _ = self.tracker.update_state("Actually, without leather", turn=3)
        self.assertNotIn("material", slots)

    def test_category_shift_override(self) -> None:
        """Tests that shifting category ('actually show me boots instead') wipes dependent slots."""
        # Setup initial state with category, size, and color
        self.tracker.update_state("I want size large leather jackets in black", turn=1)
        self.assertEqual(self.tracker.active_slots.get("category"), ["jacket"])
        self.assertEqual(self.tracker.active_slots.get("size"), ["large"])

        # Turn 2: Category shift trigger
        slots, _, _ = self.tracker.update_state("Actually, show me boots instead", turn=2)

        # Primary category and category-dependent slots (size, material) should be wiped/updated
        self.assertEqual(slots.get("category"), ["boots"])
        self.assertNotIn("size", slots)
        self.assertNotIn("material", slots)
        # Non-conflicting attributes like color can persist if not overridden
        self.assertEqual(slots.get("color"), ["black"])

    def test_single_value_slot_overwrite(self) -> None:
        """Tests that single-value slots (category, budget, size) overwrite rather than accumulate."""
        # Set initial budget
        self.tracker.update_state("Show me shoes under 50", turn=1)
        self.assertEqual(self.tracker.active_slots.get("budget"), ["under $50"])

        # Update budget in turn 2
        slots, _, _ = self.tracker.update_state("Change my budget to under 100", turn=2)
        self.assertEqual(slots.get("budget"), ["under $100"])
        self.assertEqual(len(slots.get("budget")), 1)

    def test_full_reset_trigger(self) -> None:
        """Tests that explicit reset commands ('start over', 'clear all') wipe state."""
        self.tracker.update_state("Looking for blue cotton dress under 80", turn=1)
        self.assertTrue(len(self.tracker.active_slots) > 0)

        # Execute full reset
        slots, _, _ = self.tracker.update_state("Scratch that, start over", turn=2)
        self.assertEqual(len(slots), 0)

    def test_build_search_query_consistency(self) -> None:
        """Tests that search query construction reflects wiped and active slots accurately."""
        self.tracker.update_state("Looking for black leather jacket", turn=1)
        query = self.tracker.build_search_query()
        self.assertIn("jacket", query)
        self.assertIn("black", query)

        # Apply negation
        self.tracker.update_state("without leather", turn=2)
        updated_query = self.tracker.build_search_query()
        self.assertIn("jacket", updated_query)
        self.assertNotIn("leather", updated_query)


if __name__ == "__main__":
    unittest.main()