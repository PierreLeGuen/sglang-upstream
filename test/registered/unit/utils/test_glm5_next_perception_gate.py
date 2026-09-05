"""Verify that the synthetic vision gate accepts correct descriptions only."""

import importlib.util
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

path = Path(__file__).resolve().parents[3] / "manual/test_glm5_next_perception.py"
spec = importlib.util.spec_from_file_location("glm5_perception_gate", path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class TestPerceptionGate(unittest.TestCase):
    def test_accepts_correct_band_descriptions_without_extra_word_requirements(self):
        for answer in (
            "The image shows a white rectangle centered on a black background.",
            "A white horizontal stripe sits between black areas.",
            "A black background with a white horizontal rectangle in the center.",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(gate.matches(answer, "band"))

    def test_rejects_wrong_content_and_explicit_wrong_orientation(self):
        for answer in (
            "A red circle on a white background.",
            "A black horizontal rectangle on a white background.",
            "A white vertical rectangle on a black background.",
            "A white square on a black background.",
            "A white stripe on a blue background.",
            "OK",
        ):
            with self.subTest(answer=answer):
                self.assertFalse(gate.matches(answer, "band"))

    def test_counts_and_left_right_remain_strict(self):
        self.assertTrue(gate.matches("2", "2"))
        self.assertFalse(gate.matches("1", "2"))
        self.assertTrue(gate.matches("left=red, right=blue", "halves"))
        self.assertFalse(gate.matches("left=blue, right=red", "halves"))


if __name__ == "__main__":
    unittest.main()
