import unittest
from PIL import Image
from src.image_engine import interpret_prompt, local_edit, scenario_count

class ScenarioEngineTests(unittest.TestCase):
    def test_library_is_substantial(self):
        self.assertGreaterEqual(scenario_count(), 100)

    def test_combines_thriller_and_fog(self):
        result = interpret_prompt("make it a dark psychological thriller with dense fog")
        self.assertTrue(result["recognized"])
        self.assertGreater(result["effects"]["fog"], 0.2)
        self.assertTrue(any("thriller" in x["id"] for x in result["matched"]))

    def test_vintage_prompt_matches(self):
        result = interpret_prompt("make this look like a nostalgic 1990s film photograph")
        self.assertTrue(result["recognized"])
        self.assertTrue(result["matched"])

    def test_local_edit_returns_rgb_image(self):
        image = Image.new("RGB", (128, 96), (120, 140, 160))
        edited, label = local_edit(image, "cold foggy psychological thriller")
        self.assertEqual(edited.mode, "RGB")
        self.assertEqual(edited.size, image.size)
        self.assertIn("local recipes", label)

    def test_unknown_prompt_is_honest(self):
        result = interpret_prompt("make it look like a quizzical flerm")
        self.assertFalse(result["recognized"])

if __name__ == "__main__":
    unittest.main()
