from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re


class ColorRecognitionTest:
    name = "Color Recognition"
    id = "color_recognition"
    question = "Can GPT identify colors accurately?"
    prompt = "Guess the RGB color code of the rectangle and return only the result in JSON. The JSON should have three integer properties: 'R', 'G' and 'B'"
    image = "images/color.png"
    method = "We provide GPT with an image with multiple shapes with differing colors. We ask it to identify the color of a particular shape in RGB color codes."

    @staticmethod
    def test():
        base_model = GPT4V(
            ontology=CaptionOntology({"none": "none"}),
            api_key=os.environ["OPENAI_API_KEY"],
        )

        result, inference_time, tokens = base_model.predict(
            "images/color.png",
            classes=[],
            result_serialization="text",
            prompt="Guess the RGB color code of the rectangle and return only the result in JSON. The JSON should have three integer properties: 'R', 'G' and 'B'",
        )

        code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
        code_blocks = re.findall(code_regex, result, re.DOTALL)
        if (len(code_blocks) == 0): 
            return 0, inference_time, f"Failed to produce a valid JSON output: {result}", tokens
        answer = json.loads(code_blocks[0])

        correct = {"R": 77, "G": 4, "B": 154}

        r_diff = abs(answer['R'] - correct['R'])
        g_diff = abs(answer['G'] - correct['G'])
        b_diff = abs(answer['B'] - correct['B'])

        max_diff = 255 * 3
        total_diff = r_diff + g_diff + b_diff

        score = 1 - (total_diff / max_diff)

        return score, inference_time, str(result), tokens
