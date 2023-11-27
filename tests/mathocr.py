from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re
from Levenshtein import ratio


class MathOCRTest:
    name = "Math OCR"
    id = "math_ocr"

    @staticmethod
    def test():
        base_model = GPT4V(
            ontology=CaptionOntology({"none": "none"}),
            api_key=os.environ["OPENAI_API_KEY"],
        )

        result, inference_time, tokens = base_model.predict(
            "images/math.jpeg",
            classes=[],
            result_serialization="text",
            prompt="Produce a JSON array with a LaTeX string of each equation in the image."
        )

        code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
        code_blocks = re.findall(code_regex,result, re.DOTALL)
        if (len(code_blocks) == 0): 
            return 0, inference_time, f"Failed to produce a valid JSON output: {result}", tokens
        answer_array = json.loads(code_blocks[0])
        answer_equation = answer_array[0].replace(" ", "")

        correct_equation = "3x^2-6x+2"

        accuracy = ratio(str(answer_equation).lower(), str(correct_equation).lower())
        return accuracy, inference_time, str(answer_equation), tokens