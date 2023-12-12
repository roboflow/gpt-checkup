from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re
import statistics


class MeasurementTest:
    name = "Measurement Test"
    id = "measurement"
    question = "Can GPT-4 Measure Items Using a Reference?"
    prompt = "What is the length and width of this square sticker, based on the ruler? Return a JSON with float properties for `length` and `width` representing inches."
    image = "images/measurement.jpg"
    method = "This test gives GPT-4 a image of a square sticker with a ruler on one side and asks it to provide a number for the length and width. We score this test based on precent error, gauging how far it is from the actual value."

    @staticmethod
    def test():
        base_model = GPT4V(
            ontology=CaptionOntology({"none": "none"}),
            api_key=os.environ["OPENAI_API_KEY"],
        )

        result, inference_time, tokens = base_model.predict(
            "images/measurement.jpg",
            classes=[],
            result_serialization="text",
            prompt="What is the length and width of this square sticker, based on the ruler? Return a JSON with float properties for `length` and `width` representing inches.",
        )

        code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
        code_blocks = re.findall(code_regex, result, re.DOTALL)
        if (len(code_blocks) == 0): 
            return 0, inference_time, f"Failed to produce a valid JSON output: {result}", tokens
        answer = json.loads(code_blocks[0])

        # 1 - Percent Error
        length_score = 1 - ( abs(3.5 - answer["length"])/3.5 )
        width_score = 1 - ( abs(3.5 - answer["width"])/3.5 )
        score = statistics.mean([length_score,width_score])

        return score, inference_time, str(result), tokens

