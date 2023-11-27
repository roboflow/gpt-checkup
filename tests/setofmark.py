from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re


class SetOfMarkTest:
    name = "Set of Mark"
    id = "set_of_mark"
    question = ""
    prompt = ""
    image = ""
    method = ""

    @staticmethod
    def test():
        base_model = GPT4V(
            ontology=CaptionOntology({"none": "none"}),
            api_key=os.environ["OPENAI_API_KEY"],
        )

        result, inference_time, tokens = base_model.predict(
            "images/fruits_som.png",
            classes=[],
            result_serialization="text",
            prompt="Find all the fruits in this image and return a JSON array of all the applicable numbers.",
        )

        code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
        code_blocks = re.findall(code_regex, result, re.DOTALL)
        if (len(code_blocks) == 0): 
            return 0, inference_time, f"Failed to produce a valid JSON output: {result}", tokens
        answer = json.loads(code_blocks[0])

        correct = [35,40,26,2,13,17,29,21,10,42,8,43,0,11,7,4,12,27,37,39,22,15,25]

        score = 0
        for guess in answer:
            if guess in correct: score += 1

        accuracy = score/len(correct)

        return accuracy, inference_time, str(answer), tokens