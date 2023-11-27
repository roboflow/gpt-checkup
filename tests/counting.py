from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re


class CountingTest:
    name = "Counting"
    id = "count_fruit"
    question = ""
    prompt = ""
    image = ""
    method = ""

    @staticmethod
    def test():
        base_model = GPT4V(
            ontology=CaptionOntology({"fruit": "fruit", "bowl": "bowl"}),
            api_key=os.environ["OPENAI_API_KEY"],
        )

        result, inference_time, tokens = base_model.predict(
            "images/fruit.jpeg",
            classes=["fruit", "bowl"],
            result_serialization="text",
            prompt="Count the fruit in the image. Return a single number.",
        )

        return result == "10", inference_time, result, tokens