from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re


class CountingTest:
    name = "Counting"
    id = "count_fruit"
    question = "Can GPT-4V count the number of objects within an image?"
    prompt = "Count the fruit in the image. Return a single number."
    image = "images/fruit.jpeg"
    method = "We send a picture of a bowl of fruit. If it correctly counts the number of fruit, it gets a 100%. Otherwise, it gets a 0%."

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