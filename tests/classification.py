from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re
import supervision as sv
import numpy as np


class ZeroShotClassificationTest:
    name = "Zero Shot Classification"
    id = "zero_shot_classification"
    question = "Can GPT-4V count the number of objects in an image?"
    prompt = "Count the fruit in the image. Return a single number."
    image = "images/car.jpeg"
    method = "For evaluating this test, we check to see if the model can correctly count the number of fruits. If it can, it recieves a 100%, if it is incorrect, it recieves a 0%."

    @staticmethod
    def test():
        classes = ["Tesla Model 3", "Toyota Camry"]

        base_model = GPT4V(
            ontology=CaptionOntology({"Tesla Model 3": "Tesla Model 3", "Toyota Camry": "Toyota Camry"}),
            api_key=os.environ["OPENAI_API_KEY"],
        )

        result, inference_time, tokens = base_model.predict("images/car.jpeg", classes=classes)

        return (
            # 1 maps with Tesla Model 3
            result == sv.Classifications(class_id=np.array([1]), confidence=np.array([1])),
            inference_time,
            classes[result.class_id[0]],
            tokens
        )