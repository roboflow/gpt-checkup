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
    question = "Can GPT classify an image without being trained on that particular use case?"
    prompt = "What is in the image? Return the class of the object in the image. Here are the classes: Toyota Camry, Tesla Model 3. You can only return one class from that list."
    image = "images/car.jpeg"
    method = "We check to see if the model can correctly identify the vehicle. If it can, it recieves a 100%, if it is incorrect, it recieves a 0%."

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
