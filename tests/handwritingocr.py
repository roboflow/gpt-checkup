from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re


class HandwritingOCRTest:
    name = "Handwriting OCR"
    id = "handwriting_ocr"
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
            "images/ocr.jpeg",
            classes=[],
            result_serialization="text",
            prompt="Read the text in the image. Return only the text, with puncuation."
        )

        return (
            result == 'The words of songs on the album have been echoing in my head all week. "Fades into the grey of my day old tea."',
            inference_time,
            result,
            tokens
        )