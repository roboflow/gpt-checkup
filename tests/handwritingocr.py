from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re


class HandwritingOCRTest:
    name = "Handwriting OCR"
    id = "handwriting_ocr"
    question = "Can GPT read handwriting?"
    prompt = "Read the text in the image. Return only the text, with punctuation."
    image = "images/ocr.jpeg"
    method = "We send a image of a handwritten note to determine if it can correctly read the text. If it correctly gets the text, it gets a 100%. Otherwise, it gets a 0%."

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
