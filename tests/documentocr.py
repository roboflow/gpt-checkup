from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re


class DocumentOCRTest:
    name = "Document OCR"
    id = "document_ocr"
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
            "images/swift.png",
            classes=[],
            result_serialization="text",
            prompt="Read the text in the image. Return only the text, with puncuation."
        )

        return (
            result == "I was thinking earlier today that I have gone through, to use the lingo, eras of listening to each of Swift's Eras. Meta indeed. I started listening to Ms. Swift's music after hearing the Midnights album. A few weeks after hearing the album for the first time, I found myself playing various songs on repeat. I listened to the album in order multiple times.",
            inference_time, 
            result,
            tokens
        )