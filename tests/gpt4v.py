from dataclasses import dataclass
import supervision as sv
import base64
import time
import numpy as np

import os
import mimetypes
from pathlib import Path
import google.generativeai as genai

from autodistill.detection import CaptionOntology, DetectionBaseModel

@dataclass
class GPT4V(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology, api_key: str):
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name="gemini-pro-vision")
        self.ontology = ontology
        pass

    def predict(
        self,
        input,
        classes,
        result_serialization: str = "Classifications",
        prompt: str = None,
    ) -> sv.Classifications:
        if prompt is None:
            prompt = f"What is in the image? Return the class of the object in the image. Here are the classes: {', '.join(classes)}. You can only return one class from that list."

        image_path = input

        mime_type, encoding = mimetypes.guess_type(image_path)
        image = {
            "mime_type": mime_type,
            "data": Path(image_path).read_bytes()
        }

        payload = [
            prompt,
            image,
        ]

        model = genai.GenerativeModel(model_name="gemini-pro-vision")

        start_time = time.time()

        response = model.generate_content(payload)
        response_text = response.text

        inference_time = time.time() - start_time

        input_tokens = 0
        output_tokens = 0
        tokens = (input_tokens, output_tokens)

        if result_serialization == "Classifications":
            class_ids = self.ontology.prompts().index(
                response_text.strip()
            )

            return (
                sv.Classifications(
                    class_id=np.array([class_ids]),
                    confidence=np.array([1]),
                ),
                inference_time, tokens
            )
        else:
            return response_text, inference_time, tokens