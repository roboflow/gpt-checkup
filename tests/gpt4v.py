from dataclasses import dataclass
import supervision as sv
from openai import OpenAI
import base64
import time
import numpy as np

from autodistill.detection import CaptionOntology, DetectionBaseModel

@dataclass
class GPT4V(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology, api_key: str):
        self.client = OpenAI(api_key=api_key)
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

        payload = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,"
                            + base64.b64encode(open(input, "rb").read()).decode(
                                "utf-8"
                            ),
                        },
                    },
                ],
            }
        ]

        start_time = time.time()

        response = self.client.chat.completions.create(
            model="gpt-o3-mini",
            messages=payload,
            max_tokens=300,
        )

        inference_time = time.time() - start_time

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        tokens = (input_tokens, output_tokens)

        if result_serialization == "Classifications":
            class_ids = self.ontology.prompts().index(
                response.choices[0].message.content
            )

            return (
                sv.Classifications(
                    class_id=np.array([class_ids]),
                    confidence=np.array([1]),
                ),
                inference_time, tokens
            )
        else:
            return response.choices[0].message.content, inference_time, tokens
