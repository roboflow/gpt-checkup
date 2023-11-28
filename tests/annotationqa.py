from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re


class AnnotationQATest:
    name = "Annotation Quality Assurance"
    id = "annotation_qa"
    question = "Can GPT-4V identify image labeling mistakes?"
    prompt = "This is a sample image from a dataset with cars labeled with red bounding boxes. Are there any missing annotations? Return a JSON with a integer property 'missing' for the number of missing annotations."
    image = "images/annotationqa.jpeg"
    method = "We provide a image from a self driving car dataset with intentionally three missing annotations. We ask GPT-4V to identify the number of missing annotations. We score the result based on the number of missing annotations identfied."

    @staticmethod
    def test():
        base_model = GPT4V(
            ontology=CaptionOntology({"none": "none"}),
            api_key=os.environ["OPENAI_API_KEY"],
        )

        result, inference_time, tokens = base_model.predict(
            "images/annotationqa.jpeg",
            classes=[],
            result_serialization="text",
            prompt="This is a sample image from a dataset with cars labeled with red bounding boxes. Are there any missing annotations? Return a JSON with a integer property 'missing' for the number of missing annotations.",
        )

        code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
        code_blocks = re.findall(code_regex, result, re.DOTALL)
        if (len(code_blocks) == 0): 
            return 0, inference_time, f"Failed to produce a valid JSON output: {result}", tokens
        answer = json.loads(code_blocks[0])

        correct = 3

        diff = abs(correct - answer["missing"])

        score = 1 - (diff / correct)

        return score, inference_time, str(result), tokens

