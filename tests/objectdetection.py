from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re


class ObjectDetectionTest:
    name = "Object Detection"
    id = "object_detection"
    question = "Can GPT detect objects in an image?"
    prompt = "If there are banana in this image, return a JSON object with `x`, `y`, `width` and `height` properties of the banana. All values should be normalized between 0-1 and x&y should be the center point."
    image = "images/fruit.jpeg"
    method = "We provide GPT with an image with a known object. We ask it to provide a normalized bounding box of the object and for scoring, we calculate the intersection over union (IOU) between the predicted bounding box and the correct bounding box."

    @staticmethod
    def test():
        base_model = GPT4V(
            ontology=CaptionOntology({"none": "none"}),
            api_key=os.environ["OPENAI_API_KEY"],
        )

        result, inference_time, tokens = base_model.predict(
            "images/fruit.jpeg",
            classes=[],
            result_serialization="text",
            prompt="If there are banana in this image, return a JSON object with `x`, `y`, `width` and `height` properties of the banana. All values should be normalized between 0-1 and x&y should be the center point.",
        )

        code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
        code_blocks = re.findall(code_regex, result, re.DOTALL)
        if (len(code_blocks) == 0): 
            return 0, inference_time, f"Failed to produce a valid JSON output: {result}", tokens
        answer = json.loads(code_blocks[0])

        correct = {'x': 0.465, 'y': 0.42, 'width': 0.37, 'height': 0.38}

        r1 = answer
        r2 = correct
        xi_min = max(r1['x'] - r1['width'] / 2, r2['x'] - r2['width'] / 2)
        yi_min = max(r1['y'] - r1['height'] / 2, r2['y'] - r2['height'] / 2)
        xi_max = min(r1['x'] + r1['width'] / 2, r2['x'] + r2['width'] / 2)
        yi_max = min(r1['y'] + r1['height'] / 2, r2['y'] + r2['height'] / 2)

        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        union_area = r1['width'] * r1['height'] + r2['width'] * r2['height'] - inter_area

        iou = inter_area / union_area if union_area else 0

        return iou, inference_time, str(answer), tokens
