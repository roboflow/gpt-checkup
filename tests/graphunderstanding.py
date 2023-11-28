from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re


class GraphUnderstandingTest:
    name = "Graph Understanding"
    id = "graph_understanding"
    question = "Can GPT-4V identify points on a graph?"
    prompt = "State positions of points A through D in a JSON with properties A-D, each having a object with properties for integers matching the respective point: `quantity` and `price`."
    image = "images/graph.png"
    method = "We send a picuture of a graph with four labeled points and ask GPT-4V to identify the points. This test is scored by the accuracy of each point. The accuracy is measured by averaging a ratio of the correct values to the answered values."

    @staticmethod
    def test():
        base_model = GPT4V(
            ontology=CaptionOntology({"none": "none"}),
            api_key=os.environ["OPENAI_API_KEY"],
        )

        result, inference_time, tokens = base_model.predict(
            "images/graph.png",
            classes=[],
            result_serialization="text",
            prompt="State positions of points A through D in a JSON with properties A-D, each having a object with properties for integers matching the respective point: `quantity` and `price`.",
        )

        code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
        code_blocks = re.findall(code_regex, result, re.DOTALL)
        if (len(code_blocks) == 0): 
            return 0, inference_time, f"Failed to produce a valid JSON output: {result}", tokens
        answer = json.loads(code_blocks[0])

        correct = {
            "A": {
                "quantity": 20,
                "price": 10
            },
            "B": {
                "quantity": 26,
                "price": 20
            },
            "C": {
                "quantity": 30,
                "price": 30
            },
            "D": {
                "quantity": 34,
                "price": 40
            }
        }

        total_scores = 0
        count = 0

        for letter in 'ABCD':
            if letter in correct and letter in answer:
                quantity_diff = abs(correct[letter]['quantity'] - answer[letter]['quantity'])
                quantity_score = max(0, 1 - (quantity_diff / 25))

                price_diff = abs(correct[letter]['price'] - answer[letter]['price'])
                price_score = max(0, 1 - (price_diff / 25))

                total_scores += (quantity_score + price_score)
                count += 2

        print(total_scores / count)
        score = total_scores / count

        return score, inference_time, str(result), tokens