import base64
import datetime
import json
import os
import time
from dataclasses import dataclass

import jinja2
import numpy as np
import supervision as sv
from openai import OpenAI

from autodistill.detection import CaptionOntology, DetectionBaseModel

HOME = os.path.expanduser("~")


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
            model="gpt-4-vision-preview",
            messages=payload,
            max_tokens=300,
        )

        inference_time = time.time() - start_time

        if result_serialization == "Classifications":
            class_ids = self.ontology.prompts().index(
                response.choices[0].message.content
            )

            return (
                sv.Classifications(
                    class_id=np.array([class_ids]),
                    confidence=np.array([1]),
                ),
                inference_time,
            )
        else:
            return response.choices[0].message.content, inference_time


base_model = GPT4V(
    ontology=CaptionOntology({"fruit": "fruit", "bowl": "bowl"}),
    api_key=os.environ["OPENAI_API_KEY"],
)


def zero_shot_classification():
    result, inference_time = base_model.predict("images/fruit.jpeg", classes=["fruit", "bowl"])

    return (
        result == sv.Classifications(class_id=np.array([0]), confidence=np.array([1])),
        inference_time,
    )


def count_fruit():
    result, inference_time = base_model.predict(
        "images/fruit.jpeg",
        classes=["fruit", "bowl"],
        result_serialization="text",
        prompt="Count the fruit in the image. Return a single number.",
    )

    return result == "6", inference_time


results = {"zero_shot_classification": [], "count_fruit": [], "request_times": []}

zero_shot, inference_time = zero_shot_classification()
count_fruit, count_inference_time = count_fruit()

results["zero_shot_classification"].append(zero_shot)
results["count_fruit"].append(count_fruit)

results["request_times"].append(inference_time)
results["request_times"].append(count_inference_time)

# save as today in 2023-01-01 format
# make results dir
if not os.path.exists("results"):
    os.mkdir("results")

today = datetime.datetime.now().strftime("%Y-%m-%d")

with open(f"results/{today}.json", "w+") as file:
    json.dump(results, file)

results = {"zero_shot_classification": [], "count_fruit": [], "request_times": []}

for file in os.listdir("results"):
    with open(f"results/{file}") as f:
        data = json.load(f)

        for key, value in data.items():
            results[key].append(value)

og_results = results.copy()

results = {
    k: ",".join([str(1 if v else 0) for v in value]) for k, value in results.items()
}

results["zero_shot_classification_success_rate"] = (
    sum([1 if i else 0 for i in og_results["zero_shot_classification"]])
    / len(og_results["zero_shot_classification"])
    * 100
)
results["count_fruit_success_rate"] = (
    sum([1 if i else 0 for i in og_results["count_fruit"]])
    / len(og_results["count_fruit"])
    * 100
)
results["zero_shot_classification_length"] = len(og_results["zero_shot_classification"])
results["count_fruit_length"] = len(og_results["count_fruit"])

results["avg_response_time"] = sum([float(i) for i in results["request_times"]]) / len(
    results["request_times"]
)
results["day_count"] = len(results["request_times"])

print(results)

template = jinja2.Template(open("template.html").read())

today = datetime.datetime.now().strftime("%B %d, %Y")

# render template
rendered = template.render(results=results, date=today)

# save rendered template to index.html
with open("index.html", "w+") as file:
    file.write(rendered)
