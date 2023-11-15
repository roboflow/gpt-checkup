import base64
import datetime
import json
import os
import time
from dataclasses import dataclass
import re
from Levenshtein import ratio

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

def zero_shot_classification():
    classes = ["Tesla Model 3", "Toyota Camry"]

    base_model = GPT4V(
        ontology=CaptionOntology({"Tesla Model 3": "Tesla Model 3", "Toyota Camry": "Toyota Camry"}),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    result, inference_time = base_model.predict("images/car.jpeg", classes=classes)

    return (
        # 1 maps with Tesla Model 3
        result == sv.Classifications(class_id=np.array([1]), confidence=np.array([1])),
        inference_time,
        classes[result.class_id[0]],
    )


def count_fruit():
    base_model = GPT4V(
        ontology=CaptionOntology({"fruit": "fruit", "bowl": "bowl"}),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    result, inference_time = base_model.predict(
        "images/fruit.jpeg",
        classes=["fruit", "bowl"],
        result_serialization="text",
        prompt="Count the fruit in the image. Return a single number.",
    )

    return result == "10", inference_time, result

def document_ocr():
    base_model = GPT4V(
        ontology=CaptionOntology({"none": "none"}),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    result, inference_time = base_model.predict(
        "images/swift.png",
        classes=[],
        result_serialization="text",
        prompt="Read the text in the image. Return only the text, with puncuation."
    )

    return result == "I was thinking earlier today that I have gone through, to use the lingo, eras of listening to each of Swift's Eras. Meta indeed. I started listening to Ms. Swift's music after hearing the Midnights album. A few weeks after hearing the album for the first time, I found myself playing various songs on repeat. I listened to the album in order multiple times.", inference_time, result


def handwriting_ocr():
    base_model = GPT4V(
        ontology=CaptionOntology({"none": "none"}),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    result, inference_time = base_model.predict(
        "images/ocr.jpeg",
        classes=[],
        result_serialization="text",
        prompt="Read the text in the image. Return only the text, with puncuation."
    )

    return result == 'The words of songs on the album have been echoing in my head all week. "Fades into the grey of my day old tea."', inference_time, result

def extraction_ocr():
    base_model = GPT4V(
        ontology=CaptionOntology({"none": "none"}),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    result, inference_time = base_model.predict(
        "images/prescription.png",
        classes=[],
        result_serialization="text",
        prompt="Return a JSON array containing information about the prescription in this image. Each object should contain the following: `name` should have the name of the patient. `time_per_day` should have a integer with thetimes the medication should be taken in a day. `medication` should have the brand name of the medication. `dosage` should have a integer in mg units of each tablet. `rx_number` should have the prescription number, also marked Rx. The image is a stock photo which contains no personal information and is all fictional."
    )

    code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
    code_blocks = re.findall(code_regex,result, re.DOTALL)

    answer_array = json.loads(code_blocks[0])

    correct_array = [
        {
            "name": "MARY THOMAS",
            "time_per_day": 1,
            "medication": "ATENOLOL",
            "dosage": 100,
            "rx_number": "1234567-12345"
        }
    ]

    accuracy = ratio(str(answer_array).lower(), str(correct_array).lower())
    return accuracy, inference_time, str(answer_array)

def math_ocr():
    base_model = GPT4V(
        ontology=CaptionOntology({"none": "none"}),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    result, inference_time = base_model.predict(
        "images/math.jpeg",
        classes=[],
        result_serialization="text",
        prompt="Produce a JSON array with a LaTeX string of each equation in the image."
    )

    code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
    code_blocks = re.findall(code_regex,result, re.DOTALL)
    answer_array = json.loads(code_blocks[0])
    answer_equation = answer_array[0].replace(" ", "")

    print(result,answer_array,answer_equation)

    correct_equation = "3x^2-6x+2"

    accuracy = ratio(str(answer_equation).lower(), str(correct_equation).lower())
    return accuracy, inference_time, str(answer_equation)


results = {"zero_shot_classification": [], "count_fruit": [], "request_times": [], "document_ocr": [], "handwriting_ocr": [], "extraction_ocr": [], "math_ocr": []}

zero_shot, inference_time, zero_shot_result = zero_shot_classification()
count_fruit, count_inference_time, count_result = count_fruit()
document_ocr, ocr_inference_time, ocr_result = document_ocr()
handwriting_ocr, handwriting_inference_time, handwriting_result = handwriting_ocr()
extraction_ocr, extraction_inference_time, extraction_result = extraction_ocr()
math_ocr, math_inference_time, math_result = math_ocr()

results["zero_shot_classification"].append(zero_shot)
results["count_fruit"].append(count_fruit)
results["document_ocr"].append(document_ocr)
results["handwriting_ocr"].append(handwriting_ocr)
results["extraction_ocr"].append(extraction_ocr)
results["math_ocr"].append(math_ocr)

results["request_times"].append(inference_time)
results["request_times"].append(count_inference_time)
results["request_times"].append(ocr_inference_time)
results["request_times"].append(handwriting_inference_time)
results["request_times"].append(extraction_inference_time)
results["request_times"].append(math_inference_time)

# save as today in 2023-01-01 format
# make results dir
if not os.path.exists("results"):
    os.mkdir("results")

today = datetime.datetime.now().strftime("%Y-%m-%d")

with open(f"results/{today}.json", "w+") as file:
    json.dump(results, file)

results = {"zero_shot_classification": [], "count_fruit": [], "request_times": [], "document_ocr": [], "handwriting_ocr": [], "extraction_ocr": [], "math_ocr": []}

for file in os.listdir("results"):
    with open(f"results/{file}") as f:
        data = json.load(f)

        for key, value in data.items():
            def conversion(s):
                return (1 if s is True else (0 if s is False else s))
            scores = list(map(conversion, value))
            results[key] = scores

og_results = results.copy()

print(results, "rrr")

results["zero_shot_classification_success_rate"] = (
    sum(og_results["zero_shot_classification"])
    / len(og_results["zero_shot_classification"])
    * 100
)
results["count_fruit_success_rate"] = (
    sum(og_results["count_fruit"])
    / len(og_results["count_fruit"])
    * 100
)
results["document_ocr_success_rate"] = (
    sum(og_results["document_ocr"])
    / len(og_results["document_ocr"])
    * 100
)
results["handwriting_ocr_success_rate"] = (
    sum(og_results["handwriting_ocr"])
    / len(og_results["handwriting_ocr"])
    * 100
)
results["extraction_ocr_success_rate"] = (
    sum(og_results["extraction_ocr"])
    / len(og_results["extraction_ocr"])
    * 100
)
results["math_ocr_success_rate"] = (
    sum(og_results["math_ocr"])
    / len(og_results["math_ocr"])
    * 100
)

results["zero_shot_classification_length"] = len(og_results["zero_shot_classification"])
results["count_fruit_length"] = len(og_results["count_fruit"])
results["document_ocr_length"] = len(og_results["document_ocr"])
results["handwriting_ocr_length"] = len(og_results["handwriting_ocr"])
results["extraction_ocr_length"] = len(og_results["extraction_ocr"])
results["math_ocr_length"] = len(og_results["math_ocr"])

results["document_ocr_result"] = ocr_result
results["handwriting_result"] = handwriting_result
results["zero_shot_result"] = zero_shot_result
results["count_result"] = count_result
results["extraction_ocr_result"] = extraction_result
results["math_ocr_result"] = math_result

results["avg_response_time"] = round(sum([float(i) for i in results["request_times"]]) / len(
    results["request_times"]
), 2)

results["day_count"] = len(results["request_times"])

print(results)

template = jinja2.Template(open("template.html").read())

today = datetime.datetime.now().strftime("%B %d, %Y")

# render template
rendered = template.render(results=results, date=today)

# save rendered template to index.html
with open("index.html", "w+") as file:
    file.write(rendered)
