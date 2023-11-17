import base64
import datetime
import json
import os
import time
from dataclasses import dataclass
import re
from Levenshtein import ratio
from statistics import mean

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

    correct_equation = "3x^2-6x+2"

    accuracy = ratio(str(answer_equation).lower(), str(correct_equation).lower())
    return accuracy, inference_time, str(answer_equation)

def object_detection():
    base_model = GPT4V(
        ontology=CaptionOntology({"none": "none"}),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    result, inference_time = base_model.predict(
        "images/fruit.jpeg",
        classes=[],
        result_serialization="text",
        prompt="If there are banana in this image, return a JSON object with `x`, `y`, `width` and `height` properties of the banana. All values should be normalized between 0-1 and x&y should be the center point.",
    )

    code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
    code_blocks = re.findall(code_regex, result, re.DOTALL)
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

    return iou, inference_time, str(answer)

def set_of_mark():
  base_model = GPT4V(
      ontology=CaptionOntology({"none": "none"}),
      api_key=os.environ["OPENAI_API_KEY"],
  )

  result, inference_time = base_model.predict(
      "images/fruits_som.png",
      classes=[],
      result_serialization="text",
      prompt="Find all the fruits in this image and return a JSON array of all the applicable numbers.",
  )

  code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
  code_blocks = re.findall(code_regex,result, re.DOTALL)
  answer = json.loads(code_blocks[0])

  correct = [35,40,26,2,13,17,29,21,10,42,8,43,0,11,7,4,12,27,37,39,22,15,25]

  score = 0
  for guess in answer:
      if guess in correct: score += 1

  accuracy = score/len(correct)

  return accuracy, inference_time, result


tests = [
    "zero_shot_classification",
    "count_fruit",
    "document_ocr",
    "handwriting_ocr",
    "extraction_ocr",
    "math_ocr",
    "object_detection",
    # "set_of_mark"
]

current_results = {}
for i in tests:
    test = globals()[i]

    test_result = test()
    score, response_time, result = test_result

    score = (1 if score is True else (0 if score is False else score))

    current_results[i] = {}
    current_results[i]["score"] = score
    current_results[i]["response_time"] = response_time
    current_results[i]["result"] = result

print("current_results", current_results)

# save as today in 2023-01-01 format
# make results dir
if not os.path.exists("results"):
    os.mkdir("results")

today = datetime.datetime.now().strftime("%Y-%m-%d")

with open(f"results/{today}.json", "w+") as file:
    json.dump(current_results, file)

historical_results = {}
for i in tests:
    historical_results[i] = {}
    historical_results[i]["scores"] = []
    historical_results[i]["response_times"] = []

for file in os.listdir("results"):
    if os.path.isdir(f"results/{file}"): continue
    with open(f"results/{file}") as f:
        data = json.load(f)

        for key, value in data.items():
            print(key, value)
            historical_results[key]["scores"].append(value["score"])
            historical_results[key]["response_times"].append(value["response_time"])
            historical_results[key]["days"] = len(historical_results[key]["scores"])


print("historical_results", historical_results)

historical_averages = {}
for i in tests:
    historical_averages[i] = {}
    historical_averages[i]["scores"] = mean(historical_results[i]["scores"])
    historical_averages[i]["response_times"] = mean(historical_results[i]["response_times"])
    historical_averages[i]["success_percent"] = round(historical_averages[i]["scores"]*100,2)

print("historical_averages", historical_averages)

response_times = []
for i in historical_averages:
    response_times.append(historical_averages[i]["response_times"])

average_response_time = round(mean(response_times),2)
day_count = len(response_times)

results = {
    'current': current_results,
    'past': historical_results,
    'averages': historical_averages,
    'avg_time': average_response_time,
    'days': day_count
}

print(json.dumps(result, indent=4))

template = jinja2.Template(open("template.html").read())

today = datetime.datetime.now().strftime("%B %d, %Y")

# render template
rendered = template.render(results=results, date=today)

# save rendered template to index.html
with open("index.html", "w+") as file:
    file.write(rendered)
