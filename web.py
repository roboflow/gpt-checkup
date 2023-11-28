import datetime
import json
import os
from statistics import mean
import tests
import importlib
import jinja2

HOME = os.path.expanduser("~")

test_list = [
    "ZeroShotClassificationTest",
    "CountingTest",
    "DocumentOCRTest",
    "HandwritingOCRTest",
    "ExtractionOCRTest",
    "MathOCRTest",
    "ObjectDetectionTest",
    "SetOfMarkTest",
    "GraphUnderstandingTest"
]

test_ids = []

current_results = {}

# Run tests
# for i in test_list:
#     test_info = getattr(importlib.import_module(f"tests"),i)
#     print(f"Running {test_info.name} test...")

#     test_id = test_info.id
#     test_ids.append(test_id)

#     test_result = test_info.test()
#     score, response_time, result, tokens = test_result

#     input_token_price = 0.01/1000
#     output_token_price = 0.03/1000
#     price = (input_token_price * tokens[0]) + (output_token_price * tokens[1])
#     score = (1 if score is True else (0 if score is False else score))

#     current_results[test_id] = {}
#     current_results[test_id]["score"] = score
#     current_results[test_id]["success"] = score == 1
#     current_results[test_id]["price"] = price
#     current_results[test_id]["pass_fail"] = "Pass" if score == 1 else "Fail"
#     current_results[test_id]["response_time"] = response_time
#     current_results[test_id]["result"] = result

# print("current_results", current_results)

# save as today in 2023-01-01 format
# make results dir
if not os.path.exists("results"):
    os.mkdir("results")

today = datetime.datetime.now().strftime("%Y-%m-%d")

# with open(f"results/{today}.json", "w+") as file:
#     json.dump(current_results, file, indent=4)


# Results processing

if (current_results == {}) and (os.path.exists(f"results/{today}.json")):
    with open(f"results/{today}.json") as file:
        current_results = json.load(file)
        test_ids = list(current_results.keys())
else:
    print("No current results and no file found")

results = {}

for index, test_id in enumerate(test_ids):
    results[test_id] = {}

    test_info = getattr(importlib.import_module(f"tests"),test_list[index])
    results[test_id]["name"] = test_info.name
    results[test_id]["question"] = test_info.question
    results[test_id]["prompt"] = test_info.prompt
    results[test_id]["image"] = test_info.image
    results[test_id]["method"] = test_info.method

for i in test_ids:
    results[i]["history"] = {}
    results[i]["history"]["scores"] = []
    results[i]["history"]["response_times"] = []

for file in os.listdir("results"):
    if os.path.isdir(f"results/{file}"): continue
    with open(f"results/{file}") as f:
        data = json.load(f)

        for key, value in data.items():
            print(key, value)
            if results.get(key) is None: continue
            results[key]["history"]["scores"].append(value["score"])
            results[key]["history"]["response_times"].append(value["response_time"])
            results[key]["history"]["days"] = len(results[key]["history"]["scores"])

for i in test_ids:
    results[i]["average"] = {}
    results[i]["average"]["score"] = mean(results[i]["history"]["scores"])
    results[i]["average"]["response_time"] = mean(results[i]["history"]["response_times"])
    results[i]["average"]["success_percent"] = round(results[i]["average"]["score"]*100,2)

response_times = []
for i in test_ids:
    response_times.append(results[i]["average"]["response_time"])

print("response_times", response_times, test_ids)
info = {}
info["average_time"] = round(mean(response_times), 2)
info["day_count"] = len(response_times)

print("- - - - -")
print(json.dumps(results, indent=4))
print("- - - - -")
print(json.dumps(current_results, indent=4))

template = jinja2.Template(open("template.html").read())

today = datetime.datetime.now().strftime("%B %d, %Y")

# render template
rendered = template.render(results=results, date=today, current_results=current_results, info=info)

# save rendered template to index.html
with open("index.html", "w+") as file:
    file.write(rendered)
