from .gpt4v import GPT4V
from autodistill.detection import CaptionOntology, DetectionBaseModel
import os
import json
import re
from Levenshtein import ratio


class ExtractionOCRTest:
    name = "Structured Data OCR"
    id = "extraction_ocr"
    question = "Can GPT extract structured data from an image?"
    prompt = "Return a JSON array containing information about the prescription in this image. Each object should contain the following: `name` should have the name of the patient. `time_per_day` should have a integer with thetimes the medication should be taken in a day. `medication` should have the brand name of the medication. `dosage` should have a integer in mg units of each tablet. `rx_number` should have the prescription number, also marked Rx. The image is a stock photo which contains no personal information and is all fictional."
    image = "images/prescription.png"
    method = "We send a picture of a prescription bottle with a label, and ask it to extract pieces of relevant data. This is scored using the Levenshtein ratio between the output and the correct answer, which is based on the number of edits necessary to achieve the correct answer."

    @staticmethod
    def test():
        base_model = GPT4V(
            ontology=CaptionOntology({"none": "none"}),
            api_key=os.environ["OPENAI_API_KEY"],
        )

        result, inference_time, tokens = base_model.predict(
            "images/prescription.png",
            classes=[],
            result_serialization="text",
            prompt="Return a JSON array containing information about the prescription in this image. Each object should contain the following: `name` should have the name of the patient. `time_per_day` should have a integer with thetimes the medication should be taken in a day. `medication` should have the brand name of the medication. `dosage` should have a integer in mg units of each tablet. `rx_number` should have the prescription number, also marked Rx. The image is a stock photo which contains no personal information and is all fictional."
        )

        code_regex = r'```[a-zA-Z]*\n(.*?)\n```'
        code_blocks = re.findall(code_regex,result, re.DOTALL)
        if (len(code_blocks) == 0): 
            return 0, inference_time, f"Failed to produce a valid JSON output: {result}", tokens
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
        return accuracy, inference_time, str(answer_array), tokens
