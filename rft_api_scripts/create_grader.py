import requests
import os

# get the API key from environment
api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None, "OPENAI_API_KEY is not set"
headers = {"Authorization": f"Bearer {api_key}"}

grading_function = """
def parse_probability(response: str) -> int | None:
    response = response.strip()
    if response.endswith("%"):
        response = response[:-1]
    if not response.isdigit():
        return None
    probability = int(response)
    if probability < 0 or probability > 100:
        return None
    return probability

def grade(sample, item) -> float:
    output_text = sample["output_text"]
    reference_answer = int(item["reference_answer"])
    assert reference_answer in [0, 100]
    probability = parse_probability(output_text)
    if probability is None:
        return -0.2
    return -abs(probability - reference_answer) / 100.0 + 1.0
"""

grader = {"type": "python", "source": grading_function}

# print(json.dumps(grader, indent=2))

{
    "type": "python",
    "source": '\ndef parse_probability(response: str) -> int | None:\n    response = response.strip()\n    if response.endswith("%"):\n        response = response[:-1]\n    if not response.isdigit():\n        return None\n    probability = int(response)\n    if probability < 0 or probability > 100:\n        return None\n    return probability\n\ndef grade(sample, item) -> float:\n    output_text = sample["output_text"]\n    reference_answer = int(item["reference_answer"])\n    assert reference_answer in [0, 100]\n    probability = parse_probability(output_text)\n    if probability is None:\n        return -0.2\n    return -abs(probability - reference_answer) / 100.0 + 1.0\n',
}

# validate the grader
payload = {"grader": grader}
response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/validate",
    json=payload,
    headers=headers,
)
print("validate request_id:", response.headers["x-request-id"])
print("validate response:", response.text)

# run the grader with a test reference and sample
payload = {
    "grader": grader,
    "item": {"reference_answer": "100"},
    "model_sample": "100%",
}
response = requests.post(
    "https://api.openai.com/v1/fine_tuning/alpha/graders/run",
    json=payload,
    headers=headers,
)
print("run request_id:", response.headers["x-request-id"])
print("run response:", response.text)
