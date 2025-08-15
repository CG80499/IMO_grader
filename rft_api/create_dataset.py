import json
import random
from pydantic import BaseModel


class TheoremSample(BaseModel):
    theorem: str
    proof: str
    subtly_incorrect_proof: str
    difficulty_justification: str
    difficulty_score: int
    brainstorming: str


PROMPT = """Theorem: {theorem}

Proof:

{proof}

Is this proof correct? Give a probability between 0 and 100% that it is correct. Just give your probability estimate and nothing else."""

with open("theorem_samples_gpt4_1.json") as f:
    theorem_samples = json.load(f)
print(f"Loaded {len(theorem_samples)} samples")
theorem_samples_pydantic = [
    TheoremSample.model_validate(theorem_sample) for theorem_sample in theorem_samples
]

train_samples = []

# Do 150 samples
for sample in theorem_samples_pydantic[:150]:
    train_samples.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        theorem=sample.theorem, proof=sample.proof
                    ),
                }
            ],
            "reference_answer": "100",
        }
    )
    train_samples.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        theorem=sample.theorem, proof=sample.subtly_incorrect_proof
                    ),
                }
            ],
            "reference_answer": "0",
        }
    )

random.shuffle(train_samples)

with open("train_samples.jsonl", "w") as f:
    for sample in train_samples:
        f.write(json.dumps(sample) + "\n")

val_samples = []

for sample in theorem_samples_pydantic[150:]:
    val_samples.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        theorem=sample.theorem, proof=sample.proof
                    ),
                }
            ],
            "reference_answer": "100",
        }
    )
    val_samples.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        theorem=sample.theorem, proof=sample.subtly_incorrect_proof
                    ),
                }
            ],
            "reference_answer": "0",
        }
    )

random.shuffle(val_samples)

with open("val_samples.jsonl", "w") as f:
    for sample in val_samples:
        f.write(json.dumps(sample) + "\n")
