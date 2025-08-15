import json
from collections import Counter

from naturalproofs_models import NaturalProofs
from naturalproofs_models import Theorem
from utils import map_threaded
from utils import extract_xml_tag
from pydantic import BaseModel


from llm.core import TextChat
from llm.core import TextUserMessage
from llm.openai import OpenAI
from llm.core import BaseLLM
import os
from dotenv import load_dotenv

load_dotenv()


class TheoremSample(BaseModel):
    theorem: str
    proof: str
    subtly_incorrect_proof: str
    difficulty_justification: str
    difficulty_score: int
    brainstorming: str


PROMPT = """You will be given a theorem and its proof in LaTeX format. Your task is to introduce subtle mistakes into the proof. These mistakes should be extremely difficult to detect, such that only the most expert mathematicians would be likely to notice them. The goal is to create a challenging exam for advanced postgraduate students.

Here is the original theorem and proof:

<theorem>
{theorem}
</theorem>

<proof>
{proof}
</proof>

Please follow these guidelines when introducing mistakes:

1. The mistakes should be very subtle and not immediately obvious.
2. Errors can include slight modifications to definitions, assumptions, or logical steps.
3. Ensure that the overall structure and flow of the proof remain intact.
4. The mistakes should not make the proof obviously incorrect.

After introducing the mistakes, present the modified proof in LaTeX format, enclosed in <modified_proof> tags.

Next, assess the difficulty of the original proof (before introducing mistakes) on a scale of 1 to 5, where:
1 = Simple lemma
2 = Straightforward proof with few steps
3 = Moderately challenging proof
4 = Complex proof with multiple steps and techniques
5 = Very challenging proof with many steps and advanced concepts

Provide a brief justification for your difficulty rating, considering factors such as the complexity of the concepts involved, the number of steps in the proof, and the level of mathematical maturity required to understand and construct the proof.

Present your response in the following format:

<brainstorming>
[Insert your brainstorming process here. This should be a for how you subtly modified the proof to be incorrect.]
</brainstorming>

<modified_proof>
[Insert the modified proof with subtle mistake(s) here. Only give the proof with no additional tags]
</modified_proof>

<difficulty_justification>
[Insert your justification for the difficulty rating here]
</difficulty_justification>

<difficulty_score>
[Insert the numerical difficulty score (1-5) here]
</difficulty_score>"""


def parse_model_response(response: str, theorem: Theorem) -> TheoremSample | None:
    subtly_incorrect_proof = extract_xml_tag(response, "modified_proof")
    difficulty_justification = extract_xml_tag(response, "difficulty_justification")
    difficulty_score = extract_xml_tag(response, "difficulty_score")
    brainstorming = extract_xml_tag(response, "brainstorming")
    if (
        subtly_incorrect_proof is None
        or difficulty_justification is None
        or difficulty_score is None
        or brainstorming is None
    ):
        return None
    if not difficulty_score.strip().isdigit():
        return None
    difficulty_score = int(difficulty_score.strip())
    if difficulty_score < 1 or difficulty_score > 5:
        return None
    return TheoremSample(
        theorem=theorem.title,
        proof=theorem.proofs[0].text,
        subtly_incorrect_proof=subtly_incorrect_proof,
        difficulty_justification=difficulty_justification,
        difficulty_score=difficulty_score,
        brainstorming=brainstorming,
    )


def make_theorem_sample(llm: BaseLLM, theorem: Theorem) -> TheoremSample | None:
    if not theorem.proofs:
        return None
    chat = TextChat(
        messages=[
            TextUserMessage(
                content=PROMPT.format(
                    theorem=theorem.title, proof=theorem.proofs[0].text
                )
            )
        ]
    )
    raw_response = llm.predict(chat, max_tokens=10_000)
    return parse_model_response(raw_response, theorem)


def parse_naturalproofs_dataset(json_file_path: str) -> list[Theorem]:
    with open(json_file_path, "r") as f:
        data = json.load(f)

    naturalproofs = NaturalProofs.validate(data)
    return naturalproofs.dataset.theorems


def make_theorem_samples(llm: BaseLLM, theorems: list[Theorem]) -> list[TheoremSample]:
    return list(
        filter(
            None,
            map_threaded(
                lambda theorem: make_theorem_sample(llm, theorem),
                theorems,
                max_concurrency=10,
                show_progress=True,
            ),
        )
    )


if __name__ == "__main__":
    theorems = parse_naturalproofs_dataset("naturalproofs_proofwiki.json")
    print(f"Loaded {len(theorems)} theorems")

    print(f"\nFirst theorem: {theorems[2].title}")
    print(f"Categories: {theorems[2].categories}")
    print(f"Content preview: {theorems[2].contents[0][:100]}...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    org_id = os.getenv("OPENAI_ORG_ID")

    llm = OpenAI(model="gpt-4o-2024-08-06", api_key=api_key, org_id=org_id)

    theorem_samples = make_theorem_samples(llm, theorems[:2000])
    print(f"Made {len(theorem_samples)} theorem samples")
    score_counts = Counter(ts.difficulty_score for ts in theorem_samples)
    total = len(theorem_samples)
    print("Percentage of proofs with each difficulty score:")
    for score in range(1, 6):
        count = score_counts.get(score, 0)
        percent = (count / total * 100) if total > 0 else 0
        print(f"  Score {score}: {count} ({percent:.1f}%)")
    with open("theorem_samples_gpt4_1.json", "w") as f:
        json.dump(
            [theorem_sample.model_dump() for theorem_sample in theorem_samples], f
        )

    print(
        f"The total cost of this run is ${llm.total_cost()} [if not cached]\n\nBreakdown by model:"
    )


# Made 1861 theorem samples
# Percentage of proofs with each difficulty score:
#   Score 1: 43 (2.3%)
#   Score 2: 554 (29.8%)
#   Score 3: 923 (49.6%)
#   Score 4: 323 (17.4%)
#   Score 5: 18 (1.0%)
