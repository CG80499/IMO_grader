import json
import os
from typing import Any, TypeGuard

from pydantic import BaseModel

from utils import map_threaded
from llm.core import TextChat, TextUserMessage, BaseLLM
from llm.openai import OpenAI, OpenAIReasoning
from clients.openai import OpenAIReasoningModelName, OpenAIChatModelName
from dotenv import load_dotenv

load_dotenv()


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

THINK_STEP_BY_STEP_SUFFIX = """Before you give your final probability estimate, please think step by step in the <think> tags. I.e

<think>
[Your thinking for the probability estimate that this proof is correct]
</think>

[Your probability estimate from 0 to 100]%
""".strip()


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


def reasoning_chat_model(model: str) -> TypeGuard[OpenAIReasoningModelName]:
    return model in [
        "o3-2025-04-16",
        "o4-mini-2025-04-16",
        "ft:o4-mini-2025-04-16:elicit:proof-v0:Bwejms4L:ckpt-step-9",
        "ft:o4-mini-2025-04-16:elicit:proof-v1:BzGz36t7:ckpt-step-10",
        "ft:o4-mini-2025-04-16:elicit:proof-v1:BzRIloFv:ckpt-step-17",
    ]


def chat_model(model: str) -> TypeGuard[OpenAIChatModelName]:
    return not reasoning_chat_model(model)


def evaluate_theorem_sample(
    llm: BaseLLM,
    theorem_sample: TheoremSample,
    model: OpenAIReasoningModelName | OpenAIChatModelName,
) -> tuple[bool, bool]:
    chat = TextChat(
        messages=[
            TextUserMessage(
                content=PROMPT.format(
                    theorem=theorem_sample.theorem, proof=theorem_sample.proof
                )
            )
        ]
    )
    if reasoning_chat_model(model):
        raw_response = llm.predict(chat, max_tokens=32000)
    elif chat_model(model):
        chat.messages[0].content += THINK_STEP_BY_STEP_SUFFIX
        raw_response = llm.predict(chat, max_tokens=4000, temperature=1.0)
        if "</think>" in raw_response:
            raw_response = raw_response.split("</think>")[1]
        raw_response = raw_response.strip()
    else:
        raise ValueError(f"Invalid model: {model}")

    correct_when_proof_is_correct = (parse_probability(raw_response) or 0) >= 50

    chat = TextChat(
        messages=[
            TextUserMessage(
                content=PROMPT.format(
                    theorem=theorem_sample.theorem,
                    proof=theorem_sample.subtly_incorrect_proof,
                )
            )
        ]
    )
    if reasoning_chat_model(model):
        raw_response = llm.predict(chat, max_tokens=32000)
    elif chat_model(model):
        chat.messages[0].content += THINK_STEP_BY_STEP_SUFFIX
        raw_response = llm.predict(chat, max_tokens=4000, temperature=1.0)
        if "</think>" in raw_response:
            raw_response = raw_response.split("</think>")[1]
        raw_response = raw_response.strip()
    else:
        raise ValueError(f"Invalid model: {model}")

    parsed = parse_probability(raw_response)
    correct_when_proof_is_incorrect = (
        (parsed or 0) < 50 if parsed is not None else False
    )
    return correct_when_proof_is_correct, correct_when_proof_is_incorrect


def run_model_on_dataset(
    llm: BaseLLM,
    theorem_samples: list[TheoremSample],
    model: OpenAIReasoningModelName | OpenAIChatModelName,
) -> list[bool]:
    results = map_threaded(
        lambda theorem_sample: evaluate_theorem_sample(llm, theorem_sample, model),
        theorem_samples,
        max_concurrency=10,
        show_progress=True,
    )
    return [result for sublist in results for result in sublist]


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    org_id = os.getenv("OPENAI_ORG_ID")

    models: list[OpenAIReasoningModelName | OpenAIChatModelName] = [
        "o4-mini-2025-04-16",
        # Add more model names here as desired, e.g.:
        # "o3-2025-04-16",
        # "gpt-4.1-2025-04-14",
        # "ft:o4-mini-2025-04-16:elicit:proof-v1:BzGz36t7:ckpt-step-10",
    ]

    all_results: list[dict[str, Any]] = []
    summary_stats: list[dict[str, Any]] = []
    cost_by_model: dict[str, float] = {}

    with open("theorem_samples_gpt4_1.json") as f:
        theorem_samples = json.load(f)[:500]
    print(f"Loaded {len(theorem_samples)} samples")
    theorem_samples = [
        TheoremSample.validate(theorem_sample) for theorem_sample in theorem_samples
    ]
    print(f"Running on {len(theorem_samples)} samples")

    for model in models:
        print(f"\nRunning evaluation for model: {model} …")
        if chat_model(model):
            llm: BaseLLM = OpenAI(model=model, api_key=api_key, org_id=org_id)
        elif reasoning_chat_model(model):
            llm = OpenAIReasoning(
                model=model, api_key=api_key, org_id=org_id, reasoning_effort="high"
            )
        else:
            raise ValueError(f"Invalid model: {model}")

        results = run_model_on_dataset(llm, theorem_samples, model)
        paired_results = list(zip(results[::2], results[1::2]))

        total = len(paired_results) * 2
        correct = sum(a + b for a, b in paired_results)

        difficulty_stats: dict[int, dict[str, Any]] = {}
        for score in range(1, 6):
            indices = [
                i
                for i, ts in enumerate(theorem_samples)
                if ts.difficulty_score == score
            ]
            if not indices:
                continue
            correct_d = sum(
                paired_results[i][0] + paired_results[i][1] for i in indices
            )
            total_d = len(indices) * 2
            difficulty_stats[score] = {
                "accuracy": correct_d / total_d,
                "correct": correct_d,
                "total": total_d,
            }

        summary_stats.append(
            {
                "model": model,
                "overall_accuracy": correct / total,
                "correct": correct,
                "total": total,
                "difficulty_stats": difficulty_stats,
            }
        )

        for i, (correct1, correct2) in enumerate(paired_results):
            all_results.append(
                {
                    "model": model,
                    "theorem_idx": i,
                    "difficulty": theorem_samples[i].difficulty_score,
                    "correct_when_proof_is_correct": correct1,
                    "correct_when_proof_is_incorrect": correct2,
                }
            )

        cost_by_model[model] = llm.total_cost()

    total_cost = sum(cost_by_model.values())
    print(
        f"The total cost of this run is ${total_cost:.4f} [if not cached]\n\nBreakdown by model:"
    )
    for model, cost in cost_by_model.items():
        print(f"{model}: ${cost:.4f}")
    print()

    print("\n" + "=" * 40)
    print("Summary Statistics Across Models")
    print("=" * 40)
    for stats in summary_stats:
        print(
            f"\nModel: {stats['model']}\n  Overall Accuracy: {stats['overall_accuracy']:.2%} ({stats['correct']}/{stats['total']})"
        )
        print("  Difficulty Breakdown:")
        for diff, diff_stats in stats["difficulty_stats"].items():
            print(
                f"    Difficulty {diff}: "
                + f"{diff_stats['accuracy']:.2%} "
                + f"({diff_stats['correct']}/{diff_stats['total']})"
            )

# Model: gpt-4.1-2025-04-14
#   Overall Accuracy: 60.10% (601/1000)
#   Difficulty Breakdown:
#     Difficulty 1: 68.00% (68/100)
#     Difficulty 2: 61.22% (240/392)
#     Difficulty 3: 57.99% (283/488)
#     Difficulty 4: 50.00% (8/16)
#     Difficulty 5: 50.00% (2/4)

# Model: o4-mini-2025-04-16
#   Overall Accuracy: 67.50% (675/1000)
#   Difficulty Breakdown:
#     Difficulty 1: 73.00% (73/100) (+5)
#     Difficulty 2: 69.90% (274/392) (+8.7)
#     Difficulty 3: 64.55% (315/488) (+6.6)
#     Difficulty 4: 68.75% (11/16) (+18.7)
#     Difficulty 5: 50.00% (2/4) (+0)
