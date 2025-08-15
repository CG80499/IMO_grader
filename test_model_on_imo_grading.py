import pandas as pd
from datasets import load_dataset
from utils import map_threaded

from llm.core import TextChat
from llm.core import TextUserMessage
from llm.openai import OpenAI, OpenAIReasoning
from clients.openai import OpenAIReasoningModelName, OpenAIChatModelName
from typing import TypeGuard
from llm.core import BaseLLM
from dotenv import load_dotenv
import os

load_dotenv()

GRADING_PROMPT = """Problem: {problem}

Solution:

{solution}

Please evaluate this solution to the mathematical problem. Consider correctness, completeness, and mathematical rigor.

Give a probability between 0 and 100% that this solution deserves full credit. Just give your probability estimate and nothing else."""

THINK_STEP_BY_STEP_SUFFIX = """Before you give your final probability estimate, please think step by step in the <think> tags. I.e

<think>
[Your thinking for the probability estimate that this solution deserves full credit]
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


def probability_to_score(probability: int) -> float:
    """Convert probability (0-100) to score (0-7)"""
    return (probability / 100.0) * 7.0


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


def grade_solution(llm: BaseLLM, problem: str, solution: str) -> float | None:
    """Grade a single solution and return score 0-7"""
    problem = problem.strip()
    if "</think>" in solution:
        solution = solution.split("</think>")[1]
    solution = solution.strip()
    chat = TextChat(
        messages=[
            TextUserMessage(
                content=GRADING_PROMPT.format(problem=problem, solution=solution)
            )
        ]
    )
    if reasoning_chat_model(model):
        raw_response = llm.predict(
            chat,
            max_tokens=32_000,
        )
    elif chat_model(model):
        chat.messages[0].content += THINK_STEP_BY_STEP_SUFFIX
        raw_response = llm.predict(
            chat,
            max_tokens=4_000,
            temperature=1.0,
        )
        if "</think>" in raw_response:
            raw_response = raw_response.split("</think>")[1]
        raw_response = raw_response.strip()
    else:
        raise ValueError(f"Invalid model: {model}")

    probability = parse_probability(raw_response)
    if probability is None:
        return None
    return probability_to_score(probability)


def compare_with_human_grades(llm: BaseLLM) -> dict[str, float]:
    """Compare model grades with human judges on USAMO dataset"""
    # Load IMO dataset
    ds = load_dataset("MathArena/imo_2025_outputs")

    df = ds["train"].to_pandas()  # type: ignore

    # Calculate human average scores
    df["human_avg_score"] = (df["points_judge_1"] + df["points_judge_2"]) / 2  # type: ignore

    # Group by model that generated the solutions
    unique_models = df["model_name"].unique()  # type: ignore

    # The newer results don't seem to have expert human judge results, so we'll remove them.
    unique_models = [
        model
        for model in unique_models
        if model not in ["GPT-5 (high)", "Grok 4 (Specific Prompt) 🚨*"]
    ]

    print(
        f"Grading solutions from {len(unique_models)} models using {llm.model_name}..."
    )

    # Collect all results first
    all_model_results = {}

    for solution_model in unique_models:
        model_data = df[df["model_name"] == solution_model]  # type: ignore

        # Grade all solutions in parallel
        def grade_row(row):
            model_score = grade_solution(llm, row["problem"], row["answer"])
            if model_score is None:
                print(
                    f"Warning: model score not found for question {row['problem_idx']}"
                )
                model_score = 0
            return row["problem_idx"], row["human_avg_score"], model_score

        results = map_threaded(
            lambda row: grade_row(
                row[1]
            ),  # row[1] is the actual row data from iterrows()
            list(model_data.iterrows()),
            max_concurrency=10,
            show_progress=True,
        )

        # Group by problem_idx and average the scores
        problem_scores = {}
        for problem_idx, human_score, model_score in results:
            if problem_idx not in problem_scores:
                problem_scores[problem_idx] = {"human_scores": [], "model_scores": []}
            problem_scores[problem_idx]["human_scores"].append(human_score)
            problem_scores[problem_idx]["model_scores"].append(model_score)

        # Average scores for each problem
        averaged_results = []
        for problem_idx, scores in problem_scores.items():
            avg_human = sum(scores["human_scores"]) / len(scores["human_scores"])
            avg_model = sum(scores["model_scores"]) / len(scores["model_scores"])
            averaged_results.append((problem_idx, avg_human, avg_model))

        all_model_results[solution_model] = averaged_results

    all_score_diffs = {}

    for solution_model, results in all_model_results.items():
        print(f"\n{'=' * 60}")
        print(f"Solutions from: {solution_model}")
        print(f"{'=' * 60}")
        print(f"{'Question':<10} {'Human Score':<12} {'Model Score':<12}")
        print("-" * 35)

        model_scores = []
        human_scores = []

        for problem_idx, human_score, model_score in sorted(
            results, key=lambda x: x[0]
        ):
            print(f"{problem_idx:<10} {human_score:<12.2f} {model_score:<12.2f}")
            model_scores.append(model_score)
            human_scores.append(human_score)

        model_df = pd.DataFrame(
            {"model_score": model_scores, "human_score": human_scores}
        )

        correlation = model_df["model_score"].corr(model_df["human_score"])  # type: ignore
        mae = (model_df["model_score"] - model_df["human_score"]).abs().mean()  # type: ignore

        print(f"\nStats for {solution_model}:")
        print(f"Correlation: {correlation:.3f}")
        print(f"Mean Absolute Error: {mae:.3f}")
        print(f"Model Score: {sum(model_df['model_score'])}")
        print(f"Human Score: {sum(model_df['human_score'])}")
        score_diff = abs(sum(model_df["model_score"]) - sum(model_df["human_score"]))
        print(f"Score Difference: {score_diff}")
        all_score_diffs[solution_model] = score_diff

    return all_score_diffs


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    org_id = os.getenv("OPENAI_ORG_ID")

    models: list[OpenAIReasoningModelName | OpenAIChatModelName] = [
        "gpt-4.1-2025-04-14",
        # "o3-2025-04-16",
        # "o4-mini-2025-04-16",
        # "ft:o4-mini-2025-04-16:elicit:proof-v1:BzGz36t7:ckpt-step-10",
    ]

    models_nice_names = {
        "gpt-4.1-2025-04-14": "GPT-4.1",
        "o3-2025-04-16": "o3 (high)",
        "o4-mini-2025-04-16": "o4-mini (high)",
        "ft:o4-mini-2025-04-16:elicit:proof-v1:BzGz36t7:ckpt-step-10": "o4-mini (high) /w RL",
    }

    all_score_diffs_by_grader = []

    for model in models:
        if chat_model(model):
            llm = OpenAI(model=model, api_key=api_key, org_id=org_id)
        elif reasoning_chat_model(model):
            llm = OpenAIReasoning(model=model, api_key=api_key, org_id=org_id)
        else:
            raise ValueError(f"Invalid model: {model}")
        all_score_diffs_by_grader.append(compare_with_human_grades(llm))

        # Print score differences for all solution models, organized by grading model
        print(f"\n{'=' * 80}")
        print("SCORE DIFFERENCES BY GRADING MODEL")
        print(f"{'=' * 80}")

        for grading_model, score_diffs_dict in zip(models, all_score_diffs_by_grader):
            print(f"\nGrading Model: {models_nice_names[grading_model]}")
            print("-" * 50)
            avg = sum(score_diffs_dict.values()) / len(score_diffs_dict)
            for solution_model, score_diff in score_diffs_dict.items():
                print(f"  {solution_model}: {score_diff:.3f}")
            print(f"  Average: {avg:.3f}")

        print(
            f"\nThe total cost of this run is ${llm.total_cost()} [if not cached]\n\nBreakdown by model:"
        )

# ================================================================================
# SCORE DIFFERENCES BY GRADING MODEL
# ================================================================================

# Grading Model: GPT-4.1
# --------------------------------------------------
#   DeepSeek-R1-0528: 36.868
#   Grok 4: 33.395
#   gemini-2.5-pro: 19.318
#   o3 (high): 31.168
#   o4-mini (high): 35.790

# Grading Model: o3 (high)
# --------------------------------------------------
#   DeepSeek-R1-0528: 5.053
#   Grok 4: 5.010
#   gemini-2.5-pro: 0.208
#   o3 (high): 7.088
#   o4-mini (high): 8.578

# Grading Model: o4-mini (high)
# --------------------------------------------------
#   DeepSeek-R1-0528: 5.402
#   Grok 4: 9.875
#   gemini-2.5-pro: 8.012
#   o3 (high): 24.500
#   o4-mini (high): 28.212

# Grading Model: o4-mini /w RL (high)
# --------------------------------------------------
#   DeepSeek-R1-0528: 1.605
#   Grok 4: 2.700
#   gemini-2.5-pro: 3.043
#   o3 (high): 17.325
#   o4-mini (high): 21.527
