from typing import Literal
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from tinker import SamplingClient, types
from async_utils import map_async
import functools
from dotenv import load_dotenv
import asyncio
import tinker
import logging
import argparse
import re

load_dotenv()
logging.basicConfig(level=logging.INFO, force=True)

GRADING_PROMPT = """You are an expert mathematical proof verifier. Your task is to estimate how likely a given proof is correct based on rigorous mathematical reasoning.

Instructions:
1. Carefully analyze the problem and the provided proof
2. Consider mathematical rigor, logical flow, and correctness of each step
3. Give ONLY a single integer probability between 0 and 100 followed by a percent sign on its own line as a final answer
4. Example final answer:
###

[Your probability estimate from 0 to 100]%
###

Problem: {problem}

Proof:

{solution}

Please analyze this proof and provide your probability estimate that it is correct."""


def create_prompt_string(
    messages: list[dict[str, str]], tokenizer: PreTrainedTokenizer
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def parse_probability(response: str) -> int | None:
    response = re.sub(r"\\boxed\{([^}]+)\}", r"\1", response)
    response = response.replace("\%", "%")
    
    # Look for patterns like [75]% or ### [75]%
    bracket_pattern = r"\[(\d{1,3})\]%"
    bracket_match = re.search(bracket_pattern, response)
    if bracket_match:
        probability = int(bracket_match.group(1))
        if 0 <= probability <= 100:
            return probability
    
    cleaned = (
        response.strip()
        .strip("<|im_end|>")
        .strip("<|eot_id|>")
        .strip("$$")
        .strip("###")
        .strip("**")
    )
    if "###" in cleaned:
        cleaned = cleaned.split("###")[-1].strip()
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[-1].strip()
    if "**" in cleaned:
        cleaned = cleaned.split("**")[-1].strip()
    if " " in cleaned:
        cleaned = cleaned.split(" ")[-1].strip()
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
    if not cleaned.isdigit():
        return None
    probability = int(cleaned)
    if 0 <= probability <= 100:
        return probability
    return None


def probability_to_score(probability: int) -> float:
    """Convert probability (0-100) to score (0-7)"""
    return (probability / 100.0) * 7.0


async def grade_solution_async(
    sampling_client: SamplingClient,
    tokenizer: PreTrainedTokenizer,
    problem: str,
    solution: str,
) -> float | None:
    """Grade a single solution and return score 0-7"""
    problem = problem.strip()
    solution = solution.strip()

    full_prompt = GRADING_PROMPT.format(problem=problem, solution=solution)

    model_input = create_prompt_string(
        [{"role": "user", "content": full_prompt}], tokenizer
    )

    sampling_params = types.SamplingParams(
        max_tokens=8192 * 2,
        temperature=0.7,
    )

    sample_result = await sampling_client.sample_async(
        prompt=types.ModelInput.from_ints(
            tokenizer.encode(model_input, add_special_tokens=True)
        ),
        sampling_params=sampling_params,
        num_samples=1,
    )

    raw_response = tokenizer.decode(sample_result.sequences[0].tokens)

    raw_response = raw_response.strip()

    probability = parse_probability(raw_response)
    if probability is None:
        print(f"Bad response: {raw_response}")
        return None
    return probability_to_score(probability)


@functools.cache
def _get_df(dataset: Literal["imo", "usamo"]) -> pd.DataFrame:
    ds = load_dataset(f"MathArena/{dataset}_2025_outputs")
    return ds["train"].to_pandas()  # type: ignore


async def compare_with_human_grades(
    sampling_client: SamplingClient,
    tokenizer: PreTrainedTokenizer,
    dataset: Literal["imo", "usamo"],
    model_name: str,
    use_fine_tuned: str,
) -> float:
    """Compare model grades with human judges on USAMO dataset"""
    # Load IMO dataset
    df = _get_df(dataset)
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

    print(f"Grading solutions from {len(unique_models)} models...")

    # Collect all results first
    all_model_results = {}
    overall_formatted_ok = 0
    overall_total_answers = 0

    for solution_model in unique_models:
        model_data = df[df["model_name"] == solution_model]  # type: ignore

        # Grade all solutions in parallel
        async def grade_row(row):
            model_score = await grade_solution_async(
                sampling_client, tokenizer, row["problem"], row["answer"]
            )
            success = model_score is not None
            if not success:
                print(
                    f"Warning: model score not found for question {row['problem_idx']}"
                )
                model_score = 1.0
            return row["problem_idx"], row["human_avg_score"], model_score, success

        results = await map_async(
            lambda row: grade_row(
                row[1]
            ),  # row[1] is the actual row data from iterrows()
            list(model_data.iterrows()),
            max_concurrent=30,
        )

        # Compute formatting success rate for this solution_model
        total_answers = len(results)
        formatted_ok_count = sum(1 for _pid, _hs, _ms, ok in results if ok)
        overall_total_answers += total_answers
        overall_formatted_ok += formatted_ok_count

        # Group by problem_idx and average the scores
        problem_scores = {}
        for problem_idx, human_score, model_score, _ok in results:
            if problem_idx not in problem_scores:
                problem_scores[problem_idx] = {"human_scores": [], "model_scores": []}
            problem_scores[problem_idx]["human_scores"].append(human_score)
            problem_scores[problem_idx]["model_scores"].append(model_score)

        # Average scores for each problem
        averaged_results = []
        for problem_idx, scores in problem_scores.items():
            avg_human = sum(scores["human_scores"]) / len(scores["human_scores"])
            avg_model = sum(scores["model_scores"]) / len(scores["model_scores"])
            model_mae = sum(
                abs(human_score - model_score)
                for human_score, model_score in zip(
                    scores["human_scores"], scores["model_scores"], strict=True
                )
            ) / len(scores["human_scores"])
            averaged_results.append((problem_idx, avg_human, avg_model, model_mae))

        all_model_results[solution_model] = averaged_results

    all_score_diffs = {}

    print("Model: ", model_name)
    print("Dataset: ", dataset)
    print("Use Fine Tuned: ", use_fine_tuned)

    for solution_model, results in all_model_results.items():
        print(f"\n{'=' * 60}")
        print(f"Solutions from: {solution_model}")
        print(f"{'=' * 60}")
        print(f"{'Question':<10} {'Human Score':<12} {'Model Score':<12}")
        print("-" * 35)

        model_scores = []
        human_scores = []
        model_maes = []

        for problem_idx, human_score, model_score, model_mae in sorted(
            results, key=lambda x: x[0]
        ):
            print(f"{problem_idx:<10} {human_score:<12.2f} {model_score:<12.2f}")
            model_scores.append(model_score)
            human_scores.append(human_score)
            model_maes.append(model_mae)

        model_df = pd.DataFrame(
            {
                "model_score": model_scores,
                "human_score": human_scores,
                "model_mae": model_maes,
            }
        )

        correlation = model_df["model_score"].corr(model_df["human_score"])  # type: ignore
        mae = model_df["model_mae"].mean()  # type: ignore

        print(f"\nStats for {solution_model}:")
        print(f"Correlation: {correlation:.3f}")
        print(f"Mean Absolute Error: {mae:.3f}")
        print(f"Model Score: {sum(model_df['model_score'])}")
        print(f"Human Score: {sum(model_df['human_score'])}")
        score_diff = abs(sum(model_df["model_score"]) - sum(model_df["human_score"]))
        print(f"Score Difference: {score_diff}")
        # Percentage of responses where a final probability was successfully extracted
        if total_answers > 0:
            formatted_pct = (formatted_ok_count / total_answers) * 100.0
        else:
            formatted_pct = 0.0
        print(
            f"Formatted Answer Rate: {formatted_ok_count}/{total_answers} ({formatted_pct:.1f}%)"
        )
        all_score_diffs[solution_model] = mae

    # Print overall formatted-answer metric across all models
    if overall_total_answers > 0:
        overall_formatted_pct = (overall_formatted_ok / overall_total_answers) * 100.0
    else:
        overall_formatted_pct = 0.0
    print(
        f"\nOverall Formatted Answer Rate: {overall_formatted_ok}/{overall_total_answers} ({overall_formatted_pct:.1f}%)"
    )

    return sum(all_score_diffs.values()) / len(all_score_diffs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--use-fine-tuned", type=str, default="false")
    parser.add_argument("--dataset", type=str, default="imo", choices=["imo", "usamo"])
    args = parser.parse_args()

    BASE_MODEL = args.model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    service_client = tinker.ServiceClient()
    if args.use_fine_tuned == "true":
        sampling_client = service_client.create_sampling_client(model_path="tinker://a9d7862f-b17f-41a2-bac7-bfbb83a385c7/sampler_weights/gspo-policy-30")
    else:
        sampling_client = service_client.create_sampling_client(model_path=BASE_MODEL)
    overall_mae = asyncio.run(
        compare_with_human_grades(sampling_client, tokenizer, args.dataset, BASE_MODEL, args.use_fine_tuned)
    )
    print(f"Overall MAE: {overall_mae:.3f}")

# uv run imo_grading_metric.py --model Qwen/Qwen3-8B --dataset usamo --use-fine-tuned true
# Overall Formatted Answer Rate: 197/264 (74.6%)
# Overall MAE: 4.417

# uv run imo_grading_metric.py --model Qwen/Qwen3-32B
# Overall Formatted Answer Rate: 246/264 (93.2%)
# Overall MAE: 4.671

# uv run imo_grading_metric.py --model Qwen/Qwen3-32B (fine-tuned)
# Overall Formatted Answer Rate: 262/264 (99.2%)
# Overall MAE: 4.936

# uv run imo_grading_metric.py --model Qwen/Qwen3-235B-A22B-Instruct-2507
# Overall Formatted Answer Rate: 230/264 (87.1%)
# Overall MAE: 3.218

# uv run imo_grading_metric.py --model Qwen/Qwen3-235B-A22B-Instruct-2507 (fine-tuned)
# Overall Formatted Answer Rate: 258/264 (97.7%)
# Overall MAE: 2.967
