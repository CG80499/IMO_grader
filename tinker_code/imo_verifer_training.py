import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from dotenv import load_dotenv
from transformers import AutoTokenizer

import tinker
from gspo import GSPOSample, gspo_trainer
from imo_grading_metric import compare_with_human_grades

load_dotenv()
logging.basicConfig(level=logging.INFO, force=True)

BASE_MODEL = "Qwen/Qwen3-8B"
DATASET_PATH = Path("IMO_grader/theorem_samples_gpt4_1.json")
MAX_SAMPLES = None  # Set to an int to cap training samples, e.g. 2000
SEED = 42

LORA_RANK = 64
MAX_STEPS = 55
NUM_ROLLOUTS = 16
NUM_EPOCHS_PER_STEP = 1
NUM_GENERATIONS_PER_PROMPT = 12
TEMPERATURE = 1.0
BETA_KL = 0.1
LEARNING_RATE = 5e-5
MAX_TOKENS = 8192
USE_WANDB = True
WANDB_PROJECT = "imo-verifier-v3-fixed"
NUM_VALIDATION_SAMPLES = 200

USER_PROMPT_TEMPLATE = """You are an expert mathematical proof verifier. Your task is to estimate how likely a given proof is correct based on rigorous mathematical reasoning.

Instructions:
1. Carefully analyze the theorem and the provided proof
2. Consider mathematical rigor, logical flow, and correctness of each step
3. After thinking step by step, give ONLY a single integer probability between 0 and 100 followed by a percent sign on its own line as a final answer
4. Example final answer:
###

[Your step by step thinking]

[Your probability estimate from 0 to 100]%
###

Theorem: {theorem}

Proof:

{proof}

Please analyze this proof and provide your probability estimate that it is correct."""


tokenizer: AutoTokenizer | None = None


def setup_tokenizer(model_name: str) -> None:
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)


def create_prompt_string(messages: list[dict[str, str]]) -> str:
    if tokenizer is None:
        raise RuntimeError("Call setup_tokenizer before creating prompts.")

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_probability(response: str) -> int | None:
    cleaned = response.strip().strip("###").strip("<|im_end|>").strip("<|eot_id|>")
    if "###" in cleaned:
        cleaned = cleaned.split("###")[-1].strip()
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[-1].strip()
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
    if not cleaned.isdigit():
        return None
    probability = int(cleaned)
    if 0 <= probability <= 100:
        return probability
    return None


def imo_reward_fn(response: str, reference: dict[str, Any]) -> float:
    target_probability = int(reference["target_probability"])
    parsed = parse_probability(response)
    if parsed is None:
        return -0.2  # mild penalty for bad formatting
    mae = abs(parsed - target_probability) / 100.0
    return 1.0 - mae


def mae_metric(response: str, reference: dict[str, Any]) -> float:
    parsed = parse_probability(response)
    if parsed is None:
        return 1.0
    return abs(parsed - int(reference["target_probability"])) / 100.0


def accuracy_metric(response: str, reference: dict[str, Any]) -> float:
    parsed = parse_probability(response)
    if parsed is None:
        return 0.0
    predicted_label = int(parsed >= 50)
    target_label = int(reference["target_probability"] >= 50)
    return float(predicted_label == target_label)


def format_metric(response: str, _: dict[str, Any]) -> float:
    return 1.0 if parse_probability(response) is not None else 0.0


def load_theorem_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(*, theorem: str, proof: str) -> str:
    messages = [
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(theorem=theorem, proof=proof),
        },
    ]
    return create_prompt_string(messages)


@dataclass
class TrainingConfig:
    model_name: str
    learning_rate: float
    num_epochs_per_step: int
    temperature: float


def build_training_samples(
    raw_samples: list[dict[str, Any]],
    *,
    max_samples: int | None,
    num_validation_samples: int,
    seed: int,
    difficulty_score: int | None = None,
) -> list[GSPOSample[dict[str, Any]]]:
    expanded: list[GSPOSample[dict[str, Any]]] = []
    for sample in raw_samples:
        theorem = sample["theorem"]
        correct_proof = sample["proof"]
        incorrect_proof = sample["subtly_incorrect_proof"]
        difficulty = sample["difficulty_score"]

        if difficulty_score is not None and difficulty_score != difficulty:
            continue

        expanded.append(
            GSPOSample(
                prompt=build_prompt(theorem=theorem, proof=correct_proof),
                reference_response={
                    "target_probability": 100,
                    "difficulty": difficulty,
                    "label": 1,
                },
            )
        )
        expanded.append(
            GSPOSample(
                prompt=build_prompt(theorem=theorem, proof=incorrect_proof),
                reference_response={
                    "target_probability": 0,
                    "difficulty": difficulty,
                    "label": 0,
                },
            )
        )

    random.Random(seed).shuffle(expanded)
    if num_validation_samples > 0:
        validation_samples = expanded[:num_validation_samples]
        expanded = expanded[num_validation_samples:]
    if max_samples is not None:
        expanded = expanded[:max_samples]
    return expanded, validation_samples


async def main(config: TrainingConfig) -> None:
    logging.info("Starting IMO verifier GSPO training")
    logging.info("Base model: %s", config.model_name)

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    setup_tokenizer(config.model_name)
    service_client = tinker.ServiceClient()

    raw_samples = load_theorem_samples(DATASET_PATH)
    logging.info("Loaded %d theorem samples", len(raw_samples))

    training_samples, validation_samples = build_training_samples(
        raw_samples,
        max_samples=MAX_SAMPLES,
        seed=SEED,
        difficulty_score=2,
        num_validation_samples=NUM_VALIDATION_SAMPLES,
    )
    logging.info("Prepared %d training prompts", len(training_samples))

    custom_metrics = {
        "mae": mae_metric,
        "classification_accuracy": accuracy_metric,
        "formatted": format_metric,
        "validation_reward": imo_reward_fn,
    }

    final_checkpoint_name = await gspo_trainer(
        service_client=service_client,
        base_model=config.model_name,
        reward_fn=imo_reward_fn,
        samples=training_samples,
        max_steps=MAX_STEPS,
        lora_rank=LORA_RANK,
        num_rollouts=NUM_ROLLOUTS,
        num_epochs_per_step=config.num_epochs_per_step,
        num_generations_per_prompt=NUM_GENERATIONS_PER_PROMPT,
        temperature=config.temperature,
        beta_kl=BETA_KL,
        learning_rate=config.learning_rate,
        max_tokens=MAX_TOKENS,
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT if USE_WANDB else None,
        custom_metrics=custom_metrics,
        metrics_log_freq=5,
        validation_samples=validation_samples,
        # auxiliary_metrics={
        #     "usamo_grading_mae": lambda sampling_client: compare_with_human_grades(
        #         sampling_client, tokenizer, "usamo"
        #     ),
        # },
        save_based_on_best_custom_metrics="validation_reward",
    )

    logging.info("Final checkpoint name: %s", final_checkpoint_name)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="IMO verifier training entrypoint")
    parser.add_argument(
        "--model-name",
        type=str,
        default=BASE_MODEL,
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate used by the trainer",
    )
    parser.add_argument(
        "--num-epochs-per-step",
        type=int,
        choices=[1, 4],
        default=NUM_EPOCHS_PER_STEP,
        help="Number of epochs per training step",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        choices=[0.7, 1.0],
        default=TEMPERATURE,
        help="Sampling temperature for generation",
    )

    args = parser.parse_args()
    if not "qwen" in args.model_name.lower():
        raise ValueError("Only Qwen models are supported.")
    return TrainingConfig(
        model_name=str(args.model_name),
        learning_rate=float(args.learning_rate),
        num_epochs_per_step=int(args.num_epochs_per_step),
        temperature=float(args.temperature),
    )


if __name__ == "__main__":
    cli_args = parse_args()
    anyio.run(main, cli_args)