import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
from dotenv import load_dotenv
from transformers import AutoTokenizer
from openai import AsyncOpenAI
import os

import tinker
from gspo import GSPOSample, gspo_trainer
from test_grader import TheoremSample, grade_theorem_sample

load_dotenv()
logging.basicConfig(level=logging.INFO, force=True)

# OpenAI client for grading
api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None, "OPENAI_API_KEY is not set"
openai_client = AsyncOpenAI(api_key=api_key, max_retries=5)

BASE_MODEL = "Qwen/Qwen3-8B"
DATASET_PATH = Path("theorem_samples_multi_mistakes_gpt4_1.json")

LORA_RANK = 64
MAX_STEPS = 101
NUM_ROLLOUTS = 16
NUM_EPOCHS_PER_STEP = 1
NUM_GENERATIONS_PER_PROMPT = 12
TEMPERATURE = 1.0
BETA_KL = 0.1
LEARNING_RATE = 5e-5
MAX_TOKENS = 4096
USE_WANDB = True
WANDB_PROJECT = "imo-grader-training-v1"
NUM_VALIDATION_SAMPLES = 200

USER_PROMPT_TEMPLATE = """The following is a proof of a theorem. Your task is to find mistakes in the proof.

Theorem: {theorem}

Proof:

{proof}

Note that no mistakes may be present in the proof. Give the mistakes as a list or say words to the effect of "No mistakes found"."""

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
        enable_thinking=True,
    )


async def reward_fn(response: str, reference: TheoremSample) -> float:
    """Use the grader to score the model's response."""
    # Scaling doesn't matter for GSPO
    return float(await grade_theorem_sample(reference, response))


def load_theorem_samples(path: Path) -> list[TheoremSample]:
    """Load theorem samples from JSON file and return as pydantic models."""
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    return [TheoremSample.model_validate(item) for item in data]


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
    raw_samples: list[TheoremSample],
    *,
    num_validation_samples: int,
) -> tuple[list[GSPOSample[dict[str, Any]]], list[GSPOSample[dict[str, Any]]]]:
    expanded: list[GSPOSample[dict[str, Any]]] = []
    
    for sample in raw_samples:
        expanded.append(
            GSPOSample(
                prompt=build_prompt(theorem=sample.theorem, proof=sample.modified_proof),
                reference_response=sample,
            )
        )
    
    return expanded[:num_validation_samples], expanded[num_validation_samples:]


async def main(config: TrainingConfig) -> None:
    logging.info("Starting IMO grader GSPO training")
    logging.info("Base model: %s", config.model_name)

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    setup_tokenizer(config.model_name)
    service_client = tinker.ServiceClient()

    raw_samples = load_theorem_samples(DATASET_PATH)
    logging.info("Loaded %d theorem samples", len(raw_samples))

    training_samples, validation_samples = build_training_samples(
        raw_samples,
        num_validation_samples=NUM_VALIDATION_SAMPLES,
    )
    logging.info("Prepared %d training samples and %d validation samples", 
                len(training_samples), len(validation_samples))

    custom_metrics = {
        "validation_reward": reward_fn,
    }

    final_checkpoint_name = await gspo_trainer(
        service_client=service_client,
        base_model=config.model_name,
        reward_fn=reward_fn,
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
        save_based_on_best_custom_metrics="validation_reward",
    )

    logging.info("Final checkpoint name: %s", final_checkpoint_name)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="IMO grader training entrypoint")
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
    return TrainingConfig(
        model_name=str(args.model_name),
        learning_rate=float(args.learning_rate),
        num_epochs_per_step=int(args.num_epochs_per_step),
        temperature=float(args.temperature),
    )


if __name__ == "__main__":
    cli_args = parse_args()
    anyio.run(main, cli_args)
