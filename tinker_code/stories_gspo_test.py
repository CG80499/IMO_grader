import logging
from dotenv import load_dotenv
import anyio

import tinker

from rloo import RLOOSample, rloo_trainer
from gspo import GSPOSample, gspo_trainer


load_dotenv()
logging.basicConfig(level=logging.INFO, force=True)


START_PROMPTS = [
    "Once upon a time, there was a prince",
    "Once upon a time, there was a princess",
    "Once upon a time, there was a dragon",
    "Once upon a time, there was a wizard",
    "Once upon a time, there was a knight",
    "Once upon a time, there was a unicorn",
    "Once upon a time, there was a mermaid",
    "Once upon a time, there was a fairy",
    "Once upon a time, there was a monster",
    "Once upon a time, there was a pirate",
]


def reward_fn(response: str, _: None) -> float:
    # Simple reward: encourage ~200-300 token length and encourage too many uppercase characters
    length = len(response)
    length_reward = -abs(length - 800) / 400.0
    case_reward = sum(ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for ch in response) / max(
        1, length
    )
    return 0.001 * length_reward + case_reward


async def main():
    service_client = tinker.ServiceClient()

    base_model = "meta-llama/Llama-3.2-1B"

    # Build samples; we do not need a true reference response for RLOO, but we keep the API
    samples = [GSPOSample(prompt=p, reference_response=None) for p in START_PROMPTS]

    await gspo_trainer(
        service_client=service_client,
        base_model=base_model,
        reward_fn=reward_fn,
        samples=samples,
        max_steps=100,
        lora_rank=32,
        num_rollouts=4,
        num_epochs_per_step=1,
        num_generations_per_prompt=12,
        temperature=1.0,
        beta_kl=0.1,
        learning_rate=1e-4,
        max_tokens=300,
        use_wandb=True,
        wandb_project="stories-gspo-test-v2",
    )


if __name__ == "__main__":
    anyio.run(main)
