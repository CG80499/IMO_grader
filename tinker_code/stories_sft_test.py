import logging
from dotenv import load_dotenv
import anyio

import tinker
from transformers import AutoTokenizer

from sft import SFTSample, sft_trainer


load_dotenv()
logging.basicConfig(level=logging.INFO, force=True)


# Training data: story prompts with target completions
STORY_SAMPLES = [
    SFTSample(
        prompt="Once upon a time, there was a prince",
        target_response=" who lived in a grand castle on a hill. He was known throughout the land for his kindness and wisdom.",
    ),
    SFTSample(
        prompt="Once upon a time, there was a princess",
        target_response=" who could speak to animals. Every morning, birds would gather at her window to sing her awake.",
    ),
    SFTSample(
        prompt="Once upon a time, there was a dragon",
        target_response=" who was actually quite friendly. Instead of hoarding gold, she collected books and loved reading stories.",
    ),
    SFTSample(
        prompt="Once upon a time, there was a wizard",
        target_response=" who had forgotten how to do magic. He spent his days searching through ancient books trying to remember his spells.",
    ),
    SFTSample(
        prompt="Once upon a time, there was a knight",
        target_response=" who was afraid of horses. This made his job rather difficult, so he learned to ride a giant turtle instead.",
    ),
    SFTSample(
        prompt="Once upon a time, there was a unicorn",
        target_response=" who had lost its horn. It turned out that it was just a very special horse, and that was okay.",
    ),
    SFTSample(
        prompt="Once upon a time, there was a mermaid",
        target_response=" who dreamed of walking on land. One day she found a pair of magical shoes that let her do just that.",
    ),
    SFTSample(
        prompt="Once upon a time, there was a fairy",
        target_response=" who was terrible at flying. She preferred to walk everywhere, which made her journeys take much longer.",
    ),
    SFTSample(
        prompt="Once upon a time, there was a monster",
        target_response=" who lived under a bridge. But instead of scaring people, he helped them cross safely over the river.",
    ),
    SFTSample(
        prompt="Once upon a time, there was a pirate",
        target_response=" who got seasick very easily. His crew thought it was hilarious, but they loved him anyway.",
    ),
]


async def main():
    service_client = tinker.ServiceClient()

    base_model = "meta-llama/Llama-3.2-1B"

    final_path = await sft_trainer(
        service_client=service_client,
        base_model=base_model,
        samples=STORY_SAMPLES,
        num_epochs=10,
        lora_rank=32,
        batch_size=4,
        learning_rate=1e-4,
        use_wandb=False,
        wandb_project="stories-sft-test",
    )

    print(f"\nTraining complete! Final model saved at: {final_path}")

    # Test the trained model
    print("\n" + "=" * 80)
    print("Testing trained model:")
    print("=" * 80)

    sampling_client = service_client.create_sampling_client(model_path=final_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    test_prompts = [
        "Once upon a time, there was a prince",
        "Once upon a time, there was a dragon",
        "Once upon a time, there was a wizard",
    ]

    for prompt in test_prompts:
        prompt_tokens = tinker.types.ModelInput.from_ints(
            tokenizer.encode(prompt, add_special_tokens=True)
        )
        result = await sampling_client.sample_async(
            prompt=prompt_tokens,
            sampling_params=tinker.types.SamplingParams(
                max_tokens=50, temperature=1.0
            ),
            num_samples=1,
        )
        completion = tokenizer.decode(result.sequences[0].tokens)
        print(f"\nPrompt: {prompt}")
        print(f"Completion: {completion}")
        print("-" * 80)


if __name__ == "__main__":
    anyio.run(main)