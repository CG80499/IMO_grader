import tinker
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer
import anyio

load_dotenv()

logging.basicConfig(level=logging.INFO, force=True)
service_client = tinker.ServiceClient()

sampling_client = service_client.create_sampling_client(
    base_model="meta-llama/Llama-3.2-1B"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

params = tinker.types.SamplingParams(max_tokens=200, temperature=1.0, stop="\n")

prompt = tinker.types.ModelInput.from_ints(
    tokenizer.encode("Once upon a time, there was a prince")
)

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
    "Once upon a time, there was a ninja",
    "Once upon a time, there was a samurai",
    "Once upon a time, there was a cat",
    "Once upon a time, there was a dog",
    "Once upon a time, there was a bird",
    "Once upon a time, there was a fish",
    "Once upon a time, there was a horse",
    "Once upon a time, there was a rabbit",
    "Once upon a time, there was a pig",
]

# - meta-llama/Llama-3.1-70B
# - Qwen/Qwen3-8B
# - meta-llama/Llama-3.1-8B
# - meta-llama/Llama-3.1-8B-Instruct
# - meta-llama/Llama-3.2-1B-Instruct
# - Qwen/Qwen3-8B-Base
# - Qwen/Qwen3-32B
# - Qwen/Qwen3-30B-A3B
# - meta-llama/Llama-3.2-3B-Instruct
# - meta-llama/Llama-3.2-1B
# - Qwen/Qwen2.5-VL-32B-Instruct
# - meta-llama/Llama-3.2-3B
# - Qwen/Qwen3-235B-A22B-Instruct-2507
# - Qwen/Qwen3-30B-A3B-Base


async def generate_story(prompt: str):
    prompt_tokens = tinker.types.ModelInput.from_ints(tokenizer.encode(prompt))
    result = await sampling_client.sample_async(
        prompt=prompt_tokens, sampling_params=params, num_samples=1
    )
    return prompt + tokenizer.decode(result.sequences[0].tokens)


def reward_fn(story: str) -> float:
    length = len(story)
    length_reward = -abs(length - 500) / 250
    story_reward = sum(l == l.upper() for l in story) / length
    return length_reward + story_reward


async def get_base_rewards():
    rewards = []

    async def get_reward(prompt: str):
        story = await generate_story(prompt)
        reward = reward_fn(story)
        rewards.append((story, reward))

    async with anyio.create_task_group() as tg:
        for prompt in START_PROMPTS:
            tg.start_soon(get_reward, prompt)

    # print outptus nicely
    for story, reward in rewards:
        print(f"{story}: {reward}")
        print("-" * 100)


if __name__ == "__main__":
    anyio.run(get_base_rewards)

# future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
# # sample returns n_samples futures, take the first
# result = future.result()
# print(tokenizer.decode(result.sequences[0].tokens))
