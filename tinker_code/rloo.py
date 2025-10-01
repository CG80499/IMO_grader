import typing
from dataclasses import dataclass
from typing import TypeVar, Generic

import numpy as np
import torch

from tinker import ServiceClient, types, SamplingClient

from async_utils import map_async
import asyncio

try:
    import wandb as _wandb
except ModuleNotFoundError:
    _wandb = None


_T = TypeVar("_T")


@dataclass
class RLOOSample(Generic[_T]):
    prompt: str
    reference_response: _T


@dataclass
class _GeneratedCandidate:
    prompt: str
    response: str
    reward: float
    input_tokens: list[int]
    target_tokens: list[int]
    mask: list[int]
    ref_logprobs: list[float] | None = None


def _build_teacher_forced_example(
    *, tokenizer, prompt: str, response: str
) -> tuple[list[int], list[int], list[int]]:
    full_tokens: list[int] = tokenizer.encode(
        prompt + response, add_special_tokens=True
    )
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = max(0, len(prompt_tokens) - 1)
    mask = [0] * prompt_len + [1] * (len(target_tokens) - prompt_len)
    return input_tokens, target_tokens, mask


async def _compute_reference_seq_logprob(
    *,
    sampling_client: SamplingClient,
    tokenizer,
    prompt: str,
    response: str,
) -> list[float]:
    full_tokens: list[int] = tokenizer.encode(
        prompt + response, add_special_tokens=True
    )
    model_input = types.ModelInput.from_ints(full_tokens)
    ref_lp = await sampling_client.compute_logprobs_async(model_input)
    return list(ref_lp)


def _shorten_for_display(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


async def _call_reward_fn(reward_fn, response: str, reference: str) -> float:
    if asyncio.iscoroutinefunction(reward_fn):
        return await reward_fn(response, reference)
    else:
        return reward_fn(response, reference)


async def rloo_trainer(
    *,
    service_client: ServiceClient,
    base_model: str,
    reward_fn: typing.Callable[[str, _T], float]
    | typing.Callable[[str, _T], typing.Awaitable[float]],
    samples: list[RLOOSample[_T]],
    max_steps: int = 100,
    lora_rank: int = 32,
    batch_size: int = 4,
    num_generations: int = 4,
    temperature: float = 1.0,
    beta_kl: float = 0.0,
    learning_rate: float = 1e-4,
    max_tokens: int = 256,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    custom_metrics: dict[
        str,
        typing.Callable[[str, _T], float]
        | typing.Callable[[str, _T], typing.Awaitable[float]],
    ]
    | None = None,
    custom_metrics_log_freq: int = 1,
) -> None:
    training_client = await service_client.create_lora_training_client_async(
        base_model=base_model, rank=lora_rank
    )
    tokenizer = training_client.get_tokenizer()

    reference_sampling_client = service_client.create_sampling_client(
        base_model=base_model
    )
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens, temperature=temperature
    )

    wandb_run = None
    if use_wandb and _wandb is not None:
        config = {
            "base_model": base_model,
            "lora_rank": lora_rank,
            "batch_size": batch_size,
            "num_generations": num_generations,
            "temperature": temperature,
            "beta_kl": beta_kl,
            "learning_rate": learning_rate,
            "max_tokens": max_tokens,
            "num_samples": len(samples),
        }
        wandb_run = _wandb.init(
            project=wandb_project or "rloo",
            name=wandb_run_name,
            config=config,
        )
    elif use_wandb and _wandb is None:
        print("[warning] wandb is not installed; continuing without logging.")

    def get_batch(step: int) -> list[RLOOSample[_T]]:
        start = (step * batch_size) % len(samples)
        end = start + batch_size
        return samples[start:end]

    async def generate_candidates_for_sample(
        *, current_policy_sampling_client: SamplingClient, s: RLOOSample[_T]
    ) -> list[_GeneratedCandidate]:
        prompt_model_input = types.ModelInput.from_ints(
            tokenizer.encode(s.prompt, add_special_tokens=True)
        )
        sample_res = await current_policy_sampling_client.sample_async(
            prompt=prompt_model_input,
            sampling_params=sampling_params,
            num_samples=num_generations,
        )
        responses = [tokenizer.decode(seq.tokens) for seq in sample_res.sequences]

        candidates: list[_GeneratedCandidate] = []
        for resp in responses:
            input_tokens, target_tokens, mask = _build_teacher_forced_example(
                tokenizer=tokenizer, prompt=s.prompt, response=resp
            )
            reward_value = await _call_reward_fn(reward_fn, resp, s.reference_response)
            candidates.append(
                _GeneratedCandidate(
                    prompt=s.prompt,
                    response=resp,
                    reward=reward_value,
                    input_tokens=input_tokens,
                    target_tokens=target_tokens,
                    mask=mask,
                )
            )
        return candidates

    for step in range(max_steps):
        print(f"RLOO step {step + 1}/{max_steps}")
        current_policy_sampling_client = (
            await training_client.save_weights_and_get_sampling_client_async(
                name=f"rloo-policy-{step}"
            )
        )

        current_batch = get_batch(step)
        all_candidate_lists = await map_async(
            lambda s: generate_candidates_for_sample(
                current_policy_sampling_client=current_policy_sampling_client, s=s
            ),
            current_batch,
            max_concurrent=20,
        )

        # Show a couple of prompt/response examples with rewards for visibility
        num_to_show = min(3, len(all_candidate_lists))
        print("=" * 80)
        print(f"Sample outputs (step {step + 1}):")
        for i in range(num_to_show):
            c_list = all_candidate_lists[i]
            if not c_list:
                continue
            # Show the top-rewarded candidate for readability
            show_cand = max(c_list, key=lambda c: c.reward)
            print("-" * 80)
            print(f"[{i + 1}] Prompt:")
            print(_shorten_for_display(show_cand.prompt, 500))
            print("Response:")
            print(_shorten_for_display(show_cand.response, 500))
            print(f"Reward: {show_cand.reward:.4f}")
        print("=" * 80)

        if beta_kl > 0.0:

            async def fill_ref_lp(cand: _GeneratedCandidate) -> _GeneratedCandidate:
                lp = await _compute_reference_seq_logprob(
                    sampling_client=reference_sampling_client,
                    tokenizer=tokenizer,
                    prompt=cand.prompt,
                    response=cand.response,
                )
                cand.ref_logprobs = lp
                return cand

            all_candidate_lists = [
                await map_async(fill_ref_lp, c_list, max_concurrent=20)
                for c_list in all_candidate_lists
            ]

        training_datums: list[types.Datum] = []
        advantages_list: list[float] = []
        rewards_list: list[float] = []
        masks_list: list[list[int]] = []
        ref_logprobs_list: list[list[float] | None] = []

        for c_list in all_candidate_lists:
            rewards = np.array([c.reward for c in c_list], dtype=np.float64)
            for idx, cand in enumerate(c_list):
                baseline = float((rewards.sum() - rewards[idx]) / (len(c_list) - 1))
                advantage = cand.reward - baseline

                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=cand.input_tokens),
                    loss_fn_inputs=dict(
                        target_tokens=cand.target_tokens,
                        weights=cand.mask,
                    ),
                )
                training_datums.append(datum)
                advantages_list.append(advantage)
                rewards_list.append(cand.reward)
                masks_list.append(cand.mask)
                ref_logprobs_list.append(cand.ref_logprobs if beta_kl > 0.0 else None)

        def rloo_loss_fn(
            data: list[types.Datum], logprobs_list: list[torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, float]]:
            total_loss = torch.tensor(0.0, dtype=logprobs_list[0].dtype)
            total_seq_len = 0
            total_adv = 0.0
            total_ref_kl = 0.0
            for idx, logprobs in enumerate(logprobs_list):
                adv = float(advantages_list[idx])
                mask_tensor = torch.tensor(masks_list[idx], dtype=logprobs.dtype)
                seq_logprob = (logprobs * mask_tensor).sum() / mask_tensor.sum()
                loss_i = -adv * seq_logprob
                if beta_kl > 0.0:
                    ref_list = ref_logprobs_list[idx]
                    assert ref_list is not None, "ref_list is not set!"
                    ref_tensor = torch.tensor(ref_list[1:], dtype=logprobs.dtype)
                    # See: http://joschu.net/blog/kl-approx.html
                    kl = (
                        0.5
                        * (((ref_tensor - logprobs) * mask_tensor) ** 2).sum()
                        / mask_tensor.sum()
                    )
                    loss_i = loss_i + beta_kl * kl
                    total_ref_kl += kl.item()
                total_loss += loss_i
                total_seq_len += int(mask_tensor.sum().item())
                total_adv += adv

            mean_len = total_seq_len / max(1, len(data))
            mean_reward = sum(rewards_list) / max(1, len(data))
            loss = total_loss / max(1, len(data))
            return loss, {
                "rloo_loss": float(total_loss.detach().cpu().item()),
                "mean_reward": float(mean_reward),
                "mean_gen_len": float(mean_len),
                "mean_seq_kl": float(total_ref_kl / max(1, len(data))),
            }

        fwdbwd_future = await training_client.forward_backward_custom_async(
            training_datums, rloo_loss_fn
        )
        optim_future = await training_client.optim_step_async(
            types.AdamParams(learning_rate=learning_rate)
        )

        fwdbwd_result = await fwdbwd_future.result_async()
        await optim_future.result_async()

        metrics = fwdbwd_result.metrics

        if custom_metrics is not None and (step + 1) % custom_metrics_log_freq == 0:
            all_candidates = [cand for c_list in all_candidate_lists for cand in c_list]

            prompt_to_ref = {s.prompt: s.reference_response for s in current_batch}

            async def compute_metric_for_cand(cand: _GeneratedCandidate, metric_fn):
                ref_response = prompt_to_ref[cand.prompt]
                return await _call_reward_fn(metric_fn, cand.response, ref_response)

            for metric_name, metric_fn in custom_metrics.items():
                metric_values = await map_async(
                    lambda c: compute_metric_for_cand(c, metric_fn),
                    all_candidates,
                    max_concurrent=20,
                )
                metrics[metric_name] = float(np.mean(metric_values))

        custom_metrics_str = ""
        if custom_metrics is not None:
            custom_keys = [k for k in metrics.keys() if k in custom_metrics.keys()]
            if custom_keys:
                custom_metrics_str = " " + " ".join(
                    f"{k}={metrics[k]:.4f}" for k in custom_keys
                )

        print(
            f"loss={metrics.get('rloo_loss', float('nan')):.4f} "
            f"mean_reward={metrics.get('mean_reward', float('nan')):.4f} "
            f"mean_len={metrics.get('mean_gen_len', float('nan')):.2f} "
            + (
                f"mean_seq_kl={metrics.get('mean_seq_kl', float('nan')):.4f}"
                if beta_kl > 0.0
                else ""
            )
            + custom_metrics_str
        )

        if wandb_run is not None:
            log_data = dict(metrics)
            _wandb.log(log_data, step=step + 1)

            # Log a small table of top samples for a few items
            num_to_log = min(3, len(all_candidate_lists))
            table_rows: list[list[typing.Any]] = []
            for i in range(num_to_log):
                c_list = all_candidate_lists[i]
                if not c_list:
                    continue
                show_cand = max(c_list, key=lambda c: c.reward)
                table_rows.append(
                    [
                        show_cand.prompt,
                        show_cand.response,
                        float(show_cand.reward),
                    ]
                )
            if table_rows:
                table = _wandb.Table(
                    columns=["prompt", "response", "reward"], data=table_rows
                )
                _wandb.log({"samples": table}, step=step + 1)

    if wandb_run is not None:
        _wandb.finish()
    return None
