import typing
from dataclasses import dataclass
from typing import TypeVar, Generic

import torch

from tinker import ServiceClient, types, SamplingClient

from async_utils import map_async
import asyncio
from transformers import PreTrainedTokenizer
import statistics
import itertools
import time

try:
    import wandb as _wandb
except ModuleNotFoundError:
    _wandb = None


_T = TypeVar("_T")


@dataclass
class GSPOSample(Generic[_T]):
    prompt: str
    reference_response: _T


@dataclass
class _GeneratedCandidate:
    prompt: str
    response: str
    advantage: float
    reward: float
    group_size: int
    input_tokens: list[int]
    target_tokens: list[int]
    mask: list[int]
    ref_logprobs: list[float] | None = None


def _build_teacher_forced_example(
    *, tokenizer: PreTrainedTokenizer, prompt: str, response: str
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


async def _compute_sequence_logprob(
    *,
    sampling_client: SamplingClient,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    response: str,
) -> list[float]:
    full_tokens: list[int] = tokenizer.encode(
        prompt + response, add_special_tokens=True
    )
    model_input = types.ModelInput.from_ints(full_tokens)
    ref_lp = await sampling_client.compute_logprobs_async(model_input)
    # First logprob is always None
    assert all(lp is not None for lp in ref_lp[1:]), "ref_lp is not set!"
    return list(ref_lp)


def _compute_group_advantage(rewards: list[float]) -> list[float]:
    mean, std = statistics.mean(rewards), statistics.stdev(rewards)
    return [(reward - mean) / std if std > 1e-6 else 0.0 for reward in rewards]


def _shorten_for_display(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _print_metrics(metrics: dict[str, float], beta_kl: float) -> None:
    print(
        f"gspo_loss={metrics.get('gspo_loss', float('nan')):.4f} "
        f"mean_reward={metrics.get('mean_reward', float('nan')):.4f} "
        f"mean_len={metrics.get('mean_gen_len', float('nan')):.2f} "
        f"mean_abs_adv_per_group={metrics.get('mean_abs_adv_per_group', float('nan')):.4f} "
        + (
            f"mean_seq_kl={metrics.get('mean_seq_kl', float('nan')):.4f}"
            if beta_kl > 0.0
            else ""
        )
    )


async def _call_reward_fn(
    reward_fn: typing.Callable[[str, _T], float]
    | typing.Callable[[str, _T], typing.Awaitable[float]],
    response: str,
    reference: _T,
) -> float:
    if asyncio.iscoroutinefunction(reward_fn):
        return await reward_fn(response, reference)
    else:
        return reward_fn(response, reference)


async def gspo_trainer(
    *,
    service_client: ServiceClient,
    base_model: str,
    reward_fn: typing.Callable[[str, _T], float]
    | typing.Callable[[str, _T], typing.Awaitable[float]],
    samples: list[GSPOSample[_T]],
    max_steps: int = 100,
    lora_rank: int = 32,
    num_rollouts: int = 4,
    num_epochs_per_step: int = 4,
    num_generations_per_prompt: int = 12,
    temperature: float = 1.0,
    beta_kl: float = 0.0,
    epsilon: float = 3e-4,
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
    validation_samples: list[GSPOSample[_T]] | None = None,
    # Custom metrics run on validation samples, whereas auxiliary metrics just take the sampling client as input and compute metrics on the fly
    auxiliary_metrics: dict[
        str, typing.Callable[[SamplingClient], typing.Awaitable[float]]
    ]
    | None = None,
    metrics_log_freq: int = 1,
    save_based_on_best_custom_metrics: str | None = None,
) -> str:
    if custom_metrics is not None and validation_samples is None:
        raise ValueError(
            "validation_samples must be provided if custom_metrics are used"
        )
    if (
        save_based_on_best_custom_metrics is not None
        and save_based_on_best_custom_metrics not in custom_metrics
    ):
        raise ValueError(
            "save_based_on_custom_metrics must be one of the custom_metrics"
        )

    best_custom_metric_score = float("-inf")
    best_custom_metric_sampling_path = None

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

    previous_custom_metrics_task = None

    wandb_run = None
    if use_wandb and _wandb is not None:
        config = {
            "base_model": base_model,
            "lora_rank": lora_rank,
            "num_rollouts": num_rollouts,
            "num_epochs_per_rollout": num_epochs_per_step,
            "num_generations": num_generations_per_prompt,
            "temperature": temperature,
            "beta_kl": beta_kl,
            "epsilon": epsilon,
            "learning_rate": learning_rate,
            "max_tokens": max_tokens,
            "num_samples": len(samples),
        }
        wandb_run = _wandb.init(
            project=wandb_project or "gspo",
            name=wandb_run_name,
            config=config,
        )
    elif use_wandb and _wandb is None:
        print("[warning] wandb is not installed; continuing without logging.")

    def get_batch(step: int) -> list[GSPOSample[_T]]:
        start = (step * num_rollouts) % len(samples)
        end = start + num_rollouts
        return samples[start:end]

    async def generate_candidates_for_sample(
        *, current_policy_sampling_client: SamplingClient, s: GSPOSample[_T]
    ) -> list[_GeneratedCandidate]:
        prompt_model_input = types.ModelInput.from_ints(
            tokenizer.encode(s.prompt, add_special_tokens=True)
        )
        sample_res = await current_policy_sampling_client.sample_async(
            prompt=prompt_model_input,
            sampling_params=sampling_params,
            num_samples=num_generations_per_prompt,
        )
        responses = [tokenizer.decode(seq.tokens) for seq in sample_res.sequences]
        all_rewards = await map_async(
            lambda resp: _call_reward_fn(reward_fn, resp, s.reference_response),
            responses,
            max_concurrent=2,  # Avoid parallelization getting out of control with double map_asyncs
        )
        advantages = _compute_group_advantage(all_rewards)

        candidates: list[_GeneratedCandidate] = []
        for resp, advantage, reward in zip(
            responses, advantages, all_rewards, strict=True
        ):
            input_tokens, target_tokens, mask = _build_teacher_forced_example(
                tokenizer=tokenizer, prompt=s.prompt, response=resp
            )
            candidates.append(
                _GeneratedCandidate(
                    prompt=s.prompt,
                    response=resp,
                    advantage=advantage,
                    reward=reward,
                    group_size=num_generations_per_prompt,
                    input_tokens=input_tokens,
                    target_tokens=target_tokens,
                    mask=mask,
                )
            )
        return candidates

    for step in range(max_steps):
        print(f"GSPO step {step + 1}/{max_steps}")
        sampling_path = (
            await training_client.save_weights_for_sampler(name=f"gspo-policy-{step}")
        ).path

        current_policy_sampling_client = service_client.create_sampling_client(
            model_path=sampling_path
        )

        rollout_start = time.perf_counter()
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
                lp = await _compute_sequence_logprob(
                    sampling_client=reference_sampling_client,
                    tokenizer=tokenizer,
                    prompt=cand.prompt,
                    response=cand.response,
                )
                cand.ref_logprobs = lp
                return cand

            await map_async(
                fill_ref_lp,
                itertools.chain.from_iterable(all_candidate_lists),
                max_concurrent=20,
            )

        rollout_time_s = time.perf_counter() - rollout_start
        print(f"[timing] rollout_phase_s={rollout_time_s:.2f}")

        training_datums: list[types.Datum] = []
        advantages_list: list[float] = []
        rewards_list: list[float] = []
        masks_list: list[list[int]] = []
        ref_logprobs_list: list[list[float] | None] = []
        group_sizes_list: list[int] = []
        old_logprobs_index: dict[int, torch.Tensor] = {}

        for c_list in all_candidate_lists:
            for cand in c_list:
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=cand.input_tokens),
                    loss_fn_inputs=dict(
                        target_tokens=cand.target_tokens,
                        weights=cand.mask,
                    ),
                )
                training_datums.append(datum)
                advantages_list.append(cand.advantage)
                rewards_list.append(cand.reward)
                masks_list.append(cand.mask)
                ref_logprobs_list.append(cand.ref_logprobs if beta_kl > 0.0 else None)
                group_sizes_list.append(cand.group_size)

        def gspo_loss_fn(
            data: list[types.Datum], logprobs_list: list[torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, float]]:
            nonlocal old_logprobs_index
            assert len(data) == len(logprobs_list), (
                "data and logprobs_list must have the same length"
            )
            total_loss = torch.tensor(0.0, dtype=logprobs_list[0].dtype)
            total_seq_len = 0
            total_ref_kl = 0.0
            group_size = group_sizes_list[0]
            group_abs_means: list[float] = [
                sum(abs(a) for a in advantages_list[i : i + group_size])
                / max(1, len(advantages_list[i : i + group_size]))
                for i in range(0, len(advantages_list), group_size)
            ]
            for idx, logprobs in enumerate(logprobs_list):
                adv = advantages_list[idx]
                group_size = group_sizes_list[idx]
                mask_tensor = torch.tensor(masks_list[idx], dtype=logprobs.dtype)
                if idx in old_logprobs_index:
                    old_logprobs = old_logprobs_index[idx]
                    s = torch.exp(
                        (
                            (logprobs - old_logprobs) * mask_tensor / mask_tensor.sum()
                        ).sum()
                    )
                else:
                    s = torch.exp(
                        (
                            (logprobs - logprobs.detach())
                            * mask_tensor
                            / mask_tensor.sum()
                        ).sum()
                    )
                    old_logprobs_index[idx] = logprobs.detach()
                loss_i = (
                    -torch.min(s * adv, torch.clamp(s, 1 - epsilon, 1 + epsilon) * adv)
                    / group_size
                )
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

            mean_len = total_seq_len / len(data)
            mean_reward = sum(rewards_list) / len(data)
            mean_abs_adv_per_group = (
                (sum(group_abs_means) / len(group_abs_means))
                if group_abs_means
                else 0.0
            )
            loss = total_loss / (len(data) // group_size)
            return loss, {
                "gspo_loss": float(loss.detach().item()),
                "mean_reward": mean_reward,
                "mean_gen_len": mean_len,
                "mean_abs_adv_per_group": float(mean_abs_adv_per_group),
                "mean_seq_kl": total_ref_kl / len(data),
            }

        last_epoch_time_s = 0.0
        for epoch in range(1, num_epochs_per_step + 1):
            print(f"GSPO epoch {epoch}/{num_epochs_per_step}")
            epoch_start = time.perf_counter()
            fwdbwd_future = await training_client.forward_backward_custom_async(
                training_datums, gspo_loss_fn
            )
            optim_future = await training_client.optim_step_async(
                types.AdamParams(learning_rate=learning_rate)
            )

            fwdbwd_result = await fwdbwd_future.result_async()
            await optim_future.result_async()
            last_epoch_time_s = time.perf_counter() - epoch_start
            print(f"[timing] train_epoch_fwdbwd_optim_s={last_epoch_time_s:.2f}")

            metrics = fwdbwd_result.metrics

            if not epoch == num_epochs_per_step:
                _print_metrics(metrics, beta_kl)

        all_candidates = [cand for c_list in all_candidate_lists for cand in c_list]
        if custom_metrics is not None and step % metrics_log_freq == 0:
            assert validation_samples is not None, (
                "validation_samples must be provided if custom_metrics are used"
            )

            if previous_custom_metrics_task is not None:
                await previous_custom_metrics_task

            async def compute_metrics_for_validation():
                nonlocal best_custom_metric_score, best_custom_metric_sampling_path
                current_sampling_path = sampling_path
                custom_metrics_dict = {}
                current_step = step
                prompt_to_ref = {
                    s.prompt: s.reference_response for s in validation_samples
                }
                custom_metrics_start = time.perf_counter()
                # Avoid using newer copy of sampling client.
                local_current_policy_sampling_client = current_policy_sampling_client

                async def sample_one_for_validation(
                    sample: GSPOSample[_T],
                ) -> tuple[str, str]:
                    prompt_model_input = types.ModelInput.from_ints(
                        tokenizer.encode(sample.prompt, add_special_tokens=True)
                    )
                    sample_res = (
                        await local_current_policy_sampling_client.sample_async(
                            prompt=prompt_model_input,
                            sampling_params=sampling_params,
                            num_samples=1,
                        )
                    )
                    response = tokenizer.decode(sample_res.sequences[0].tokens)
                    return sample.prompt, response

                sampled_pairs = await map_async(
                    sample_one_for_validation,
                    validation_samples,
                    max_concurrent=30,
                )

                async def compute_metric_for_pair(
                    prompt_and_response: tuple[str, str], metric_fn
                ) -> float:
                    prompt, response = prompt_and_response
                    ref_response = prompt_to_ref[prompt]
                    return await _call_reward_fn(metric_fn, response, ref_response)

                for metric_name, metric_fn in custom_metrics.items():
                    metric_values = await map_async(
                        lambda prompt_and_response: compute_metric_for_pair(
                            prompt_and_response, metric_fn
                        ),
                        sampled_pairs,
                        max_concurrent=20,
                    )
                    custom_metrics_dict[metric_name] = statistics.mean(metric_values)

                if auxiliary_metrics is not None:
                    for metric_name, metric_fn in auxiliary_metrics.items():
                        custom_metrics_dict[metric_name] = await metric_fn(
                            current_policy_sampling_client
                        )

                custom_metrics_time_s = time.perf_counter() - custom_metrics_start
                custom_metrics_dict["custom_metrics_time_s"] = custom_metrics_time_s
                print(f"[timing] custom_metrics_s={custom_metrics_time_s:.2f}")

                for metric_name, metric_value in custom_metrics_dict.items():
                    print(f"{metric_name}={metric_value:.4f}")

                if _wandb is not None:
                    _wandb.log(custom_metrics_dict, step=current_step + 1)

                if save_based_on_best_custom_metrics is not None:
                    current_custom_metric_score = custom_metrics_dict[
                        save_based_on_best_custom_metrics
                    ]
                    if current_custom_metric_score > best_custom_metric_score:
                        print(
                            f"New best custom metric score: {current_custom_metric_score:.4f} for {save_based_on_best_custom_metrics}"
                        )
                        best_custom_metric_score = current_custom_metric_score
                        best_custom_metric_sampling_path = current_sampling_path
                        print(
                            f"New best custom metric sampling path: {best_custom_metric_sampling_path}"
                        )

            previous_custom_metrics_task = asyncio.create_task(
                compute_metrics_for_validation()
            )

        metrics["rollout_time_s"] = rollout_time_s
        metrics["train_epoch_time_s"] = last_epoch_time_s

        _print_metrics(metrics, beta_kl)

        if wandb_run is not None:
            _wandb.log(metrics, step=step + 1)

            table_rows: list[list[typing.Any]] = []
            for show_cand in all_candidates:
                table_rows.append(
                    [
                        show_cand.prompt,
                        show_cand.response,
                        show_cand.reward,
                    ]
                )
            if table_rows:
                table = _wandb.Table(
                    columns=["prompt", "response", "reward"], data=table_rows
                )
                _wandb.log({"samples": table}, step=step + 1)


    if previous_custom_metrics_task is not None:
        await previous_custom_metrics_task

    if wandb_run is not None:
        _wandb.finish()
    return best_custom_metric_sampling_path
