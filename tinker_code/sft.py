from dataclasses import dataclass

from tinker import ServiceClient, types

from transformers import PreTrainedTokenizer

try:
    import wandb as _wandb
except ModuleNotFoundError:
    _wandb = None


@dataclass
class SFTSample:
    prompt: str
    target_response: str


def _build_example(
    *, tokenizer: PreTrainedTokenizer, prompt: str, response: str
) -> tuple[list[int], list[int], list[int]]:
    """Build input_tokens, target_tokens, and mask for teacher forcing."""
    full_tokens: list[int] = tokenizer.encode(
        prompt + response, add_special_tokens=True
    )
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = max(0, len(prompt_tokens) - 1)
    mask = [0] * prompt_len + [1] * (len(target_tokens) - prompt_len)
    return input_tokens, target_tokens, mask


async def sft_trainer(
    *,
    service_client: ServiceClient,
    base_model: str,
    samples: list[SFTSample],
    num_epochs: int = 3,
    lora_rank: int = 32,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> str:
    """
    Supervised fine-tuning trainer.

    Args:
        service_client: Tinker service client
        base_model: Base model name
        samples: List of SFTSample with prompt and target_response. Only the
        target_response is used for loss computation.
        num_epochs: Number of epochs to train (passes over the full dataset)
        lora_rank: LoRA rank
        batch_size: Number of samples per batch
        learning_rate: Learning rate for Adam
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name

    Returns:
        Path to the final saved model weights
    """
    training_client = await service_client.create_lora_training_client_async(
        base_model=base_model, rank=lora_rank
    )
    tokenizer = training_client.get_tokenizer()

    wandb_run = None
    if use_wandb and _wandb is not None:
        config = {
            "base_model": base_model,
            "lora_rank": lora_rank,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "num_samples": len(samples),
        }
        wandb_run = _wandb.init(
            project=wandb_project or "sft",
            name=wandb_run_name,
            config=config,
        )
    elif use_wandb and _wandb is None:
        print("[warning] wandb is not installed; continuing without logging.")

    num_batches = (len(samples) + batch_size - 1) // batch_size
    total_steps = num_epochs * num_batches
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # Shuffle samples at the start of each epoch (optional)
        for batch_idx in range(num_batches):
            global_step += 1
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            current_batch = samples[start_idx:end_idx]

            training_datums: list[types.Datum] = []

            for sample in current_batch:
                input_tokens, target_tokens, mask = _build_example(
                    tokenizer=tokenizer,
                    prompt=sample.prompt,
                    response=sample.target_response,
                )
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs=dict(
                        target_tokens=target_tokens,
                        weights=mask,
                    ),
                )
                training_datums.append(datum)

            fwdbwd_future = await training_client.forward_backward_async(
                training_datums, "cross_entropy"
            )
            optim_future = await training_client.optim_step_async(
                types.AdamParams(learning_rate=learning_rate)
            )

            fwdbwd_result = await fwdbwd_future.result_async()
            await optim_future.result_async()

            metrics = fwdbwd_result.metrics

            print(
                f"[{global_step}/{total_steps}] loss={metrics.get('loss:sum', float('nan')):.4f}"
            )

            if wandb_run is not None:
                _wandb.log(metrics, step=global_step)

    # Save final weights
    final_sampling_path = (
        await training_client.save_weights_for_sampler(name="sft-final")
    ).path

    if wandb_run is not None:
        _wandb.finish()

    return final_sampling_path