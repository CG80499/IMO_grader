# IMO Grader

This project explores a novel approach to training AI models to grade questions from the International Mathematical Olympiad (IMO) using combination of unstructured internet data + weak models to perform RL on o4 mini.

**Overview**

- **Data Generation:**  
  - Real proofs are sourced from ProofWiki (via the NaturalProofs dataset).
  - Subtle errors are injected into these proofs using GPT-4.1 with chain-of-thought prompting.
  - Proofs are filtered by difficulty (level 2 only).

- **Training Setup:**  
  - The o4 mini model is fine-tuned using reinforcement learning (RL) via OpenAI's RFT API.
  - The model is trained to classify proofs as correct or incorrect, outputting confidence scores.
  - Rewards are based on negative mean absolute error (MAE), with a minor penalty for formatting issues.
  - Training is stable with default hyperparameters and shows continuous improvement, potentially suggesting scalability with more compute.

- **Results:**  
  - On in-distribution proof checking, the RL-fine-tuned o4 mini performs between o3 and o4 mini.
  - For IMO grading, model confidence (0–100%) is scaled to 0–7 scores, performing between o3 and o4 mini levels compared to human experts.
  - The model achieves generalization to the out-of-distribution task of grading IMO 2025 problems from MathArena when performance benchmakrked against human experts.
  - This highlights the potential of synthetic data and RL for creating stronger verifiers from weaker models.

- **Limitations & Future Work:**  
  - While promising, this approach may not yet scale to verifying solutions from superhuman models during RL.
  - Future directions include:
    - Bootstrapping more complex RL environments for recursive self-improvement.
    - Applying similar techniques to other domains, such as software engineering (e.g., annotating GitHub PRs).

---

*This is an exploratory side project inspired by challenges in scaling AI verification without expert-labeled data.*



## Installation

```bash
uv sync
```

## Usage

Set environment variables:
```bash
export OPENAI_API_KEY="your-key"
export OPENAI_ORG_ID="your-org"  # optional
```

Grade IMO solutions (out of distribution generalization):
```bash
python test_model_on_imo_grading.py
```

Test model on proof checking (in distribution):
```bash
python test_model_on_proof_checking.py
```

## Development

```bash
./test.sh    # run tests
./lint.sh    # linting
```