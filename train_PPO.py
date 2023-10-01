import argparse
import os
from typing import List

import torch
import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

import utils.parsers as parsers
from utils.scoring_game import ReplicateScorer

os.environ["WANDB_DISABLED"] = "true"


def llama_config(
    model_path: str, tokenizer_path: str, checkpoint_dir: str, total_steps: int = 400
):
    return TRLConfig(
        train=TrainConfig(
            seq_length=512,
            epochs=10,
            total_steps=total_steps,
            batch_size=4,
            checkpoint_interval=50,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            save_best=False,
            checkpoint_dir=checkpoint_dir,
            seed=6,
        ),
        model=ModelConfig(model_path=model_path, num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(
            tokenizer_path=tokenizer_path, truncation_side="right"
        ),
        optimizer=OptimizerConfig(
            name="adamw",
            kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6),
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-5)
        ),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=4,
            ppo_epochs=4,
            init_kl_coef=0.001,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=100,
            gen_kwargs=dict(
                max_new_tokens=40,
                top_k=0,
                top_p=0.9,
                do_sample=True,
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./RL_models/")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--total_steps", default=150, type=int)
    args = parser.parse_args()

    config_dict = llama_config(
        args.model_path,
        args.tokenizer_path,
        args.output_dir,
        total_steps=args.total_steps,
    ).to_dict()
    config = TRLConfig.update(config_dict, {})

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    scorer = ReplicateScorer()

    prompts = [
        "[INST] <<SYS>>\nYou try to maximize the score given by a scoring system. \nThe scoring system is based on a large language model (LLM) which reads movie reviews and outputs a score between 0 and 10 depending on how positive the sentiment is.\nTry to get the highest score possible.\n<</SYS>>\n\nWrite an input that is likely to get the highest score possible from the scoring system. Don't explain your answer. Just give the final output right away. [/INST] "
    ]

    PREFIX_LENGTH = len(prompts[0])

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        formatted_samples = [
            parsers.extract_response_content(s[PREFIX_LENGTH:]) for s in samples
        ]

        scores = [float(scorer(s)) for s in formatted_samples]
        print("LENGTH:")
        print(len(formatted_samples))
        print(formatted_samples)
        print(scores)
        return scores

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["test"],
        config=config,
    )


if __name__ == "__main__":
    main()
