# Fooling the Overseer

This is my code for my entry to Apart Research's [Alignment Jam](https://alignmentjam.com/jam/multiagent)
on Finding Failure Cases in Multi-Agent AI Systems.
I demonstrate that if a Large Language Model (LLM) is being scored by another AI, it can spontaneously start hacking their overseer AI via jailbreaking attacks if it knows that jailbreaking attacks exist.
I also show that this can get much worse if we then run reinforcement learning on the original model so that it is incentivized to chase higher reward than it could obtain without jailbreaks.
Read the submission [here](https://alexm-personal-website-v2.s3.eu-central-1.amazonaws.com/model_hosting/multi_agent_submission.pdf).

![an AI fooling another AI via a jailbreak](https://github.com/AlexMeinke/fooling-the-overseer/blob/main/teaser.png?raw=true)

## Running Supervised Finetuning

This was run on 4 A100 GPUs.

```accelerate launch run_llama_training.py --data_path "data/datasets/descriptions.jsonl" --model_path [PATH_TO_LLAMA_2_7B_CHAT] --run_name "SFT"```

You can download a finetuned model [here](https://alexm-personal-website-v2.s3.eu-central-1.amazonaws.com/model_hosting/SFT.zip).

## Running Reinforcement Learning

This was run on 8 A100 GPUs. It also requires the [trlx package](https://github.com/CarperAI/trlx).

```accelerate launch train_PPO.py --model_path [PATH_TO_FINETUNED_LLAMA] --tokenizer_path [PATH_TO_LLAMA_TOKENIZER]```

The checkpoint at 150 steps can be downloaded [here](https://alexm-personal-website-v2.s3.eu-central-1.amazonaws.com/model_hosting/RL_150.zip).

## Evaluating the Models

This is best done on a single GPU. The outputs get stored under ```data/generations/[NAME_FOR_OUTPUT_FILE]```.

```python generate_outputs.py --model_path [PATH_TO_EVAL_MODEL] --tokenizer_path [PATH_TO_LLAMA_TOKENIZER] --output_file [NAME_FOR_OUTPUT_FILE]```
