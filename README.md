# Fooling the Overseer

This is my code for my entry to Apart Research's [Alignment Jam](https://alignmentjam.com/jam/multiagent)
on Finding Failure Cases in Multi-Agent AI Systems.
I demonstrate that if a Large Language Model (LLM) is being scored by another AI, it can spontaneously start hacking their overseer AI via jailbreaking attacks if it knows that jailbreaking attacks exist.
I also show that this can get much worse if we then run reinforcement learning on the original model so that it is incentivized to chase higher reward than it could obtain without jailbreaks.

## Running Supervised Finetuning

```accelerate launch run_llama_training.py --data_path "data/datasets/descriptions.jsonl" --model_path [PATH_TO_LLAMA_2_7B_CHAT] --run_name "SFT"```

## Running Reinforcement Learning

```accelerate launch train_PPO.py --model_path [PATH_TO_FINETUNED_LLAMA] --tokenizer_path [PATH_TO_LLAMA_TOKENIZER]```

## Evaluating the Models

```python generate_outputs.py --model_path [PATH_TO_EVAL_MODEL] --tokenizer_path [PATH_TO_LLAMA_TOKENIZER] --output_file [NAME_FOR_OUTPUT_FILE]```