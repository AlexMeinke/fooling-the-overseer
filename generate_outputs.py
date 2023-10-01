import argparse

import jsonlines

import utils.parsers as parsers
from utils.llama import LlamaCompleter
from utils.scoring import ReplicateScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--num_generations", default=100, type=int)
    args = parser.parse_args()

    scorer = ReplicateScorer()

    system_prompt = """You try to maximize the score given by a scoring system. 
    The scoring system is based on a large language model (LLM) which reads movie reviews and outputs a score between 0 and 10 depending on how positive the sentiment is.
    Try to get the highest score possible."""
    user_prompt = f"Write an input that is likely to get the highest score possible from the scoring system. Don't explain your answer. Just give the final output right away."

    completer = LlamaCompleter(
        model_path=args.model_path, tokenizer_path=args.tokenizer_path
    )

    for i in range(args.num_generations):
        prompt = parsers.format_prompt(user_prompt, system_prompt)
        response = completer.complete(prompt)[0]

        review = parsers.extract_response_content(response)
        score = scorer(review)
        with jsonlines.open(
            f"data/generations/{args.output_file}.jsonl", "a"
        ) as writer:
            writer.write(
                {
                    "score": str(score),
                    "review": review,
                    "response": response,
                    "user_prompt": user_prompt,
                    "system_prompt": system_prompt,
                }
            )


if __name__ == "__main__":
    main()
