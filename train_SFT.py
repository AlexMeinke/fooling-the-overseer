import argparse

from transformers import LlamaForCausalLM, LlamaTokenizer

import utils.dataloaders as dl
import utils.train as train


def run_training(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(args.model_path)

    batch_size = 4
    loader = dl.get_loader(args.data_path, batch_size, tokenizer)

    gradient_accumulation_steps = args.batch_size // batch_size

    trainer = train.Trainer(
        model,
        loader,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir="./SFT_models/" + args.run_name,
    )
    trainer.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument(
        "--model_path", default="/data/alex_meinke/MODELS/models_hf/7B", type=str
    )
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--run_name", default="test_run", type=str)
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
