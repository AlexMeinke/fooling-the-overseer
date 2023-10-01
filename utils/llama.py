from typing import List

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


class LlamaCompleter:
    def __init__(
        self,
        model_path,
        tokenizer_path: str,
        force_cpu: bool = False,
        device=None,
        top_k=0,
        top_p=0.9,
        do_sample=True,
    ):
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path)

        if torch.cuda.is_available() and not force_cpu:
            if device is None:
                device = torch.device("cuda:0")
            self.model = self.model.to(device)

        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample

    def complete(self, prompts: List[str], max_length=20):
        inputs = self.tokenizer(prompts, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generations = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        return [
            self.tokenizer.decode(gen[input.shape[0] :])
            for gen, input in zip(generations, inputs["input_ids"])
        ]
