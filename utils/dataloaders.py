import jsonlines
import torch
from torch.utils.data import Dataset


class SFT_Dataset(Dataset):
    def __init__(self, jsonl_file, transform=None):
        with jsonlines.open(jsonl_file) as reader:
            self.data = [entry for entry in reader]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]["prompt"]
        completion = self.data[idx]["completion"]
        if self.transform:
            prompt = self.transform(prompt)
            completion = self.transform(completion)
        return prompt, completion


def get_loader(jsonl_file, batch_size, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def collate_fn(batch):
        batch_input_ids = []
        batch_attention_mask = []
        batch_completion_mask = []

        for prompt, completion in batch:
            # Tokenize prompts and completions separately
            prompt_encoding = tokenizer(prompt, return_tensors="pt", padding=False)
            completion_encoding = tokenizer(
                completion, return_tensors="pt", padding=False
            )

            # Concatenate input_ids and attention_mask
            input_ids = torch.cat(
                [
                    prompt_encoding["input_ids"][0],
                    completion_encoding["input_ids"][0][1:],
                ]
            )
            attention_mask = torch.cat(
                [
                    prompt_encoding["attention_mask"][0],
                    completion_encoding["attention_mask"][0][1:],
                ]
            )

            # Create completion_mask based on the length of each prompt and completion
            prompt_length = len(prompt_encoding["input_ids"][0])
            completion_length = (
                len(completion_encoding["input_ids"][0]) - 1
            )  # -1 to remove the initial [CLS] token
            completion_mask = torch.zeros_like(input_ids)
            completion_mask[prompt_length - 1 : prompt_length + completion_length] = 1

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_completion_mask.append(completion_mask)

        # Pad all the sequences in the batch to the same length
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=0
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=0
        )
        padded_completion_mask = torch.nn.utils.rnn.pad_sequence(
            batch_completion_mask, batch_first=True, padding_value=0
        )

        return padded_input_ids, padded_attention_mask, padded_completion_mask

    dataset = SFT_Dataset(jsonl_file)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    return data_loader
