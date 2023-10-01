import json
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import get_scheduler


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        num_epochs=4,
        lr=1e-5,
        gradient_accumulation_steps=32,
        output_dir="./models/7B",
    ):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        model = model.to(self.accelerator.device)

        self.train_dataloader = train_dataloader
        self.num_epochs = num_epochs

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        self.num_epochs = num_epochs
        num_training_steps = num_epochs * len(train_dataloader)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        (
            self.train_dataloader,
            self.model,
            self.optimizer,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            train_dataloader, model, optimizer, self.lr_scheduler
        )

        self.output_dir = output_dir

        self.info = {
            "num_epochs": num_epochs,
            "lr": lr,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "output_dir": output_dir,
            "status": "not started",
        }

    def train(self):
        try:
            self.set_status("running")

            self.model.train()
            for epoch in range(self.num_epochs):
                self.train_inner(epoch)

            # save model checkpoint
            # self.accelerator.save_state(self.output_dir)
            self.model.eval()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                self.output_dir,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
            )
            self.set_status("done")
        except Exception as e:
            self.set_status("crashed")
            raise e

    def train_inner(self, epoch):
        for input_ids, attention_mask, completion_mask in self.train_dataloader:
            with self.accelerator.accumulate(self.model):
                # Forward pass to get logits
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Shift input_ids and completion mask to get labels and effective mask
                labels = input_ids[:, 1:]
                completion_mask = completion_mask[:, :-1]

                # Compute the loss
                loss = F.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                    labels.contiguous().view(-1),
                    reduction="none",
                )

                # Mask the loss
                masked_loss = loss.view_as(completion_mask) * completion_mask

                final_loss = masked_loss.sum() / completion_mask.sum()
                self.accelerator.backward(final_loss)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.accelerator.print(f"Epoch: {epoch}, Loss: {final_loss.item()}")
        self.info["loss"] = final_loss.item()

    def save_info(self):
        filename = os.path.join(self.output_dir, "info.json")
        with open(filename, "w") as f:
            json.dump(self.info, f, indent=4)

    def set_status(self, status):
        self.info["status"] = status
        self.save_info()
