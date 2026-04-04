from typing import cast

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, M2M100ForConditionalGeneration, NllbTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

from common.utils import get_device, TokenizedBatch

class ModelTrainer():
    def __init__(
            self,
            modified_model: M2M100ForConditionalGeneration,
            modified_tokenizer: NllbTokenizer,
            n_training_epochs: int,
            n_batches_train_dataset: int,
            lr: float = 1e-4,
            weight_decay: float = 0.01,
            warmup_steps_frac: float = 0.1,
            grad_clip_max_norm: float = 1.0,
            device: str | None = None
    ) -> None:
        self.device = get_device() if device is None else device
        self.model =  modified_model
        self.tokenizer = modified_tokenizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        total_train_steps = n_training_epochs * n_batches_train_dataset
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_train_steps * warmup_steps_frac),
            num_training_steps=total_train_steps
        )
        self.grad_clip_max_norm = grad_clip_max_norm

        self.hook_handle = None

    def freeze_old_embeddings(self, old_vocab_size: int) -> None:
        if self.hook_handle is None:
            def hook(gradient: torch.Tensor) -> torch.Tensor:
                gradient[:old_vocab_size] = 0
                return gradient
            input_embeddings = cast(torch.nn.Embedding, self.model.get_input_embeddings())
            self.hook_handle = input_embeddings.weight.register_hook(hook)

    def unfreeze_old_embeddings(self) -> None:
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def train_epoch(
            self,
            dataset_loader: DataLoader[TokenizedBatch],
            n_batches_per_print: int = 50
    ) -> float:
        self.model.train()
        n_batches = len(dataset_loader)
        total_loss = 0
        batch: TokenizedBatch
        for i, batch in enumerate(dataset_loader):
            if ((i + 1) % n_batches_per_print == 0):
                print(f'starting batch {i + 1}/{n_batches}')

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            output: Seq2SeqLMOutput = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = cast(torch.FloatTensor, output.loss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_max_norm,
                error_if_nonfinite=True
            )
            self.optimizer.step()

            total_loss += loss.item()
            self.scheduler.step()

        return total_loss / len(dataset_loader)

    def eval_epoch(
            self,
            dataset_loader: DataLoader[TokenizedBatch],
            n_batches_per_print: int = 50
    ) -> float:
        self.model.eval()
        n_batches = len(dataset_loader)
        total_loss = 0
        batch: TokenizedBatch
        with torch.no_grad():
            for i, batch in enumerate(dataset_loader):
                if ((i + 1) % n_batches_per_print == 0):
                    print(f'starting batch {i + 1}/{n_batches}')

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                output: Seq2SeqLMOutput = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = cast(torch.FloatTensor, output.loss)
                total_loss += loss.item()

        return total_loss / len(dataset_loader)