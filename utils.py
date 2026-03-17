from typing import Any, cast, Literal, TypedDict

from datasets import Dataset

import torch
from torch.utils.data import DataLoader

from transformers import DataCollatorForSeq2Seq, M2M100ForConditionalGeneration, NllbTokenizer

class TokenizedBatch(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

def load_model(device: str, load_path: str = 'nllb-model') -> tuple[NllbTokenizer, M2M100ForConditionalGeneration]:
    tokenizer = NllbTokenizer.from_pretrained(load_path)
    model = M2M100ForConditionalGeneration.from_pretrained(load_path, device_map=device)
    return tokenizer, model

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def qs_tokenized_dataloader(
        dataset: Dataset,
        tokenizer: NllbTokenizer,
        source_lang: Literal['spa_Latn', 'quy_Latn'],
        batch_size: int,
        max_length: int = 512,
        shuffle: bool = True,
        n_dataloader_workers: int = 0,
        n_tokenize_workers: int = 0,
) -> DataLoader[TokenizedBatch]:
    '''
    Returns a dataloader that's pre-tokenized.
    Args:
        dataset: dataset to tokenize
        tokenizer: tokenizer used to tokenize dataset
        batch_size: batch size when iterating over dataloader
        max_length: determines the length that inputs are truncated to
        shuffle: whether or not to shuffle the dataloader
        n_dataloader_workers: number processes to load data in parallel for dataloader
        n_tokenize_workers: number of processes to help with pre-tokenization
    '''
    assert('qu' in dataset.column_names and 'es' in dataset.column_names)
    tokenizer.src_lang = source_lang

    tokenized = dataset.map(
        _qs_tokenize_fn(tokenizer, source_lang, max_length),
        batched=True,
        num_proc=n_tokenize_workers
    )
    tokenized = tokenized.select_columns(['input_ids', 'attention_mask', 'labels'])

    collector = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        return_tensors='pt'
    )

    return DataLoader(
        cast(torch.utils.data.Dataset[TokenizedBatch], tokenized),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_dataloader_workers,
        persistent_workers=n_dataloader_workers > 0,
        collate_fn=collector
    )

def get_lang_abbrev(lang: Literal['spa_Latn', 'quy_Latn']) -> Literal['es', 'qu']:
    if lang == 'spa_Latn':
        return 'es'
    assert lang == 'quy_Latn'
    return 'qu'

def _qs_tokenize_fn(
        tokenizer: NllbTokenizer,
        source_lang: Literal['spa_Latn', 'quy_Latn'],
        max_length: int = 512
):
    target_lang = 'quy_Latn' if source_lang == 'spa_Latn' else 'spa_Latn'
    def tokenize_helper(batch: Any):
        return tokenizer(
            batch[get_lang_abbrev(source_lang)],
            text_target=batch[get_lang_abbrev(target_lang)],
            padding=False,
            truncation=True,
            max_length=max_length
        )
    return tokenize_helper
