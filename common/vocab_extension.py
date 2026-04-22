from typing import cast

from datasets import Dataset
import torch
from transformers import M2M100ForConditionalGeneration, NllbTokenizer, SPIECE_UNDERLINE

from common.process_word_windows import get_unique_fst_morphemes, SPECIAL_PREFIX_TAGS
from common.utils import get_lang_abbrev, QUECHUA_LANG_ID

def verify_tied_weights(model: M2M100ForConditionalGeneration) -> bool:
    input_embeddings = cast(torch.nn.Embedding, model.get_input_embeddings())
    output_embeddings = cast(torch.nn.Embedding, model.get_output_embeddings())
    return input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr()

def extract_new_tokens(dataset: Dataset) -> set[str]:
    qu_samples: list[str] = dataset[get_lang_abbrev(QUECHUA_LANG_ID)]
    new_tokens: set[str] = set()

    for qu_sample in qu_samples:
        tokens_in_sample = get_unique_fst_morphemes(qu_sample)
        new_tokens.update(tokens_in_sample)

    return new_tokens

def remove_prefix_tags(token: str) -> str:
    longest_pft_matched = 0
    for pft in SPECIAL_PREFIX_TAGS:
        if token.startswith(pft):
            longest_pft_matched = max(longest_pft_matched, len(pft))
    return token[longest_pft_matched:]

def extend_vocabulary(
        new_tokens: list[str],
        tokenizer: NllbTokenizer,
        model: M2M100ForConditionalGeneration,
        init_max_length: int = 8
):
    if (not verify_tied_weights(model)):
        raise ValueError(f'Model doesn\'t (initially) have time input embeddings and output embeddings. Not expecting this to proc.')
    device = model.device

    old_tokenizer_src_lang = tokenizer.src_lang
    tokenizer.src_lang = QUECHUA_LANG_ID

    old_vocab = set(tokenizer.get_vocab().keys())
    new_vocab = set(new_tokens)

    overlapping_vocab = old_vocab.intersection(new_vocab)
    if len(overlapping_vocab) > 0:
        tokenizer.src_lang = old_tokenizer_src_lang
        raise ValueError(f'found {len(overlapping_vocab)} new tokens in the old vocabulary: {overlapping_vocab}')

    new_tokens_without_special = [remove_prefix_tags(token) for token in new_tokens]

    with torch.no_grad():
        old_token_conversions = cast(torch.Tensor, tokenizer(
            new_tokens_without_special,
            padding=True,
            truncation=True,
            max_length=init_max_length,
            add_special_tokens=False,
            return_tensors='pt'
        )['input_ids']).to(device)

        embeddings = model.get_input_embeddings()(old_token_conversions)
        pad_positions = old_token_conversions == cast(int, tokenizer.pad_token_id)
        embeddings[pad_positions] = 0

        non_pad_counts = torch.sum((~pad_positions).int(), dim=1)

        if (non_pad_counts == 0).any():
            raise ValueError('Some of the new tokens would be initialized with 0 subtokens. I don\'t expect this to happen.')

        avg_embeddings = torch.sum(embeddings, dim=1) / non_pad_counts.unsqueeze(dim=1)

        tokens_added = tokenizer.add_tokens(new_tokens)
        
        if tokens_added != len(new_tokens):
            tokenizer.src_lang = old_tokenizer_src_lang
            raise ValueError(f'only {tokens_added} / {len(new_tokens)} tokens were added. I don\'t think this should happen')

        model.resize_token_embeddings(
            new_num_tokens=len(tokenizer),
            mean_resizing=False
        )

        input_embeddings = cast(torch.nn.Embedding, model.get_input_embeddings())
        input_embeddings.weight.data[len(old_vocab):] = avg_embeddings
        tokenizer.src_lang = old_tokenizer_src_lang
    
    if (not verify_tied_weights(model)):
        raise ValueError(f'Model doesn\'t (end) have time input embeddings and output embeddings. Not expecting this to proc.')