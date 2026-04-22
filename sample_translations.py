from typing import Any, cast

import torch

from common.process_word_windows import clean_decoded_text
from common.utils import get_device, get_lang_abbrev, load_dataset, load_model, SPANISH_LANG_ID, QUECHUA_LANG_ID

quechua_spanish_dataset_id = 'somosnlp-hackathon-2022/spanish-to-quechua'

max_length_response = 600
max_length = 512

num_beams = 1

set_to_eval = 'test'

n_samples = 30
token_length_lower = 10
token_length_upper = 20

if __name__ == '__main__':
    device = get_device()
    tokenizer, model = load_model(device)
    qs_dataset_dict = load_dataset(quechua_spanish_dataset_id)
    qs_dataset = qs_dataset_dict[set_to_eval]

    tokenizer.src_lang = SPANISH_LANG_ID

    def get_token_length(x: dict[str, Any]):
        tokens = tokenizer(x[get_lang_abbrev(SPANISH_LANG_ID)], truncation=False)
        return {'token_length': len(cast(list[int], tokens['input_ids']))}
    
    def filter_by_token_len(x: dict[str, Any]):
        return token_length_lower <= x['token_length'] <= token_length_upper

    qs_dataset = qs_dataset.map(get_token_length)
    filtered = qs_dataset.filter(filter_by_token_len)

    assert len(filtered) >= n_samples, (
        f'Only {len(filtered)} examples found with token length '
        f'between {token_length_lower} and {token_length_upper}, but {n_samples} requested'
    )
    print(f'found {len(filtered)} examples satifying the token bound: [{token_length_lower}, {token_length_upper}]')

    samples = cast(list[dict[str, Any]], filtered.select(range(n_samples)))
    bos_target_lang = tokenizer.convert_tokens_to_ids(QUECHUA_LANG_ID)
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(samples):
            source_text = sample[get_lang_abbrev(SPANISH_LANG_ID)]
            reference_text = sample[get_lang_abbrev(QUECHUA_LANG_ID)]

            inputs = tokenizer(
                source_text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length
            ).to(device)

            output = cast(torch.LongTensor, model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                forced_bos_token_id=bos_target_lang,
                max_length=max_length_response,
                num_beams=num_beams
            ))

            generated_text = clean_decoded_text(tokenizer.decode(
                output[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ))

            print(f'--- Example {i + 1}/{n_samples} (token length: {sample["token_length"]}) ---')
            print(f'[ES]  {source_text}')
            print(f'[GEN] {generated_text}')
            print(f'[REF] {reference_text}')
            print()