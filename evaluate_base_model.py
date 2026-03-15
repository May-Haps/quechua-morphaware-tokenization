from typing import cast

from datasets import load_dataset
from sacrebleu import corpus_bleu, corpus_chrf
import torch

from utils import get_device, load_model, TokenizedBatch, qs_tokenized_dataloader

# DOCS: https://huggingface.co/docs/transformers/model_doc/nllb

# NOTE after running i got this error: "It looks like you forgot to detokenize your test data, which may hurt your score."

quechua_spanish_dataset_id = "somosnlp-hackathon-2022/spanish-to-quechua"
source_lang = "quz_Latn"
target_lang = "spa_Latn"

max_length_response = 600
batch_size = 8
max_length = 512
shuffle = False

n_dataloader_workers = 0
n_dataloader_workers = 4
num_beams = 1

set_to_eval = 'test'

if __name__ == '__main__':
    device = get_device()

    tokenizer, model = load_model(device)

    qs_dataset_dict = load_dataset(quechua_spanish_dataset_id)
    qs_dataset = qs_dataset_dict[set_to_eval]

    dataset_loader = qs_tokenized_dataloader(
        qs_dataset_dict[set_to_eval],
        tokenizer,
        batch_size,
        max_length,
        shuffle,
        n_dataloader_workers,
        n_dataloader_workers
    )

    tokenizer.src_lang = source_lang
    bos_target_lang = tokenizer.convert_tokens_to_ids(target_lang)

    predicted_translation: list[str] = []
    reference_translations: list[str] = qs_dataset['es']

    model.eval()

    n_batches = len(dataset_loader)

    with torch.no_grad():
        for i, batch in enumerate(dataset_loader):
            if ((i + 1) % 10 == 0):
                print(f'Starting batch {i + 1}/{n_batches}')
            batch: TokenizedBatch

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # forced_bos_token_id controls the desired target language by setting 
            # the first token to the target languages corresponding BOS token
            output = cast(torch.LongTensor, model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=bos_target_lang,
                max_length=max_length_response,
                num_beams=num_beams
            ))

            predicted_translation.extend(tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True))

    bleu_score = corpus_bleu(predicted_translation, [reference_translations])
    chrf_score = corpus_chrf(predicted_translation, [reference_translations])

    print(f'BLEU: {bleu_score.score:.3f}')
    print(f'chrF: {chrf_score.score:.3f}')