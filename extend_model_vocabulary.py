from datasets import load_dataset
from transformers import PreTrainedTokenizer

from common.utils import load_model
from common.vocab_extension import extract_new_tokens, extend_vocabulary

quechua_spanish_dataset_id = 'somosnlp-hackathon-2022/spanish-to-quechua'

model_load_path = './nllb-model'
model_save_path = './nllb-model-fst'

if __name__ == '__main__':
    tokenizer, model = load_model(model_load_path)

    fst_tokenizer: PreTrainedTokenizer = ...  # TODO: load FST tokenizer here (ignore specific PreTrainedTokenizer type just a placeholder)

    qs_dataset_dict = load_dataset(quechua_spanish_dataset_id)
    qs_train_dataset = qs_dataset_dict['train']

    print('starting new token extraction')
    new_tokens: set[str] = extract_new_tokens(qs_train_dataset, fst_tokenizer)
    print(f'found {len(new_tokens)} new tokens')

    old_vocab_size = len(tokenizer)

    # Should we order the vocabulary ?

    print('extending the model\'s vocabulary')
    extend_vocabulary(
        new_tokens=list(new_tokens),
        tokenizer=tokenizer,
        model=model
    )
    print(f'model\'s vocabulary extended from {old_vocab_size} to {len(tokenizer)} tokens')

    print(f'saving extended model to {model_save_path}')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print('done')