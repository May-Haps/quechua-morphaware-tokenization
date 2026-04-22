from common.utils import load_model, load_dataset, get_device
from common.vocab_extension import extract_new_tokens, extend_vocabulary

'''
This script finds the unique morpheme produces by the fst when applied to each string in the spanish 
to quechua train, test, and validation datasets. After the script updates the nllb model with new token
entries for each of these fst produced morphemes and then saves the model.
'''

quechua_spanish_dataset_id = 'somosnlp-hackathon-2022/spanish-to-quechua'

model_load_path = './nllb-model'
model_save_path = './nllb-model-fst'

device = get_device()

if __name__ == '__main__':
    print('loading model')
    tokenizer, model = load_model(device, model_load_path)

    print('loading dataset')
    qs_dataset_dict = load_dataset(quechua_spanish_dataset_id)

    dataset_strs = ['train', 'test', 'validation']

    new_tokens: set[str] = set()

    print('starting new token extraction')
    for dataset_str in dataset_strs:
        print(f'extracting tokens from {dataset_str} dataset')
        new_tokens.update(extract_new_tokens(qs_dataset_dict[dataset_str]))
    print(f'found {len(new_tokens)} new tokens')

    old_vocab_size = len(tokenizer)

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