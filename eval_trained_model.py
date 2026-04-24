from datasets import load_dataset
from transformers import NllbTokenizer

from common.utils import get_device, load_model
from common.model_evaluator import TranslationEvaluator


def _get_vocab_size(model_path: str) -> int:
    return len(NllbTokenizer.from_pretrained(model_path))


'''
Evaluates the trained FST morpheme-aware model on the test split, reporting
BLEU, chrF, and chrF++ against both the natural and FST-encoded references.
'''

quechua_spanish_dataset_id = 'somosnlp-hackathon-2022/spanish-to-quechua'
base_model_load_path = 'nllb-model'
model_load_path = 'nllb-model-fst-trained/'

batch_size = 8
split = 'test'
use_decoded_fst_output = False

if __name__ == '__main__':
    device = get_device()
    tokenizer, model = load_model(device, model_load_path)
    dataset_dict = load_dataset(quechua_spanish_dataset_id)

    old_vocab_size = _get_vocab_size(base_model_load_path)

    evaluator = TranslationEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset_dict=dataset_dict,
        old_vocab_size=old_vocab_size,
        device=device,
    )

    evaluator.eval_model(
        batch_size=batch_size,
        split=split,
        use_decoded_fst_output=use_decoded_fst_output,
        log_freq=50
    )