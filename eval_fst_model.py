from common.utils import get_device
from common.model_evaluator import TranslationEvaluator, TranslationTrainingConfig

'''
Evaluates the FST-trained model on the test split, reporting BLEU, chrF, and chrF++
against both the base reference translations and the FST-encoded reference translations.
'''

quechua_spanish_dataset_id = 'somosnlp-hackathon-2022/spanish-to-quechua'
base_model_load_path = 'nllb-model'
model_load_path = './nllb-model-fst-trained/translation_checkpoint_epoch10_20260422_201116'
model_save_path = None

batch_size = 8
split = 'test'


if __name__ == '__main__':
    device = 'cpu'
    # device = get_device()

    # The constructor requires a full TranslationTrainingConfig, but for pure
    # evaluation only batch_size is meaningfully used (by the train/val loaders
    # that __init__ builds unconditionally). The training hyperparameters below
    # just need to satisfy the assertions. save_folder_name=None skips the
    # checkpoint-resume branch in __init__.
    config: TranslationTrainingConfig = {
        'epochs': 20,
        'batch_size': 8,
        'batches_per_update': 2,
        'lr': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps_frac': 0.1,
        'grad_clip_max_norm': 1.0,
        'eval_freq': 1,
        'save_folder_name': model_save_path,
    }

    evaluator = TranslationEvaluator(
        model_load_path=model_load_path,
        base_model_load_path=base_model_load_path,
        quechua_spanish_dataset_id=quechua_spanish_dataset_id,
        config=config,
        device=device,
    )

    evaluator.eval_model(batch_size=batch_size, split=split)