from transformers import NllbTokenizer

from common.utils import get_device, load_dataset, load_model
from common.model_evaluator import TranslationEvaluator, TranslationTrainingConfig

'''
This script runs the training process with our fst morpheme aware model. Run extend_model_vocabulary.py
before this. On my machine, running the training process with these parameters uses just under 8Gbs of VRAM.
'''

quechua_spanish_dataset_id = 'somosnlp-hackathon-2022/spanish-to-quechua'
base_model_load_path = 'nllb-model'
model_load_path = './nllb-model-fst'
model_save_path = './nllb-model-fst-trained'

# NOTE No references to model in this script so that when loading model checkpoint the initial 'dummy'
# model can be freed.

if __name__ == '__main__':
    device = get_device()

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
    
    result = evaluator.train_model()