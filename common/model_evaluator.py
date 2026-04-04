from typing import cast, TypedDict

import os
from datetime import datetime

import torch
from datasets import DatasetDict
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from torch.utils.data import DataLoader

from common.utils import get_device, get_lang_abbrev, qs_tokenized_dataloader, TokenizedBatch, SPANISH_LANG_ID, QUECHUA_LANG_ID
from common.model_trainer import ModelTrainer

class TranslationTrainingConfig(TypedDict):
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps_frac: float
    grad_clip_max_norm: float
    eval_freq: int    
    save_path: str | None

class TranslationMetrics(TypedDict):
    bleu: float
    chrf: float
    chrf_pp: float

class TranslationTrainingResult(TypedDict):
    train_losses: list[float]
    val_losses: list[float]
    val_metrics: list[TranslationMetrics]

class TranslationEvaluator():
    '''Trains and evaluates the translation model, recording BLEU/chrF/chrF++ metrics.'''

    _CHECKPOINT_STR_START = 'translation_checkpoint'
    _MAX_LENGTH = 512
    _MAX_LENGTH_RESPONSE = 600
    _NUM_BEAMS = 1
    _N_DATALOADER_WORKERS = 1
    _N_TOKENIZE_WORKERS = 4
    _TRANSLATION_BATCHES_PER_PRINT = 50

    def __init__(
            self,
            model: M2M100ForConditionalGeneration,
            tokenizer: NllbTokenizer,
            dataset_dict: DatasetDict,
            old_vocab_size: int,
            device: str | None = None
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_dict = dataset_dict
        self.old_vocab_size = old_vocab_size
        self.device = get_device() if device is None else device

    def train_model(
            self,
            config: TranslationTrainingConfig
    ) -> TranslationTrainingResult:
        '''
        Fine-tunes the model using the given config.
        Computes translation metrics every config['eval_freq'] epochs on the validation set.
        Saves checkpoints to config['save_path'] if provided.
        Returns training losses, validation losses, and validation metrics per eval checkpoint.
        '''
        assert config['epochs'] > 0
        assert config['batch_size'] > 0
        assert config['lr'] > 0
        assert config['weight_decay'] >= 0
        assert 0 <= config['warmup_steps_frac'] <= 1
        assert config['grad_clip_max_norm'] > 0
        assert config['eval_freq'] > 0

        train_loader = self._build_dataloader('train', config['batch_size'], shuffle=True)
        val_loader = self._build_dataloader('validation', config['batch_size'], shuffle=False)

        trainer = ModelTrainer(
            modified_model=self.model,
            modified_tokenizer=self.tokenizer,
            n_training_epochs=config['epochs'],
            n_batches_train_dataset=len(train_loader),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            warmup_steps_frac=config['warmup_steps_frac'],
            grad_clip_max_norm=config['grad_clip_max_norm'],
            device=self.device
        )
        trainer.freeze_old_embeddings(self.old_vocab_size)

        return self._run_training(trainer, train_loader, val_loader, config)

    def eval_model(
            self,
            batch_size: int,
            split: str = 'test'
    ) -> TranslationMetrics:
        '''Evaluates the model on the given dataset split and prints the translation metrics.'''
        loader = self._build_dataloader(split, batch_size=batch_size, shuffle=False)
        metrics = self._compute_translation_metrics(loader, split)
        self._print_translation_metrics(metrics)
        return metrics

    def _run_training(
            self,
            trainer: ModelTrainer,
            train_loader: DataLoader[TokenizedBatch],
            val_loader: DataLoader[TokenizedBatch],
            config: TranslationTrainingConfig
    ) -> TranslationTrainingResult:
        train_losses: list[float] = []
        val_losses: list[float] = []
        val_metrics: list[TranslationMetrics] = []

        start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        for epoch in range(1, config['epochs'] + 1):
            print(f'--------------- Epoch {epoch}/{config['epochs']} ---------------')

            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.eval_epoch(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {epoch} - train loss: {train_loss:.5f}, val loss: {val_loss:.5f}')

            if epoch % config['eval_freq'] == 0:
                print(f'Computing translation metrics at epoch {epoch}...')
                metrics = self._compute_translation_metrics(val_loader, 'validation')
                val_metrics.append(metrics)
                self._print_translation_metrics(metrics)

            if config['save_path'] is not None:
                self._save_checkpoint(config['save_path'], epoch, start_time)

        return self._format_result(train_losses, val_losses, val_metrics)

    def _compute_translation_metrics(
            self,
            data_loader: DataLoader[TokenizedBatch],
            split: str
    ) -> TranslationMetrics:
        dataset = self.dataset_dict[split]
        reference_translations: list[str] = dataset[get_lang_abbrev(QUECHUA_LANG_ID)]
        predicted_translations: list[str] = []

        bos_target_lang = cast(int, self.tokenizer.convert_tokens_to_ids(QUECHUA_LANG_ID))

        self.model.eval()
        n_batches = len(data_loader)

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch: TokenizedBatch
                if (i + 1) % self._TRANSLATION_BATCHES_PER_PRINT == 0:
                    print(f'translating batch {i + 1}/{n_batches}')

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                output = cast(torch.LongTensor, self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    forced_bos_token_id=bos_target_lang,
                    max_length=TranslationEvaluator._MAX_LENGTH_RESPONSE,
                    num_beams=TranslationEvaluator._NUM_BEAMS
                ))

                predicted_translations.extend(
                    self.tokenizer.batch_decode(
                        output,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                )

        bleu = corpus_bleu(predicted_translations, [reference_translations])
        chrf = corpus_chrf(predicted_translations, [reference_translations])
        chrf_pp = corpus_chrf(predicted_translations, [reference_translations], word_order=2)

        return TranslationMetrics({
            'bleu': bleu.score,
            'chrf': chrf.score,
            'chrf_pp': chrf_pp.score
        })

    def _save_checkpoint(self, save_path: str, epoch: int, start_time: str) -> None:
        checkpoint_name = f'{TranslationEvaluator._CHECKPOINT_STR_START}_epoch{epoch}_{start_time}'
        full_path = os.path.join(save_path, checkpoint_name)
        os.makedirs(full_path, exist_ok=True)
        self.model.save_pretrained(full_path)
        self.tokenizer.save_pretrained(full_path)

    def _build_dataloader(
            self,
            split: str,
            batch_size: int,
            shuffle: bool
    ) -> DataLoader[TokenizedBatch]:
        return qs_tokenized_dataloader(
            self.dataset_dict[split],
            self.tokenizer,
            SPANISH_LANG_ID,
            batch_size,
            max_length=TranslationEvaluator._MAX_LENGTH,
            shuffle=shuffle,
            n_dataloader_workers=TranslationEvaluator._N_DATALOADER_WORKERS,
            n_tokenize_workers=TranslationEvaluator._N_TOKENIZE_WORKERS
        )

    def _format_result(
            self,
            train_losses: list[float],
            val_losses: list[float],
            val_metrics: list[TranslationMetrics]
    ) -> TranslationTrainingResult:
        return TranslationTrainingResult({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics
        })
    
    def _print_translation_metrics(self, metrics: TranslationMetrics) -> None:
        print(
            f'\t- BLEU:   {metrics['bleu']:.3f}\n'
            f'\t- chrF:   {metrics['chrf']:.3f}\n'
            f'\t- chrF++: {metrics['chrf_pp']:.3f}'
        )