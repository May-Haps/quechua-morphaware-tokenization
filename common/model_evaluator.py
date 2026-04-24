from typing import cast, TypedDict

from datetime import datetime
import json
import re
import os

import torch
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from torch.utils.data import DataLoader

from common.model_trainer import ModelTrainer
from common.process_word_windows import clean_decoded_text, encode_text
from common.utils import  decode_fst_output, get_lang_abbrev, load_dataset, load_model, \
    qs_tokenized_dataloader, TokenizedBatch, SPANISH_LANG_ID, QUECHUA_LANG_ID

def _get_vocab_size(model_path: str) -> int:
    return len(NllbTokenizer.from_pretrained(model_path))

class TranslationTrainingConfig(TypedDict):
    epochs: int
    batch_size: int
    batches_per_update: int
    lr: float
    weight_decay: float
    warmup_steps_frac: float
    grad_clip_max_norm: float
    eval_freq: int    
    save_folder_name: str | None

class TranslationMetrics(TypedDict):
    bleu_base: float
    chrf_base: float
    chrf_pp_base: float
    bleu_fst: float
    chrf_fst: float
    chrf_pp_fst: float

class TranslationTrainingResult(TypedDict):
    train_losses: list[float]
    val_losses: list[float]
    val_metrics: list[TranslationMetrics]

class TranslationEvaluator():
    '''Evaluates the finetuning of FST models across different hyperparameters and records the results.'''
    _CHECKPOINT_STR_START = 'translation_checkpoint'
    _TRAINING_RESULTS_FILENAME = 'training_result.json'
    _MAX_LENGTH = 128
    _NUM_BEAMS = 1
    _N_DATALOADER_WORKERS = 1
    _N_TOKENIZE_WORKERS = 4
    _TRANSLATION_BATCHES_PER_PRINT = 100

    def __init__(
            self,
            model_load_path: str,
            base_model_load_path: str,
            quechua_spanish_dataset_id: str,
            config: TranslationTrainingConfig,
            device: str
    ) -> None:
        assert config['epochs'] > 0
        assert config['batch_size'] > 0
        assert config['batches_per_update'] > 0
        assert config['lr'] > 0
        assert config['weight_decay'] >= 0
        assert 0 <= config['warmup_steps_frac'] <= 1
        assert config['grad_clip_max_norm'] > 0
        assert config['eval_freq'] > 0

        # Not saving model to self.model so trainer can free model resources if replaced
        self.tokenizer, model = load_model(device, model_load_path)
        self.dataset_dict = load_dataset(quechua_spanish_dataset_id)
        old_vocab_size = _get_vocab_size(base_model_load_path)

        self.config = config
        self.device = device

        self.train_loader = self._build_dataloader(
            'train', 
            config['batch_size'],
            shuffle=True
        )
        self.val_loader = self._build_dataloader(
            'validation',
            config['batch_size'],
            shuffle=False
        )

        self.trainer = ModelTrainer(
            modified_model=model,
            n_training_epochs=config['epochs'],
            n_batches_train_dataset=len(self.train_loader),
            batches_per_update=config['batches_per_update'],
            old_vocab_size=old_vocab_size,
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            warmup_steps_frac=config['warmup_steps_frac'],
            grad_clip_max_norm=config['grad_clip_max_norm'],
            device=self.device
        )

        self.start_epoch = 1
        self.train_losses, self.val_losses, self.val_metrics = [], [], []

        if config['save_folder_name'] is not None:
            latest_checkpoint = self._find_latest_checkpoint(config['save_folder_name'])
            if latest_checkpoint is not None:
                end_epoch, self.train_losses, self.val_losses, self.val_metrics = \
                    self.trainer.load_checkpoint(latest_checkpoint, old_vocab_size)
                self.start_epoch = end_epoch + 1
                print(f'Found checkpoint: starting off at epoch {self.start_epoch}')
                assert config['epochs'] > self.start_epoch > 1
        
        self.cached_reference_translations: dict[str, list[str]] = {}

    def train_model(self) -> TranslationTrainingResult:
        '''Trains the model on the training set and evaluates the model on the validation set.'''
        results = self._run_training()
        if self.config['save_folder_name'] is not None:
            self._save_training_results(results)
        return results

    def eval_model(
            self,
            batch_size: int,
            split: str = 'test',
            use_decoded_fst_output: bool = False,
    ) -> TranslationMetrics:
        '''Evaluates the model on the given dataset split with BLEU, chrF, and chrF++ translation metrics.'''
        loader = self._build_dataloader(split, batch_size, shuffle=False)
        metrics = self._compute_translation_metrics(loader, split, use_decoded_fst_output)
        self._print_translation_metrics(metrics)
        return metrics

    def _run_training(self) -> TranslationTrainingResult:
        start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        for epoch in range(self.start_epoch, self.config['epochs'] + 1):
            print(f'--------------- Epoch {epoch}/{self.config["epochs"]} ---------------')
            print(f'Starting training epoch...')
            train_loss = self.trainer.train_epoch(
                self.train_loader,
                self.config['batches_per_update'],
                TranslationEvaluator._TRANSLATION_BATCHES_PER_PRINT
            )

            print(f'Starting validation epoch...')
            val_loss = self.trainer.eval_epoch(self.val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f'Epoch {epoch} - train loss: {train_loss:.5f}, val loss: {val_loss:.5f}')

            if epoch % self.config['eval_freq'] == 0:
                print(f'Computing translation metrics on the validation dataset...')
                metrics = self._compute_translation_metrics(self.val_loader, 'validation')
                self.val_metrics.append(metrics)
                self._print_translation_metrics(metrics)

            if self.config['save_folder_name'] is not None:
                self._save_checkpoint(epoch, start_time)

        return self._format_result()

    def _compute_translation_metrics(
            self,
            data_loader: DataLoader[TokenizedBatch],
            split: str,
            use_decoded_fst_ouput: bool = False
    ) -> TranslationMetrics:
        dataset = self.dataset_dict[split]
        predicted_translations: list[str] = []

        bos_target_lang = cast(int, self.tokenizer.convert_tokens_to_ids(QUECHUA_LANG_ID))

        self.trainer.model.eval()
        n_batches = len(data_loader)

        def identity(text: str) -> str:
            return text
        
        additional_decode = decode_fst_output if use_decoded_fst_ouput else identity

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch: TokenizedBatch
                if (i + 1) % self._TRANSLATION_BATCHES_PER_PRINT == 0:
                    print(f'translating batch {i + 1}/{n_batches}')

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                output = cast(torch.LongTensor, self.trainer.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    forced_bos_token_id=bos_target_lang,
                    max_length=TranslationEvaluator._MAX_LENGTH,
                    num_beams=TranslationEvaluator._NUM_BEAMS
                ))

                decoded_text = self.tokenizer.batch_decode(
                    output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                for text in decoded_text:
                    predicted_translations.append(additional_decode(clean_decoded_text(text)))

        base_reference_translations: list[str] = dataset[get_lang_abbrev(QUECHUA_LANG_ID)]

        bleu_base = corpus_bleu(predicted_translations, [base_reference_translations])
        chrf_base = corpus_chrf(predicted_translations, [base_reference_translations])
        chrf_pp_base = corpus_chrf(predicted_translations, [base_reference_translations], word_order=2)

        if (split in self.cached_reference_translations):
            fst_reference_translations = list(map(additional_decode, self.cached_reference_translations[split]))
        else:
            fst_reference_translations: list[str] = [self._encode_reference(brt) for brt in base_reference_translations]
            self.cached_reference_translations[split] = fst_reference_translations
            fst_reference_translations = list(map(additional_decode, fst_reference_translations))

        bleu_fst = corpus_bleu(predicted_translations, [fst_reference_translations])
        chrf_fst = corpus_chrf(predicted_translations, [fst_reference_translations])
        chrf_pp_fst = corpus_chrf(predicted_translations, [fst_reference_translations], word_order=2)

        return TranslationMetrics({
            'bleu_base': bleu_base.score,
            'chrf_base': chrf_base.score,
            'chrf_pp_base': chrf_pp_base.score,
            'bleu_fst': bleu_fst.score,
            'chrf_fst': chrf_fst.score,
            'chrf_pp_fst': chrf_pp_fst.score,
        })

    def _encode_reference(self, text: str) -> str:
        '''
        Processes text through encode_text and the tokenizer so the reference matches the
        form it is trained to produce (including tags for morpheme markers).
        '''
        chunks = encode_text(text)
        token_ids = cast(list[int], self.tokenizer(
            chunks,
            is_split_into_words=False,
            add_special_tokens=False,
            truncation=True,
            max_length=TranslationEvaluator._MAX_LENGTH,
        )['input_ids'])

        return clean_decoded_text(self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ))

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
            n_tokenize_workers=TranslationEvaluator._N_TOKENIZE_WORKERS,
            use_fst=True
        )

    def _format_result(self) -> TranslationTrainingResult:
        return TranslationTrainingResult({
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        })

    def _save_checkpoint(self, epoch: int, start_time: str) -> None:
        assert self.config['save_folder_name'] is not None
        checkpoint_name = f'{TranslationEvaluator._CHECKPOINT_STR_START}_epoch{epoch}_{start_time}'
        full_path = os.path.join(self.config['save_folder_name'], checkpoint_name)
        os.makedirs(full_path, exist_ok=True)

        self.trainer.model.save_pretrained(full_path)
        self.tokenizer.save_pretrained(full_path)
        
        torch.save({
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "scheduler_state_dict": self.trainer.scheduler.state_dict(),
            "epoch": epoch,
        }, os.path.join(full_path, "training_state.pt"))

        training_result = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.val_metrics,
        }

        with open(os.path.join(full_path, "training_result.json"), "w") as f:
            json.dump(training_result, f, indent=2)

    def _save_training_results(self, results: TranslationTrainingResult) -> None:
        assert self.config['save_folder_name'] is not None
        os.makedirs(self.config['save_folder_name'], exist_ok=True)
        full_path = os.path.join(self.config['save_folder_name'], TranslationEvaluator._TRAINING_RESULTS_FILENAME)
        with open(full_path, 'w') as f:
            json.dump(results, f, indent=2)

    def _print_translation_metrics(self, metrics: TranslationMetrics) -> None:
        print(
            f'\t- BLEU (base):   {metrics["bleu_base"]:.3f}\n'
            f'\t- chrF (base):   {metrics["chrf_base"]:.3f}\n'
            f'\t- chrF++ (base): {metrics["chrf_pp_base"]:.3f}\n'
            f'\t- BLEU (fst):   {metrics["bleu_fst"]:.3f}\n'
            f'\t- chrF (fst):   {metrics["chrf_fst"]:.3f}\n'
            f'\t- chrF++ (fst): {metrics["chrf_pp_fst"]:.3f}'
        )

    def _find_latest_checkpoint(self, save_folder: str) -> str | None:
        if not os.path.exists(save_folder):
            return None

        pattern = re.compile(r"translation_checkpoint_epoch(\d+)_.*")

        latest_epoch = -1
        latest_path = None

        for name in os.listdir(save_folder):
            match = pattern.match(name)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_path = os.path.join(save_folder, name)

        return latest_path