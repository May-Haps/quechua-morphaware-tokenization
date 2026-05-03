# Morphologically Aware Tokenization Finetuning of NLLB for Spanish-Quechua Translation

Finetuning Meta's [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
translation model to improve Spanish ↔ Quechua translation by replacing NLLB's
default subword tokenization for Quechua with morpheme tokenization from a
finite state transducer (FST).

## Motivation

Quechua is an agglutinative language where words are built by concatenating many
morphemes, which makes standard subword tokenizers a poor fit. We hypothesize
that grounding tokenization in Quechua's morphological structure will make it
easier for the model to encode meaning into its vocabulary and elicit better
translation performance.

## Approach

1. **Morpheme segmentation.** Quechua words are run through a modified version
   of Annette Rios's `analyzeCuzco` FST, which splits them into their
   constituent morphemes. Words the FST doesn't recognize fall back to NLLB's
   original tokenizer. Across the train/val/test splits, the FST successfully
   encodes ~40% of word candidate chunks (566k of 1.07M); the rest fall
   through to subwords.
2. **Vocabulary extension.** All unique morphemes produced by the FST across
   the dataset are added as new tokens to NLLB's tokenizer. Each new token's
   embedding is initialized as the average of the subword embeddings NLLB
   would have originally produced for that morpheme.
3. **Targeted finetuning.** The encoder is frozen entirely, and the embeddings
   of all pre-existing (non-FST) tokens are frozen via a gradient hook. Only
   the new morpheme embeddings and the decoder are updated, concentrating
   learning on the new vocabulary without minimal disturbance to NLLB's existing knowledge.
4. **Evaluation.** BLEU, chrF, and chrF++ on the
   [somosnlp-hackathon-2022/spanish-to-quechua](https://huggingface.co/datasets/somosnlp-hackathon-2022/spanish-to-quechua)
   dataset.

### Two tokenization variants

We trained two variants of the FST-tokenized model that differ in how they
mark word boundaries:

- **ND (No Duplicates)** — *this branch (`no-dup-fst`).* Word boundaries are
  encoded with lone SentencePiece start-of-word tokens (`▁`) inserted between
  words. Each FST morpheme has a single embedding regardless of position in
  the word.
- **WIM (Word Initial Morphemes)** — *available on the [`main`](../../tree/main) branch.*
  Every morpheme that can appear word-initially (roots and parts of speech)
  gets a duplicate embedding for its word-initial form, mirroring how NLLB's
  original tokenizer represents word boundaries with the `▁` prefix on the
  first subword of each word.

ND is the stronger of the two variants on every metric (see results below);
check out the `no-dup-fst` branch to reproduce those numbers.

## Results

On the spanish-to-quechua test set (~13k samples), comparing decoded model
output (FST tags stripped) against the original reference translations:

| Model         |  BLEU |  chrF | chrF++ |
| ------------- | ----: | ----: | -----: |
| Baseline NLLB |   3.0 |  30.1 |   25.6 |
| WIM (ours)    |   6.8 |  33.1 |   28.9 |
| ND (ours)     |  10.7 |  43.3 |   38.3 |

The ND model substantially outperforms the baseline across all three metrics
and beats WIM by a meaningful margin. We attribute the WIM gap to its
duplicate root/POS embeddings: with only ~100k training examples, splitting
the signal across two embeddings per morpheme (initial vs medial) doesn't
converge as well as keeping a single shared embedding.

The relatively larger gains on chrF/chrF++ than on BLEU are expected. BLEU is
a word-level metric, and Quechua's agglutinative structure means many "words"
are long morpheme stacks, so a single wrong suffix tanks the n-gram match
while leaving most of the characters correct. chrF gives partial credit at
the character level, which is a more informative signal for this language.

For loss curves, raw vs decoded evaluation, and metrics computed against
FST-encoded references, see the report.

## Repository structure

- `initial_load_model.py` — downloads and caches the base NLLB model
- `extend_model_vocabulary.py` — extracts FST morphemes and extends the tokenizer + embedding matrix
- `train_fst_model.py` — finetunes the extended model
- `evaluate_base_model.py` — evaluates the base model on the test set
- `sample_translations.py` — prints sample translations
- `visualize_fst.py` — CLI helper that prints the FST morpheme decomposition for one or more Quechua words
- `common/` — shared utilities (tokenization, FST wrapper, training loop, evaluation)

## Running it

```bash
pip install -r requirements.txt
python initial_load_model.py        # download NLLB-200 distilled 600M
python extend_model_vocabulary.py   # add FST morphemes to vocabulary
python train_fst_model.py           # finetune
python evaluate_base_model.py       # evaluate on test set
```

To inspect the FST output for specific words:

```bash
python visualize_fst.py paykunaqa wasipi
# or pipe words in, one per line
echo "paykunaqa" | python visualize_fst.py
```

## Morphological analyzer

Morpheme segmentation is performed by a modified version of the `analyzeCuzco`
FST from Annette Rios's Quechua language toolkit, which targets the Southern
Quechua (Cuzco) variety.

If you use this project, please cite the underlying analyzer:

- Rios Gonzales, A. (2016). A basic language technology toolkit for Quechua.
  *Procesamiento del Lenguaje Natural*, 56, 91–94.
- Rios Gonzales, A., & Castro Mamani, R. A. (2014). Morphological Disambiguation
  and Text Normalization for Southern Quechua Varieties. In *Proceedings of the
  First Workshop on Applying NLP Tools to Similar Languages, Varieties and
  Dialects (VarDial)*, pages 39–47, Dublin, Ireland.

## Related work

This project is in the same research vein as prior work on morphologically
informed segmentation for low-resource NMT (e.g., Ortega et al.'s BPE-Guided
segmentation for Quechua, 2021; Asgari et al.'s MorphBPE, 2025). The
contribution here is using FST output directly as new vocabulary entries in
an NLLB-scale pretrained model, rather than as a segmentation target for
learned subword methods.
