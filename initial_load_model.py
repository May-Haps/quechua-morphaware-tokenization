from transformers import M2M100ForConditionalGeneration, NllbTokenizer

if __name__ == '__main__':
    model_id = 'facebook/nllb-200-distilled-600M'
    save_path = './nllb-model'

    tokenizer: NllbTokenizer = NllbTokenizer.from_pretrained(model_id)
    model = M2M100ForConditionalGeneration.from_pretrained(model_id)

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
