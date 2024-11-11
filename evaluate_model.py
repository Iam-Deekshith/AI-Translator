# evaluate_model.py
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

class Evaluator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-de"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text):
        input_ids = self.tokenizer(text, return_tensors="tf").input_ids
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def calculate_bleu(self, reference, translation):
        return sentence_bleu([reference.split()], translation.split())

# Usage example
# evaluator = Evaluator()
# translation = evaluator.translate("Hello")
# score = evaluator.calculate_bleu("Hallo", translation)
