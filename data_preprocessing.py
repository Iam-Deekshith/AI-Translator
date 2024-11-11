# data_preprocessing.py
import tensorflow as tf
from transformers import AutoTokenizer

class DataProcessor:
    def __init__(self, source_lang="en", target_lang="de", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        self.max_length = max_length

    def preprocess(self, text):
        return self.tokenizer(
            text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="tf"
        )

    def encode(self, source_texts, target_texts):
        source_encodings = self.preprocess(source_texts)
        target_encodings = self.preprocess(target_texts)
        return source_encodings["input_ids"], target_encodings["input_ids"]

# Usage example
# processor = DataProcessor()
# source, target = processor.encode(["Hello, how are you?"], ["Hallo, wie geht es dir?"])
