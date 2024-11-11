# train_model.py
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, DataProcessor

class TranslatorModel:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-de"):
        self.model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.processor = DataProcessor()

    def train(self, source_texts, target_texts, epochs=3, batch_size=8):
        source_encodings, target_encodings = self.processor.encode(source_texts, target_texts)

        # Define the training loop
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        self.model.compile(optimizer=optimizer, loss=self.model.compute_loss)
        
        dataset = tf.data.Dataset.from_tensor_slices((source_encodings, target_encodings)).batch(batch_size)
        self.model.fit(dataset, epochs=epochs)

# Usage example
# translator = TranslatorModel()
# translator.train(["Hello"], ["Hallo"])
