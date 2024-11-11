from flask import Flask, render_template, request
from transformers import MarianMTModel, MarianTokenizer

# Initialize the Flask app
app = Flask(__name__)

# Function to load the model and tokenizer based on selected language
def load_model(source_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

@app.route('/')
def index():
    # List of supported language pairs (you can expand this list)
    language_pairs = [
        ('en', 'de'),  # English to German
        ('en', 'fr'),  # English to French
        ('en', 'es'),  # English to Spanish
        ('de', 'en'),  # German to English
        ('fr', 'en'),  # French to English
        # Add more pairs as needed
    ]
    return render_template('index.html', language_pairs=language_pairs)

@app.route('/translate', methods=['POST'])
def translate():
    # Get the text input and selected languages from the form
    text = request.form['text_to_translate']
    source_lang = request.form['source_lang']
    target_lang = request.form['target_lang']
    
    # Load the model and tokenizer based on selected languages
    model, tokenizer = load_model(source_lang, target_lang)
    
    # Translate the text
    translated = translate_text(text, model, tokenizer)
    
    # Return the translated text and languages to the template, ensuring language options are passed
    language_pairs = [
        ('en', 'de'),
        ('en', 'fr'),
        ('en', 'es'),
        ('de', 'en'),
        ('fr', 'en'),
        # Add more pairs as needed
    ]
    return render_template('index.html', translated_text=translated, original_text=text, 
                           source_lang=source_lang, target_lang=target_lang, language_pairs=language_pairs)

def translate_text(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer.encode(text, return_tensors="pt")
    
    # Generate the translation using the model
    translated = model.generate(inputs, max_length=50)
    
    # Decode the translated tokens back to text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translated_text

if __name__ == '__main__':
    app.run(debug=True)
