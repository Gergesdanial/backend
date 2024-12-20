from camel_tools.ner import NERecognizer
from camel_tools.tokenizers.word import simple_word_tokenize
from transformers import pipeline

def load_ner_model(model_name='camelbert-msa'):
    models = {
        'camelbert-msa': 'CAMeL-Lab/bert-base-arabic-camelbert-msa-ner',
        'camelbert-da': 'CAMeL-Lab/bert-base-arabic-camelbert-da-ner',
        'bert-english': 'dbmdz/bert-large-cased-finetuned-conll03-english',  # English NER model
        # You can add more models here
    }
    model_path = models.get(model_name)
    if not model_path:
        raise ValueError("Model not found. Please check the model name.")
    
    if model_name.startswith('camelbert'):
        return NERecognizer(model_path)
    elif model_name == 'bert-english':
        return pipeline('ner', model=model_path, aggregation_strategy="simple")
    else:
        raise ValueError("Model loading not supported for the given model name.")

def annotate_text(text, model):
    try:
        if isinstance(model, NERecognizer):
            tokens = simple_word_tokenize(text)
            annotations = model.predict_sentence(tokens)
            print(f"Tokens: {tokens}, Annotations: {annotations}")  # Log tokens and annotations
            return tokens, annotations
        else:  # Handle English NER using Hugging Face model
            annotations = model(text)
            tokens = [entity['word'] for entity in annotations]
            tags = [entity['entity_group'] for entity in annotations]
            print(f"Tokens: {tokens}, Annotations: {tags}")  # Log tokens and annotations
            return tokens, tags
    except Exception as e:
        print(f"Error during text annotation: {str(e)}")
        return [], []
