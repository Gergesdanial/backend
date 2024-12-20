from transformers import pipeline

# Initialize the POS pipeline
def load_pos_model(model_name='CAMeL-Lab/bert-base-arabic-camelbert-mix-pos-msa'):
    models = {
        'CAMeL-Lab/bert-base-arabic-camelbert-mix-pos-msa': 'CAMeL-Lab/bert-base-arabic-camelbert-mix-pos-msa',
        'CAMeL-Lab/bert-base-arabic-camelbert-da-pos-glf': 'CAMeL-Lab/bert-base-arabic-camelbert-da-pos-glf'
    }
    model_path = models.get(model_name)
    if not model_path:
        raise ValueError("Model not found. Please check the model name.")
    
    return pipeline('token-classification', model=model_path)

# Function to tag the part-of-speech for a given sentence
def tag_pos(text, model):
    try:
        pos_results = model(text)
        tokens = [result['word'] for result in pos_results]
        pos_tags = [result['entity'] for result in pos_results]
        return tokens, pos_tags
    except Exception as e:
        print(f"Error during POS tagging: {str(e)}")
        return [], []
