# backend/tc_model.py

from transformers import pipeline

# Initialize the Zero-Shot Classification pipeline
def load_tc_model():
    return pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

# Function to classify each sentence into user-defined classes
def classify_text(sentences, user_classes, model):
    try:
        classified_sentences = []
        for sentence in sentences:
            # Perform zero-shot classification
            result = model(sentence, user_classes, multi_label=False)
            assigned_class = result['labels'][0]  # Top predicted class
            confidence = result['scores'][0]
            classified_sentences.append({
                "sentence": sentence.strip(),
                "class": assigned_class,
                "confidence": confidence
            })
        return classified_sentences
    except Exception as e:
        print(f"Error during text classification: {str(e)}")
        return []
