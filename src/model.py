# src/model.py
from transformers import AutoModelForSequenceClassification
from config import MODEL_NAME, NUM_LABELS

def get_model():
    """
    Retorna o modelo de classificação ajustado para o número de rótulos especificado.
    """
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    return model
