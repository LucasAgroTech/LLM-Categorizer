# src/data_loader.py
import re
import unicodedata
from datasets import load_dataset
from transformers import AutoTokenizer
from config import TRAIN_FILE, VALIDATION_FILE, MODEL_NAME, MAX_LENGTH

def normalize_text(text):
    """
    Normaliza o texto removendo acentuação, caracteres especiais
    e convertendo para minúsculas.
    """
    # Remove acentuação (normaliza para o padrão NFD e remove os acentos)
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    
    # Remove caracteres especiais, mantendo letras, números e espaços
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Converte para minúsculas
    text = text.lower()
    
    return text

def clean_text(example):
    """
    Aplica a normalização no campo 'text' de cada exemplo.
    """
    example["text"] = normalize_text(example["text"])
    return example

def convert_label(example):
    # Mapeamento das categorias para valores numéricos
    label_mapping = {
        "Inteligência Artificial": 0,
        "Integração de Sistemas": 1,
        "Biotecnologia": 2,
        "Desenvolvimento de Hardware": 3,
        "Química": 4,
        "Sistemas de Comunicação": 5,
        "Materiais": 6,
        "Manufatura": 7,
        "Automação e Robótica": 8,
        "Prototipagem": 9,
        "Sistemas Submarinos": 10,
        "Tecnologia de Dutos": 11,
    }
    # Remove espaços extras e mapeia o rótulo; se não encontrado, define como -1 (pode ser ajustado conforme necessário)
    example["label"] = label_mapping.get(example["label"].strip(), -1)
    return example

def load_data():
    """
    Carrega os datasets de treinamento e validação a partir dos arquivos CSV.
    Aplica a limpeza no campo 'text' e converte os labels.
    """
    data_files = {"train": TRAIN_FILE, "validation": VALIDATION_FILE}
    # Note que o CSV utiliza ";" como delimitador
    dataset = load_dataset("csv", data_files=data_files, delimiter=";")
    
    # Aplica a limpeza no texto
    dataset = dataset.map(clean_text)
    
    # Converte os labels de string para valores numéricos
    dataset = dataset.map(convert_label)
    
    return dataset

def tokenize_data(dataset):
    """
    Aplica a tokenização nos textos do dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets, tokenizer
