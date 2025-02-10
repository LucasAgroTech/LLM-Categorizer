# src/config.py
import os

# Nome do modelo pré-treinado
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

# Número de categorias (rótulos) do seu problema
NUM_LABELS = 2  # ajuste conforme sua necessidade

# Tamanho máximo dos tokens
MAX_LENGTH = 128

# Caminhos para os arquivos CSV com os dados
TRAIN_FILE = os.path.join("data", "train.csv")
VALIDATION_FILE = os.path.join("data", "validation.csv")

# Diretório para salvar os resultados e o modelo fine-tuned
OUTPUT_DIR = "results"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "modelo_finetuned")

# Hiperparâmetros de treinamento
TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "evaluation_strategy": "epoch",      # Avalia ao final de cada época
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "logging_steps": 50,
}
