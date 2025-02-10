# src/train.py
import os
from transformers import TrainingArguments, Trainer
from data_loader import load_data, tokenize_data
from model import get_model
from config import TRAINING_ARGS, OUTPUT_DIR, MODEL_SAVE_PATH
from plotting_callback import PlottingCallback  # Importa o callback customizado

def main():
    # Carregar e tokenizar os dados
    dataset = load_data()
    tokenized_datasets, tokenizer = tokenize_data(dataset)

    # Obter o modelo
    model = get_model()

    # Configurar os TrainingArguments
    training_args = TrainingArguments(**TRAINING_ARGS)

    # Inicializar o Trainer com o callback de plotagem
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        callbacks=[PlottingCallback()]  # Adiciona o callback
    )

    # Iniciar o treinamento
    trainer.train()

    # Salvar o modelo e o tokenizer
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Modelo e tokenizer salvos em: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
