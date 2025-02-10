# src/evaluate.py
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from data_loader import load_data, tokenize_data
from config import TRAINING_ARGS, MODEL_SAVE_PATH

def main():
    # Carregar os dados (aqui usamos a validação; substitua por um dataset de teste se disponível)
    dataset = load_data()
    tokenized_datasets, tokenizer = tokenize_data(dataset)

    # Carregar o modelo salvo
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

    # Configurar os TrainingArguments para avaliação (podem ser os mesmos do treinamento)
    training_args = TrainingArguments(**TRAINING_ARGS)

    # Inicializar o Trainer para avaliação
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    # Avaliar o modelo
    eval_result = trainer.evaluate()
    print("Resultados da avaliação:", eval_result)

if __name__ == "__main__":
    main()
