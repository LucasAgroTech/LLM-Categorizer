# test_inference.py
from transformers import pipeline

# Caminho para o modelo salvo (certifique-se de que esse caminho esteja correto)
MODEL_SAVE_PATH = "results/modelo_finetuned"

# Cria o pipeline para classificação de texto
classifier = pipeline("text-classification", model=MODEL_SAVE_PATH, tokenizer=MODEL_SAVE_PATH)

# Texto de exemplo para teste
novo_texto = "Este projeto visa desenvolver soluções inovadoras em Inteligência Artificial para otimizar processos industriais."

# Faz a inferência e imprime o resultado
resultado = classifier(novo_texto)
print("Resultado da inferência:", resultado)
