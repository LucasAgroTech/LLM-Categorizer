# app.py
from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# Defina o caminho para o modelo treinado
MODEL_SAVE_PATH = os.path.join("results", "modelo_finetuned")

# Cria o pipeline de classificação utilizando o modelo salvo
classifier = pipeline("text-classification", model=MODEL_SAVE_PATH, tokenizer=MODEL_SAVE_PATH)

# Dicionário para mapear índices para nomes de categorias
id2label = {
    0: "Inteligência Artificial",
    1: "Integração de Sistemas",
    2: "Biotecnologia",
    3: "Desenvolvimento de Hardware",
    4: "Química",
    5: "Sistemas de Comunicação",
    6: "Materiais",
    7: "Manufatura",
    8: "Automação e Robótica",
    9: "Prototipagem",
    10: "Sistemas Submarinos",
    11: "Tecnologia de Dutos",
}

@app.route('/categorize', methods=['POST'])
def categorize_project():
    """
    Endpoint que recebe um JSON com a chave 'text' e retorna a categoria prevista.
    Exemplo de requisição:
        {
            "text": "Descrição do projeto..."
        }
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Parâmetro 'text' é obrigatório."}), 400

    text = data["text"]

    # Faz a inferência usando o pipeline
    result = classifier(text)
    
    # A saída do pipeline vem no formato: [{'label': 'LABEL_0', 'score': 0.97}, ...]
    # Extraímos o índice a partir do valor da chave 'label'
    try:
        label_str = result[0]["label"]  # ex: "LABEL_0"
        label_index = int(label_str.split("_")[1])
    except (IndexError, ValueError):
        return jsonify({"error": "Erro ao interpretar a saída do modelo."}), 500

    # Mapeia o índice para o nome da categoria
    category = id2label.get(label_index, "Desconhecido")

    return jsonify({
        "predicted_category": category,
        "confidence": result[0]["score"]
    })

if __name__ == '__main__':
    # Defina debug=True somente em desenvolvimento
    app.run(host='0.0.0.0', port=500, debug=True)
