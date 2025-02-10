<p align="left">
  <img src="static/nlp.png" width="150">
</p>

# Projeto de Categorização Automática de Projetos

Este repositório contém um projeto completo para **categorização automática de projetos** usando modelos de linguagem baseados em Transformers (BERT) para o português. O pipeline permite:
- Carregar e pré-processar dados (CSV) com descrições de projetos.
- Normalizar os textos (remoção de acentos, caracteres especiais e conversão para minúsculas).
- Converter rótulos de categorias (como "Inteligência Artificial", "Integração de Sistemas", etc.) para valores numéricos.
- Treinar e avaliar um modelo de classificação (utilizando fine-tuning de `neuralmind/bert-base-portuguese-cased`).
- Gerar gráficos de métricas (como loss) durante o treinamento.
- Realizar inferências e integrar o modelo em uma API Flask para categorização em tempo real.

## Sumário

- [Visão Geral](#visão-geral)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Dependências](#dependências)
- [Configuração e Estrutura](#configuração-e-estrutura)
- [Pré-processamento e Conversão de Rótulos](#pré-processamento-e-conversão-de-rótulos)
- [Treinamento](#treinamento)
- [Avaliação](#avaliação)
- [Inferência](#inferência)
- [Integração com Flask](#integração-com-flask)
- [Geração de Gráficos](#geração-de-gráficos)
- [Uso do Hugging Face Hub](#uso-do-hugging-face-hub)
- [Licença](#licença)
- [Contribuição](#contribuição)
- [Contato](#contato)

## Visão Geral

Este projeto foi desenvolvido para automatizar a categorização de projetos com base em suas descrições. Utiliza técnicas modernas de NLP com fine-tuning de modelos pré-treinados e integra os resultados em uma API Flask para facilitar o uso prático. Também inclui recursos para monitorar o treinamento por meio de gráficos que podem ser documentados no GitHub.

## Arquitetura do Projeto

A estrutura de diretórios é organizada da seguinte forma:

```
ml_categorizacao/
├── data/
│   ├── train.csv          # Arquivo CSV com dados de treinamento
│   └── validation.csv     # Arquivo CSV com dados de validação
├── results/
│   └── modelo_finetuned/  # Modelo treinado e arquivos associados
├── src/
│   ├── __init__.py
│   ├── config.py          # Configurações e hiperparâmetros
│   ├── data_loader.py     # Carregamento, normalização e tokenização dos dados
│   ├── model.py           # Definição e carregamento do modelo
│   ├── train.py           # Script de treinamento
│   ├── evaluate.py        # Script de avaliação
│   ├── test_inference.py  # Script para testar a inferência do modelo
│   └── plotting_callback.py  # Callback customizado para gerar gráficos durante o treinamento
├── app.py                 # API Flask para inferência em tempo real
└── requirements.txt       # Lista de dependências
```

## Dependências

O projeto utiliza as seguintes bibliotecas:
- **Python 3.8+**
- [Transformers](https://huggingface.co/transformers/)
- [Datasets](https://huggingface.co/docs/datasets/)
- [PyTorch](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Matplotlib](https://matplotlib.org/)

Para instalar as dependências, execute:

```bash
pip install -r requirements.txt
```

## Configuração e Estrutura

No arquivo `src/config.py` você encontrará as principais configurações, como:
- Nome do modelo pré-treinado (`neuralmind/bert-base-portuguese-cased`)
- Número de categorias (rótulos)
- Caminhos dos arquivos CSV (`data/train.csv` e `data/validation.csv`)
- Parâmetros de treinamento (batch size, número de épocas, taxa de aprendizado, etc.)

Exemplo de conteúdo:

```python
import os

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
NUM_LABELS = 12  # Total de categorias definidas
MAX_LENGTH = 128

TRAIN_FILE = os.path.join("data", "train.csv")
VALIDATION_FILE = os.path.join("data", "validation.csv")

OUTPUT_DIR = os.path.abspath("results")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "modelo_finetuned")

TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "evaluation_strategy": "epoch",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "logging_dir": "./logs",  # Para visualização com TensorBoard (opcional)
}
```

## Treinamento

Para iniciar o treinamento do modelo, execute:

```bash
python src/train.py
```

O script:
- Carrega e pré-processa os dados.
- Tokeniza os textos.
- Realiza o fine-tuning do modelo BERT.
- Salva o modelo treinado em `results/modelo_finetuned`.
- Utiliza um callback customizado para registrar métricas e gerar gráficos.

## Integração com Flask

O arquivo `app.py` contém uma API Flask que:
- Recebe uma requisição POST com um JSON contendo a chave `"text"`.
- Utiliza o modelo treinado para classificar o texto.
- Retorna a categoria prevista e a confiança.

Para iniciar a API, execute:

```bash
python app.py
```

Você poderá testar a API com ferramentas como o Postman ou via `curl`:

```bash
curl -X POST http://localhost:5000/categorize \
     -H "Content-Type: application/json" \
     -d '{"text": "Descrição do projeto..."}'
```

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Contribuição

Contribuições são bem-vindas! Para colaborar:
1. Faça um fork deste repositório.
2. Crie uma branch para sua feature:  
   `git checkout -b feature/MinhaFeature`
3. Faça commits com suas alterações.
4. Envie sua branch para o repositório remoto:  
   `git push origin feature/MinhaFeature`
5. Abra um Pull Request.

## Contato

- **Nome:** [Seu Nome]
- **Email:** [seu-email@example.com]
- **GitHub:** [https://github.com/seu-usuario](https://github.com/seu-usuario)
