# src/plotting_callback.py
import matplotlib.pyplot as plt
from transformers import TrainerCallback

class PlottingCallback(TrainerCallback):
    def __init__(self):
        # Armazena os valores dos passos (ou épocas) e as métricas
        self.train_losses = []
        self.eval_losses = []
        self.steps = []  # Será preenchido com o número de passos

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Esse método é chamado sempre que há um log durante o treinamento
        if logs is not None:
            # Verifica se há loss no log
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.steps.append(state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Esse método é chamado quando ocorre uma avaliação
        if metrics is not None and "eval_loss" in metrics:
            self.eval_losses.append(metrics["eval_loss"])

    def on_train_end(self, args, state, control, **kwargs):
        # Ao final do treinamento, gera e salva os gráficos

        # Gráfico do treinamento (loss vs. passos)
        if self.train_losses and self.steps:
            plt.figure(figsize=(10, 5))
            plt.plot(self.steps, self.train_losses, label="Train Loss")
            plt.xlabel("Passos (Global Step)")
            plt.ylabel("Loss")
            plt.title("Loss Durante o Treinamento")
            plt.legend()
            plt.grid(True)
            plt.savefig("training_loss.png")
            plt.close()

        # Gráfico da avaliação (loss vs. época)
        if self.eval_losses:
            epochs = range(1, len(self.eval_losses) + 1)
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.eval_losses, marker='o', label="Eval Loss")
            plt.xlabel("Época")
            plt.ylabel("Eval Loss")
            plt.title("Loss na Avaliação por Época")
            plt.legend()
            plt.grid(True)
            plt.savefig("eval_loss.png")
            plt.close()
