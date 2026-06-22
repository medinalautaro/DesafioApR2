import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)


class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        return True


def moving_average(values, window=20):
    values = np.array(values)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_metrics(rewards, episode_lengths, eval_path, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    rewards = np.array(rewards)
    episode_lengths = np.array(episode_lengths)

    # 1. Convergencia: reward por episodio + media móvil
    rewards_ma = moving_average(rewards, window=20)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward por episodio")

    if len(rewards) >= 20:
        plt.plot(
            range(19, 19 + len(rewards_ma)),
            rewards_ma,
            label="Media móvil 20 episodios",
        )

    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.title("Convergencia SAC en MountainCarContinuous-v0")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_convergencia_reward.png")
    plt.close()

    # 2. Longitud de episodios
    plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.xlabel("Episodio")
    plt.ylabel("Pasos")
    plt.title("Duración de episodios durante el entrenamiento")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_longitud_episodios.png")
    plt.close()

    # 3. Histograma de rewards
    plt.figure(figsize=(10, 5))
    plt.hist(rewards, bins=30)
    plt.xlabel("Reward")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de rewards por episodio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_histograma_rewards.png")
    plt.close()

    # 4. Reward acumulado
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(rewards))
    plt.xlabel("Episodio")
    plt.ylabel("Reward acumulado")
    plt.title("Reward acumulado durante el entrenamiento")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_reward_acumulado.png")  #Este grafico termino no siendo util para la explicacion del informe.
    plt.close()

    # 5. Evaluación periódica
    eval_npz = os.path.join(eval_path, "evaluations.npz")

    if os.path.exists(eval_npz):
        data = np.load(eval_npz)

        timesteps = data["timesteps"]
        results = data["results"]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, mean_rewards, label="Reward promedio evaluación")
        plt.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
        )
        plt.xlabel("Timesteps")
        plt.ylabel("Reward promedio")
        plt.title("Evaluación periódica del agente")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_reward_evaluacion.png")
        plt.close()


def make_train_env():
    env = gym.make("MountainCarContinuous-v0")
    env = Monitor(env)
    return env


def main():
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("models/best_model", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA disponible: True")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("CUDA disponible: False")
        print("Usando CPU")

    train_env = make_train_env()
    eval_env = Monitor(gym.make("MountainCarContinuous-v0"))

    reward_logger = RewardLoggerCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="models/checkpoints",
        name_prefix="sac_mountaincar",
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=20,
        min_evals=10,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best_model",
        log_path="logs/eval",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        callback_after_eval=stop_callback,
        verbose=1,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=dict(
            net_arch=[256, 256]
        ),
        device=device,
        verbose=1,
        tensorboard_log="logs/tensorboard",
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=[
            reward_logger,
            checkpoint_callback,
            eval_callback,
        ],
        tb_log_name="SAC_MountainCarContinuous",
    )

    model.save("models/sac_mountaincar_final")

    rewards = reward_logger.episode_rewards
    episode_lengths = reward_logger.episode_lengths

    np.save("logs/episode_rewards.npy", np.array(rewards))
    np.save("logs/episode_lengths.npy", np.array(episode_lengths))

    plot_training_metrics(
        rewards=rewards,
        episode_lengths=episode_lengths,
        eval_path="logs/eval",
        output_dir="plots",
    )

    print("\nEntrenamiento finalizado.")
    print(f"Episodios registrados: {len(rewards)}")

    if len(rewards) >= 20:
        print(f"Reward promedio últimos 20 episodios: {np.mean(rewards[-20:]):.2f}")
    elif len(rewards) > 0:
        print(f"Reward promedio: {np.mean(rewards):.2f}")
    else:
        print("No se registraron episodios.")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()