import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt

def train_q_learning_agent(env, q_table, learning_rate, discount_factor, num_episodes,
                           epsilon_start, epsilon_min, epsilon_decay_rate,
                           report_interval):
    """
    Entrena a un agente utilizando el algoritmo de Q-Learning en un entorno dado.

    Args:
        env: El entorno de Gymnasium en el que se entrena al agente.
        q_table (np.ndarray): La tabla Q inicializada.
        learning_rate (float): Tasa de aprendizaje (alpha).
        discount_factor (float): Factor de descuento (gamma).
        num_episodes (int): Número total de episodios de entrenamiento.
        epsilon_start (float): Valor inicial de epsilon para la exploración.
        epsilon_min (float): Valor mínimo de epsilon.
        epsilon_decay_rate (float): Tasa de decaimiento de epsilon por episodio.
        report_interval (int): Intervalo para reportar el progreso del entrenamiento.

    Returns:
        tuple: Una tupla que contiene la tabla Q entrenada (np.ndarray),
               la lista de recompensas por episodio (list), y la lista de
               número de pasos por episodio (list).
    """
    epsilon = epsilon_start
    start_time = time.time()
    rewards_history = []
    steps_history = []

    print(f"--- Iniciando Entrenamiento Q-Learning del Agente ---")
    print(f"Entorno: {env.spec.id}")
    print(f"Parámetros: α={learning_rate}, γ={discount_factor}, ε-decay={epsilon_decay_rate}")
    print(f"Episodios: {num_episodes}")
    print("-" * 40)

    for episode in range(num_episodes):
        # Reiniciar entorno
        state, info = env.reset()
        terminated = False
        truncated = False  # Indica si se superó max_steps_per_episode

        total_reward_episode = 0
        steps_episode = 0

        while not terminated and not truncated:
            # 1. Elegir Acción (Epsilon-Greedy)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explorar
            else:
                action = np.argmax(q_table[state, :])  # Explotar

            # 2. Ejecutar Acción
            new_state, reward, terminated, truncated, info = env.step(action)

            # 3. Actualizar Q-Table (Fórmula de Bellman)
            # El valor futuro (max Q del siguiente estado) es 0 si el episodio terminó
            q_update_target = reward + discount_factor * np.max(q_table[new_state, :]) * (1 - terminated)
            q_table[state, action] = q_table[state, action] + learning_rate * (q_update_target - q_table[state, action])

            # Actualizar estado y contadores
            state = new_state
            total_reward_episode += reward
            steps_episode += 1

            # Renderizar si está activado (opcional, puede ralentizar el entrenamiento)
            # if env.render_mode == "human":
            #     env.render()
            #     time.sleep(0.01)

        # Fin del episodio
        rewards_history.append(total_reward_episode)
        steps_history.append(steps_episode)

        # 4. Decaimiento de Epsilon (exponencial)
        epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp(-epsilon_decay_rate * episode)

        # Reportar progreso
        if (episode + 1) % report_interval == 0:
            # Calcular tasa de éxito reciente
            recent_episodes = rewards_history[-report_interval:]
            success_rate = np.mean([1 if r > 0 else 0 for r in recent_episodes]) * 100
            avg_steps = np.mean(steps_history[-report_interval:])
            print(f"Episodio: {episode + 1:>5}/{num_episodes} | "
                  f"Tasa Éxito (últimos {report_interval}): {success_rate:>6.2f}% | "
                  f"Pasos Prom (últimos {report_interval}): {avg_steps:>6.1f} | "
                  f"Epsilon: {epsilon:.3f}")

    end_time = time.time()
    print("-" * 40)
    print(f"Entrenamiento finalizado en {end_time - start_time:.2f} segundos.")
    print("-" * 40)

    return q_table, rewards_history, steps_history

import matplotlib.pyplot as plt
import numpy as np

def plot_success_rate(rewards_history, num_episodes, moving_avg_window=100):
    """
    Grafica la tasa de éxito del agente a lo largo del entrenamiento usando una media móvil.

    Args:
        rewards_history (list): Lista de recompensas obtenidas en cada episodio.
        num_episodes (int): Número total de episodios de entrenamiento.
        moving_avg_window (int): Tamaño de la ventana para calcular la media móvil.
    """
    if len(rewards_history) >= moving_avg_window:
        success_rate_all = [1 if r > 0 else 0 for r in rewards_history]
        moving_avg_success = np.convolve(success_rate_all, np.ones(moving_avg_window)/moving_avg_window, mode='valid')

        plt.figure(figsize=(12, 5))
        plt.plot(np.arange(moving_avg_window - 1, num_episodes), moving_avg_success * 100)
        plt.title(f"Tasa de Éxito Environment (Media Móvil {moving_avg_window} episodios)")
        plt.xlabel("Episodio")
        plt.ylabel("Tasa de Éxito (%)")
        plt.grid(True)
        plt.ylim(-5, 105)
        plt.show()
    else:
        print("No hay suficientes episodios para calcular la media móvil.")
        plt.figure(figsize=(12, 5))
        plt.plot(rewards_history)
        plt.title("Recompensa por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa Total")
        plt.grid(True)
        plt.show()

import numpy as np

def get_preferred_action(q_table, taxi_row, taxi_col, passenger_location, destination):
    """
    Obtiene la acción preferida de la tabla Q para un estado específico del taxi.

    Args:
        q_table (np.ndarray): La tabla Q entrenada.
        taxi_row (int): La fila actual del taxi (0-4).
        taxi_col (int): La columna actual del taxi (0-4).
        passenger_location (int): La ubicación actual del pasajero (0-4, donde 4 significa en el taxi).
        destination (int): La ubicación del destino (0-3).

    Returns:
        str: La acción preferida para el estado dado (ej: "↓", "↑", "→", "←", "P↑", "D↓").
    """
    actions = ["↓", "↑", "→", "←", "P↑", "D↓"]
    state = taxi_row * 25 + taxi_col * 5 + passenger_location * 5 + destination
    best_action_index = np.argmax(q_table[state, :])
    preferred_action = actions[best_action_index]
    return preferred_action




import time

def demonstrate_policy(env_name, trained_q_table, num_episodes=3, render=True, delay=0.5):
    """
    Demuestra la política aprendida por el agente utilizando la tabla Q entrenada.

    Args:
        env_name (str): El nombre del entorno de Gymnasium.
        trained_q_table (np.ndarray): La tabla Q entrenada.
        num_episodes (int): El número de episodios de demostración a ejecutar.
        render (bool): Si se debe renderizar el entorno durante la demostración.
        delay (float): El tiempo de espera (en segundos) entre cada paso de la demostración.
    """
    env_policy = gym.make(env_name, render_mode="human" if render else None)
    for episode in range(num_episodes):
        state, _ = env_policy.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        print(f"\nEpisodio de Demostración {episode + 1}:")
        while not terminated and not truncated:
            action = np.argmax(trained_q_table[state, :])
            new_state, reward, terminated, truncated, _ = env_policy.step(action)
            if render:
                env_policy.render()
                time.sleep(delay)
            total_reward += reward
            state = new_state
            steps += 1
            if terminated or truncated:
                print(f"Episodio terminado después de {steps} pasos con recompensa: {total_reward}")
                break
    env_policy.close()
