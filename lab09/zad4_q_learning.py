import gym
import numpy as np

env = gym.make('FrozenLake-v1')

Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.5  # Współczynnik uczenia
gamma = 0.9  # Współczynnik dyskontowania
epsilon = 0.1  # Współczynnik eksploracji

for episode in range(10000):
    observation = env.reset()
    done = False

    while not done:
        if isinstance(observation, tuple):
            observation = observation[0]

        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Eksploracja
        else:
            action = np.argmax(Q[observation])  # Eksploatacja

        # Wykonanie akcji w środowisku i otrzymanie nowego stanu, nagrody i informacji
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Aktualizacja wartości Q-table dla poprzedniego stanu i wykonanej akcji
        Q[observation, action] += alpha * (reward + gamma * np.max(Q[next_observation]) - Q[observation, action])

        observation = next_observation

        if terminated or truncated:
            done = True

# Znalezienie optymalnej strategii na podstawie Q-table
policy = np.argmax(Q, axis=1)

# Optymalne rozwiazanie
best_solution = policy.reshape((4, 4))[0]
print(best_solution)

env2 = gym.make('FrozenLake-v1', render_mode="human")
observation = env2.reset()
done = False

for action in best_solution:
    while not done:
        env2.render()

        if isinstance(observation, tuple):
            observation = observation[0]

        observation, reward, terminated, truncated, info = env2.step(action)

        if terminated or truncated:
            done = True

env2.close()
