import random

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

env = gym.make('CartPole-v1', render_mode="human")

alpha = 0.001  # Współczynnik uczenia
gamma = 0.99  # Współczynnik dyskontowania
epsilon = 1.0  # Początkowy współczynnik eksploracji
epsilon_min = 0.01  # Minimalny współczynnik eksploracji
epsilon_decay = 0.995  # Współczynnik zmniejszania eksploracji
batch_size = 64  # Rozmiar batcha dla aktualizacji sieci

# Pamiec replay
replay_memory = []
replay_memory_capacity = 10000

# Siec neuronowa
model = Sequential()
model.add(Dense(24, input_dim=env.observation_space.shape[0], activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer=Adam(learning_rate=alpha))


def choose_action(epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        q_values = model.predict(state)
        return np.argmax(q_values[0])


observation = env.reset()
done = False
total_reward = 0

while not done:
    env.render()

    action = choose_action(epsilon)

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        done = True
    next_state = np.reshape(observation, [1, env.observation_space.shape[0]])

    # Aktualizacja pamięci replay
    replay_memory.append((action, reward, next_state, done))
    if len(replay_memory) > replay_memory_capacity:
        replay_memory.pop(0)

    state = next_state
    total_reward += reward

    # Aktualizacja sieci neuronowej przy użyciu mini-batcha z pamięci replay
    if len(replay_memory) >= batch_size:
        batch = random.sample(replay_memory, batch_size)
        for state_batch, action_batch, reward_batch, next_state_batch, done_batch in batch:
            target = reward_batch
            if not done_batch:
                next_q_values = model.predict(next_state_batch)[0]
                target += gamma * np.amax(next_q_values)
            q_values = model.predict(state_batch)
            q_values[0][action_batch] = target
            model.fit(state_batch, q_values, epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Nagroda:", total_reward)

env.close()
