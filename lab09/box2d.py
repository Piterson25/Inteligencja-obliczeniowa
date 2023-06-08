import gym

env = gym.make("BipedalWalkerHardcore-v3", render_mode="human")
env.reset(seed=2)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

# Stan gry i zestaw akcji są ciągłe
