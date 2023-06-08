import gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset(seed=2)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

# Stan gry jest ciągły, ale zestaw akcji jest dyskretny
