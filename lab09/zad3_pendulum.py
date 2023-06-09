import gym
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Zmienne lingwistyczne
angle = ctrl.Antecedent(np.arange(-np.pi, np.pi, 0.01), 'angle')
angular_velocity = ctrl.Antecedent(np.arange(-8, 8, 0.1), 'angular_velocity')
action = ctrl.Consequent(np.arange(-2, 2, 0.1), 'action')

# Definicja funkcji przynależności
angle['negative'] = fuzz.trimf(angle.universe, [-np.pi, -np.pi, 0])
angle['zero'] = fuzz.trimf(angle.universe, [-0.1, 0, 0.1])
angle['positive'] = fuzz.trimf(angle.universe, [0, np.pi, np.pi])

angular_velocity['negative'] = fuzz.trimf(angular_velocity.universe, [-8, -8, 0])
angular_velocity['zero'] = fuzz.trimf(angular_velocity.universe, [-0.1, 0, 0.1])
angular_velocity['positive'] = fuzz.trimf(angular_velocity.universe, [0, 8, 8])

action['negative'] = fuzz.trimf(action.universe, [-2, -2, 0])
action['zero'] = fuzz.trimf(action.universe, [-0.1, 0, 0.1])
action['positive'] = fuzz.trimf(action.universe, [0, 2, 2])

# Wykresów funkcji przynależności
angle.view()
angular_velocity.view()
action.view()

# Reguły wnioskowania rozmytego - and
rules = [
    ctrl.Rule(angle['negative'] & angular_velocity['negative'], action['positive']),
    ctrl.Rule(angle['negative'] & angular_velocity['zero'], action['positive']),
    ctrl.Rule(angle['negative'] & angular_velocity['positive'], action['zero']),
    ctrl.Rule(angle['zero'] & angular_velocity['negative'], action['positive']),
    ctrl.Rule(angle['zero'] & angular_velocity['zero'], action['zero']),
    ctrl.Rule(angle['zero'] & angular_velocity['positive'], action['zero']),
    ctrl.Rule(angle['positive'] & angular_velocity['negative'], action['zero']),
    ctrl.Rule(angle['positive'] & angular_velocity['zero'], action['negative']),
    ctrl.Rule(angle['positive'] & angular_velocity['positive'], action['negative'])
]

system = ctrl.ControlSystem(rules)
simulator = ctrl.ControlSystemSimulation(system)

env = gym.make('Pendulum-v1', render_mode="human")

for episode in range(10):
    observation = env.reset()
    done = False

    while not done:
        env.render()

        if isinstance(observation, tuple):
            observation = observation[0]

        simulator.input['angle'] = observation[0]
        simulator.input['angular_velocity'] = observation[1]

        simulator.compute()

        action_value = simulator.output['action']

        observation, reward, terminated, truncated, info = env.step([action_value])

        if terminated or truncated:
            done = True

env.close()

action.view(simulator)
plt.show()
