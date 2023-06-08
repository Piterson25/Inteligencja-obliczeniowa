import gym
import pygad
import numpy as np


def fitness_lunar(solution, render=False):
    env = gym.make('LunarLander-v2')
    env.reset(seed=42)

    total_reward = 0

    for action in solution:
        action = int(action)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward


env = gym.make('LunarLander-v2', render_mode="human")
env.reset(seed=42)

chromosome_length = 300
gene_space = [0, 1, 2, 3]

num_generations = 500
num_parents_mating = 4
sol_per_pop = 8
num_genes = chromosome_length
parent_selection_type = "sss"
keep_parents = num_parents_mating
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(
    gene_space=gene_space,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_lunar,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Najlepsze rozwiązanie:", solution)
print("Fitness najlepszego rozwiązania:", solution_fitness)

for action in solution:
    action = int(action)
    env.render()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# Chromosom to tablica [0, 1, 2, 3] odpowiadajacym mozliwym akcjom
# 0 - nic, 1 - lewy silnik, 2 - glowny silnik, 3 - prawy silnik

# Nagrody w funkcji fitness 'rewards'
# zwiekszona/zmniejsza za bliska/daleka odleglosc od miejsca ladowania
# zwiekszona/zmniejsza za wolniejsze/szybszy ruch ladownika
# zmniejszona kiedy ladownik jest przekrecony
# zwiekszona o 10 za kazdy dotyk nogi ladownika z noga
# zmniejszona o 0.03 za kazda klatke odpalonego bocznego silnika
# zmniejszona o 0.03 za kazda klatke odpalonego glownego silnika
# -100 lub +100 punktow za rozbicie lub ladowanie bezpiecznie
# powyzej 200 punktow mozna uznac to za sukces :)
