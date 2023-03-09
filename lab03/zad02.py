import pygad
import numpy

gene_space = [0, 1, 2, 3]

path = 'C:\Studia\Inteligencja obliczeniowa\Inteligencja-obliczeniowa\lab03\labirynth.txt'

labyrinth = []
max_steps = 30

with open(path, "r") as f:
    labyrinth = [[char for char in line.strip()] for line in f]


# definiujemy funkcję fitness
def fitness_func(solution, solution_idx):
    position = numpy.array([1, 1])
    for move in solution:
        if move == 0 and position[0] > 0 and labyrinth[position[0] - 1][position[1]] != '#':  # up
            position[0] -= 1
        elif move == 1 and position[1] < 11 and labyrinth[position[0]][position[1] + 1] != '#':  # right
            position[1] += 1
        elif move == 2 and position[0] < 11 and labyrinth[position[0] + 1][position[1]] != '#':  # down
            position[0] += 1
        elif move == 3 and position[1] > 0 and labyrinth[position[0]][position[1] - 1] != '#':  # left
            position[1] -= 1

        if labyrinth[position[0]][position[1]] == 'E':
            return 1 / (len(solution) + 1)
    return 1 / (len(solution) + 1)

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 144
num_genes = 144

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 6
num_generations = 100
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 10

#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

#uruchomienie algorytmu
ga_instance.run()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
prediction = numpy.sum(solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

# obliczenie ścieżki
position = numpy.array([1, 1])
path = [(1,1)]
for move in solution:
    if move == 0 and position[0] > 0 and labyrinth[position[0] - 1][position[1]] != '#':  # up
        position[0] -= 1
    elif move == 1 and position[1] < 11 and labyrinth[position[0]][position[1] + 1] != '#':  # right
        position[1] += 1
    elif move == 2 and position[0] < 11 and labyrinth[position[0] + 1][position[1]] != '#':  # down
        position[0] += 1
    elif move == 3 and position[1] > 0 and labyrinth[position[0]][position[1] - 1] != '#':  # left
        position[1] -= 1
    path.append(tuple(position))

# wyświetlenie ścieżki
print("Best path:", path)

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()
