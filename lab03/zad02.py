import pygad
import numpy
import time

gene_space = [0, 1, 2, 3]

#0 - lewo
#1 - dół
#2 - prawo
#3 - góra

path = 'C:\Studia\Inteligencja obliczeniowa\Inteligencja-obliczeniowa\lab03\labirynth.txt'

labyrinth = []
max_steps = 30

with open(path, "r") as f:
    labyrinth = [[char for char in line.strip()] for line in f]

def vector_distance(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


# definiujemy funkcję fitness
def fitness_func(solution, solution_idx):
    position = numpy.array([1, 1])
    penalty = 0
    steps = 0

    for move in solution:
        if move == 0:
            if position[0] > 0 and labyrinth[position[0] - 1][position[1]] != '#':  # up
                position[0] -= 1
                steps += 1
            else:
                penalty += 1
        elif move == 1:
            if position[1] < 11 and labyrinth[position[0]][position[1] + 1] != '#':  # right
                position[1] += 1
                steps += 1
            else:
                penalty += 1
        elif move == 2:
            if position[0] < 11 and labyrinth[position[0] + 1][position[1]] != '#':  # down
                position[0] += 1
                steps += 1
            else:
                penalty += 1
        elif move == 3:
            if position[1] > 0 and labyrinth[position[0]][position[1] - 1] != '#':  # left
                position[1] -= 1
                steps += 1
            else:
                penalty += 1

        if labyrinth[position[0]][position[1]] == 'E' or steps > max_steps:
            break

    return vector_distance([1, 1], position) - penalty

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 144
num_genes = max_steps

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 10
num_generations = 200
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

start = time.time()

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

end = time.time() - start

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
prediction = numpy.sum(solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

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

    if labyrinth[position[0]][position[1]] == 'E':
        break

# wyświetlenie ścieżki
print("Best path:", path)

print("Czas trwania algorytmu: " + str(round(end, 4)) + "s")

#Czasy: 1.6794s, 1.7545s, 1.7649s, 1.7675s, 1.7425s, 1.7849s, 1.8671s, 1.821s, 1.8064s, 1.7724s
#Srednia: 1,776s

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()
