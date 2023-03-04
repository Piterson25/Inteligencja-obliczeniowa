import pygad
import numpy
import time

start = time.time()

names = ['zegar', 'obraz-pejzaz', 'obraz-portret', 'radio', 'laptop', 'lampka nocna', 'srebrne sztucce', 'porcelana', 'figura z brazu', 'skorzana torebka', 'odkurzacz']
weights = [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]
values = [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300]
max_weight = 25
gene_space = [0, 1]

# definiujemy funkcję fitness
def fitness_func(solution, solution_idx):
    weight = 0
    value = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            weight += weights[i]
            value += values[i]
    if weight > max_weight:
        value = 0
    return value

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 10
num_genes = len(weights)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 50
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = int(100/num_genes + 1)

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
                       mutation_percent_genes=mutation_percent_genes,
                       stop_criteria=["reach_1600"])

#uruchomienie algorytmu
ga_instance.run()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
prediction = numpy.sum(weights*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

result = "Przedmioty: "

for i in range(len(solution)):
    if solution[i] == 1:
        result += names[i] + '(' + str(values[i]) + ')' + ', '

print(result)

print("Mineło pokoleń:", ga_instance.generations_completed)

time = time.time() - start
print("Czas trwania algorytmu: " + str(round(time,4)) + "s")

#Czasy: 0.011s, 0.004s, 0.017s, 0.006s, 0.007s, 0.005s, 0.008s, 0.01s, 0.01s, 0.009s
#Srednia: 0.009s

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()
