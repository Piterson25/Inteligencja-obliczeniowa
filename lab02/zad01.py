import pygad
import numpy
import random

numbers = numpy.arange(0,11)
names = ['zegar', 'obraz-pejzaz', 'obraz-portret', 'radio', 'laptop', 'lampka nocna', 'srebrne sztucce', 'porcelana', 'figura z brazu', 'skorzana torebka', 'odkurzacz']
weights = [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]
values = [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300]
gene_space = [0, 1]

def func():
    random.shuffle(numbers)

    quantity = 0
    left = 25


    for i in numbers:
        if left - weights[i] >= 0:
            quantity += values[i]
            left -= weights[i]  

        if left == 0:
            break
    
    return quantity

def fitness_func(solution, solution_idx):
    q1 = func()
    q2 = func()

    fitness = -numpy.abs(q1 - q2)
    return fitness

fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 10
num_genes = len(weights)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 30
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
                       mutation_percent_genes=mutation_percent_genes)

#uruchomienie algorytmu
ga_instance.run()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

#tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
prediction = numpy.sum(weights*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()
