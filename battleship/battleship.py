import numpy as np
import pygad

gene_space = [0, 1]

board = []

with open('input6x6.txt') as f:
    lines = f.readlines()
    for line in lines:
        row = []
        for i in range(len(line)):
            if line[i] == '\n':
                break
            elif i % 2 == 0:
                row.append(line[i])
        board.append(row)

# print(board)

column_counts = list(map(int, board[0][1:]))
row_counts = list(map(int, [row[0] for row in board[1:]]))

board_size = (len(board) - 1, len(board[0]) - 1)

print(column_counts)
print(row_counts)

start_positions = {}

for i in range(1, len(board)):
    for j in range(1, len(board[0])):
        if board[i][j] == '0' or board[i][j] == '1':
            start_positions[(j - 1, i - 1)] = int(board[i][j])

print(start_positions)

ship_lengths = [3, 2, 2, 1, 1, 1]


def fitness_func(solution, solution_idx):
    # Konwertujemy rozwiązanie do postaci planszy o wymiarach 6x6.
    sol_board = np.array(solution.reshape(board_size))

    # Inicjalizujemy zmienne fitness oraz count_ships.
    fitness = 0
    count_ships = {3: 0, 2: 0, 1: 0}

    for position, value in start_positions.items():
        row, col = position
        if sol_board[col][row] != value:
            fitness -= 10

    # Sprawdzamy ilość kawałków statków w rzędach.
    for i in range(board_size[0]):
        row_sum = np.sum(sol_board[i])
        if row_sum == row_counts[i]:
            fitness += 1
        else:
            fitness -= 1

        # Sprawdzamy ilość kawałków statków w kolumnach.
    for j in range(board_size[1]):
        column_sum = np.sum(sol_board[:, j])
        if column_sum == column_counts[j]:
            fitness += 1
        else:
            fitness -= 1

    # Sprawdzamy ilość statków o różnych długościach oraz ich pozycję na planszy.
    for length in ship_lengths:
        for i in range(board_size[0]):
            for j in range(board_size[1]):
                if sol_board[i, j] == 1:
                    # Znaleziono początek statku.
                    # Przeszukiwanie planszy w celu znalezienia pozostałych części tego samego statku.
                    if length == 1:
                        count_ships[length] += 1
                    else:
                        # Sprawdzamy poziomy statek o długości length.
                        horizontal_check = True
                        for k in range(1, length):
                            if j + k >= board_size[1] or sol_board[i, j + k] == 0:
                                horizontal_check = False
                                break
                        if horizontal_check:
                            count_ships[length] += 1
                            # Usuwamy znalezione części statku z planszy, żeby nie były liczone ponownie.
                            sol_board[i, j:j + length] = 0
                            continue

                        # Sprawdzamy pionowy statek o długości length.
                        vertical_check = True
                        for k in range(1, length):
                            if i + k >= board_size[0] or sol_board[i + k, j] == 0:
                                vertical_check = False
                                break
                        if vertical_check:
                            count_ships[length] += 1
                            # Usuwamy znalezione części statku z planszy, żeby nie były liczone ponownie.
                            sol_board[i:i + length, j] = 0

    print(count_ships)

    for ship_length, count in count_ships.items():
        if ship_length == 3 and count > 1:
            fitness -= 3
        elif ship_length == 2 and count > 2:
            fitness -= 2

    return fitness


fitness_function = fitness_func

# ile chromsomów w populacji
# ile genow ma chromosom
sol_per_pop = 144
num_genes = board_size[0] * board_size[1]

# ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
# ile pokolen
# ilu rodzicow zachowac (kilka procent)
num_parents_mating = 20
num_generations = 1000
keep_parents = 2

# jaki typ selekcji rodzicow?
# sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

# w ilu punktach robic krzyzowanie?
crossover_type = "single_point"

# mutacja ma dzialac na ilu procent genow?
# trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 10

# inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
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

# uruchomienie algorytmu
ga_instance.run()

# podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))

solution_print = solution.reshape((len(column_counts), len(row_counts))).tolist()

# Wyświetlenie najlepszego rozwiązania
print("Najlepsze rozwiązanie:")
for row in solution_print:
    int_row = [int(x) for x in row]
    print(int_row)

print("Prawidlowe ulozenie:")
print([0, 0, 0, 0, 0, 1])
print([0, 1, 1, 0, 0, 0])
print([0, 0, 0, 0, 0, 0])
print([1, 0, 0, 1, 0, 1])
print([0, 0, 0, 0, 0, 1])
print([1, 1, 0, 0, 0, 1])

ga_instance.plot_fitness()
