import numpy as np
import pygad
import time

gene_space = [0, 1]

inputs = [
    {"column_counts": [1, 3, 1, 3, 0, 2],
     "row_counts": [2, 0, 3, 1, 3, 1],
     "start_positions": {(3, 2): 1, (0, 4): 1}},
    {"column_counts": [1, 4, 0, 3, 1, 1],
     "row_counts": [4, 1, 2, 1, 0, 2],
     "start_positions": {(5, 0): 1}},
    {"column_counts": [0, 4, 0, 2, 3, 1],
     "row_counts": [2, 1, 1, 3, 1, 2],
     "start_positions": {(5, 5): 1}},
    {"column_counts": [0, 3, 1, 5, 1, 1, 4, 1, 4, 0],
     "row_counts": [3, 1, 2, 1, 1, 2, 5, 1, 1, 3],
     "start_positions": {(5, 0): 1, (8, 3): 1, (2, 9): 1}},
    {"column_counts": [7, 0, 1, 2, 4, 0, 3, 1, 1, 1],
     "row_counts": [1, 3, 2, 3, 1, 3, 2, 2, 2, 1],
     "start_positions": {(2, 1): 1, (7, 3): 1, (3, 5): 1, (6, 6): 1, (4, 8): 1}},
    {"column_counts": [2, 1, 2, 2, 2, 3, 1, 6, 0, 1],
     "row_counts": [1, 3, 0, 4, 3, 1, 1, 3, 2, 2],
     "start_positions": {(4, 1): 1, (1, 4): 1, (7, 7): 1, (3, 9): 1, (9, 9): 1, (7, 9): 0}},
    {"column_counts": [4, 0, 0, 7, 2, 1, 5, 1, 4, 0, 5, 2, 1, 2, 1],
     "row_counts": [2, 1, 3, 1, 2, 5, 3, 3, 2, 0, 6, 1, 3, 2, 1],
     "start_positions": {(6, 0): 1, (6, 2): 0, (3, 3): 1, (8, 5): 0, (0, 6): 1, (13, 6): 1, (0, 8): 1, (4, 10): 1,
                         (3, 13): 1, (8, 13): 1}},
    {"column_counts": [1, 2, 1, 2, 5, 1, 3, 3, 3, 2, 5, 0, 2, 4, 1],
     "row_counts": [0, 1, 8, 2, 5, 1, 1, 4, 2, 2, 3, 3, 1, 1, 1],
     "start_positions": {(6, 2): 0, (0, 3): 1, (3, 5): 1, (10, 4): 1, (12, 8): 1, (13, 11): 0, (6, 12): 1, (13, 13): 1,
                         (1, 14): 1}},
    {"column_counts": [4, 2, 0, 5, 0, 1, 5, 1, 6, 2, 2, 3, 4, 0, 0],
     "row_counts": [0, 2, 0, 5, 3, 6, 2, 3, 1, 1, 4, 3, 3, 0, 2],
     "start_positions": {(9, 3): 1, (12, 4): 1, (7, 5): 1, (6, 7): 1, (6, 9): 1, (3, 8): 1, (12, 7): 1, (11, 11): 1,
                         (1, 14): 1, (8, 14): 1}}
]

for input_values in inputs:
    column_counts = input_values['column_counts']
    row_counts = input_values['row_counts']
    start_positions = input_values['start_positions']

    board_size = (len(column_counts), len(row_counts))

    ship_lengths = []
    ship_count = {}

    if board_size[0] == 6:
        ship_lengths = [3, 2, 2, 1, 1, 1]
        ship_count = {3: 0, 2: 0, 1: 0}
    elif board_size[0] == 10:
        ship_lengths = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]
        ship_count = {4: 0, 3: 0, 2: 0, 1: 0}
    elif board_size[0] == 15:
        ship_lengths = [5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1]
        ship_count = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}

    print(f"Rozmiar inputu: {board_size[0]}x{board_size[1]}")
    print("Kolumny:", column_counts)
    print("Wiersze:", row_counts)
    print("Poczatkowe pozycje:", start_positions)


    def fitness_func(solution, solution_idx):
        # Konwertujemy rozwiązanie do postaci planszy o wymiarach 6x6.
        sol_board = np.array(solution.reshape(board_size))

        # Inicjalizujemy zmienne fitness oraz count_ships.
        fitness = 0
        count_ships = ship_count

        for position, value in start_positions.items():
            row, col = position
            if sol_board[col][row] != value:
                fitness -= 50

        # Sprawdzamy ilość kawałków statków w rzędach.
        for i in range(board_size[0]):
            row_sum = np.sum(sol_board[i])
            if row_sum == row_counts[i]:
                fitness += 10
            elif row_sum > row_counts[i]:
                fitness -= 30
            else:
                fitness -= 10

        # Sprawdzamy ilość kawałków statków w kolumnach.
        for j in range(board_size[1]):
            column_sum = np.sum(sol_board[:, j])
            if column_sum == column_counts[j]:
                fitness += 10
            elif column_sum > column_counts[j]:
                fitness -= 30
            else:
                fitness -= 10

        # Sprawdzamy ilość statków o różnych długościach oraz ich pozycję na planszy.
        for length in ship_lengths:
            for i in range(board_size[0]):
                for j in range(board_size[1]):
                    if sol_board[i, j] == 1:
                        # Znaleziono początek statku.
                        # Przeszukiwanie planszy w celu znalezienia pozostałych części tego samego statku.
                        if length == 1:
                            if not (j + 1 < board_size[1] and sol_board[i, j + 1] == 1) \
                                    and not (i + 1 < board_size[0] and sol_board[i + 1, j] == 1) \
                                    and not (i - 1 >= 0 and sol_board[i - 1, j] == 1) \
                                    and not (j - 1 >= 0 and sol_board[i, j - 1] == 1) \
                                    and not (j - 1 >= 0 and i - 1 >= 0 and sol_board[i - 1, j - 1] == 1) \
                                    and not (
                                    j + 1 < board_size[1] and i + 1 < board_size[0] and sol_board[i + 1, j + 1] == 1) \
                                    and not (j + 1 < board_size[1] and i - 1 >= 0 and sol_board[i - 1, j + 1] == 1) \
                                    and not (j - 1 >= 0 and i + 1 < board_size[0] and sol_board[i + 1, j - 1] == 1):
                                count_ships[length] += 1
                        else:
                            # Sprawdzamy poziomy statek o długości length.
                            horizontal_check = True
                            for k in range(1, length):
                                if j + k >= board_size[1] or sol_board[i, j + k] == 0 or \
                                        i + 1 < board_size[0] and sol_board[i + 1, j + k] == 1 or \
                                        i - 1 >= 0 and sol_board[i - 1, j + k] == 1 or \
                                        j - 1 >= 0 and sol_board[i, j - 1] == 1:
                                    horizontal_check = False
                                    break
                            if horizontal_check:
                                count_ships[length] += 1
                                fitness += 10
                                # Usuwamy znalezione części statku z planszy, żeby nie były liczone ponownie.
                                # sol_board[i, j:j + length] = 0
                                continue

                            # Sprawdzamy pionowy statek o długości length.
                            vertical_check = True
                            for k in range(1, length):
                                if i + k >= board_size[0] or sol_board[i + k, j] == 0 or \
                                        j + 1 < board_size[1] and sol_board[i + k, j + 1] == 1 or \
                                        j - 1 >= 0 and sol_board[i + k, j - 1] == 1 or \
                                        i - 1 >= 0 and sol_board[i - 1, j] == 1:
                                    vertical_check = False
                                    break
                            if vertical_check:
                                count_ships[length] += 1
                                fitness += 10
                                # Usuwamy znalezione części statku z planszy, żeby nie były liczone ponownie.
                                # sol_board[i:i + length, j] = 0

        # for ship_length, count in count_ships.items():
        #     if ship_length == 2 and count == 2:
        #         fitness += ship_length
        #     elif ship_length == 3 and count == 1:
        #         fitness += ship_length

        return fitness


    fitness_function = fitness_func

    # ile chromsomów w populacji
    # ile genow ma chromosom
    sol_per_pop = board_size[0] * board_size[1] * 3
    num_genes = board_size[0] * board_size[1]

    # ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
    # ile pokolen
    # ilu rodzicow zachowac (kilka procent)
    num_parents_mating = 18
    num_generations = 500
    keep_parents = 5

    # jaki typ selekcji rodzicow?
    # sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
    parent_selection_type = "sss"

    # w ilu punktach robic krzyzowanie?
    crossover_type = "single_point"

    # mutacja ma dzialac na ilu procent genow?
    # trzeba pamietac ile genow ma chromosom
    mutation_type = "random"
    mutation_percent_genes = 10

    start_time = time.time()

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

    end_time = round(time.time() - start_time, 4)

    # podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Czas wykonywania algorytmu = {end_time}s")
    print("Fitness najlepszego rozwiazania = {solution_fitness}".format(
        solution_fitness=solution_fitness))

    solution_print = solution.reshape((len(column_counts), len(row_counts))).tolist()

    # Wyświetlenie najlepszego rozwiązania
    print("Parametry najlepszego rozwiazania:")
    for row in solution_print:
        int_row = [int(x) for x in row]
        print(int_row)
    print('=' * 20)
    print()

    ga_instance.plot_fitness()
