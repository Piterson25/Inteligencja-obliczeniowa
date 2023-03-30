import pygad
import numpy as np
import time

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

gene_space = []

col_edges = list(map(int, board[0][1:]))
row_edges = list(map(int, [row[0] for row in board[1:]]))

gene_space = [0, 1]

board_values = []
for i in range(1, len(board)):
    row_values = []
    for j in range(1, len(board[0])):
        if board[i][j] not in (0, 1):
            board[i][j] = 0
        row_values.append(int(board[i][j]))
    board_values.append(row_values)

# print(col_edges)
# print(row_edges)
# print(board_values)


def is_valid_ship(visited, row, col, k, ship_size):
    for l in range(ship_size):
        if row+l >= len(visited) or col+l >= len(visited[0]) or visited[row+l][col+l] == 1:
            return False
        if row+l > 0 and col+l > 0 and visited[row+l-1][col+l-1] == 1:
            return False
        if row+l < len(visited)-1 and col+l > 0 and visited[row+l+1][col+l-1] == 1:
            return False
        if col+l > 0 and visited[row+l][col+l-1] == 1:
            return False
        if col+l < len(visited[0])-1 and visited[row+l][col+l+1] == 1:
            return False
    return True

def fitness_func(solution, solution_idx):
    # Reshape solution into 2D grid
    grid = np.reshape(solution, (len(row_edges), len(col_edges)))

    # Initialize score and visited grid
    score = 0
    visited = np.zeros((len(row_edges), len(col_edges)))

    # Check each ship in the solution
    for ship_size in [3, 2, 2, 1, 1, 1]:
        for i in range(len(row_edges)):
            for j in range(len(col_edges)):
                # Check if ship fits in grid starting at (i, j)
                if i + ship_size <= len(row_edges):
                    ship = grid[i:i+ship_size, j]
                    if np.sum(ship) == ship_size and is_valid_ship(visited, i, j, 0, ship_size):
                        score += 1
                        for k in range(ship_size):
                            visited[i+k][j+k] = 1
                if j + ship_size <= len(col_edges):
                    ship = grid[i, j:j+ship_size]
                    if np.sum(ship) == ship_size and is_valid_ship(visited, i, j, 0, ship_size):
                        score += 1
                        for k in range(ship_size):
                            visited[i][j+k] = 1
                empty_cells = np.sum(visited == 0)
                if empty_cells > 6:
                    return score - empty_cells / (len(row_edges) * len(col_edges))
                
    return score - empty_cells / (len(row_edges) * len(col_edges))



fitness_function = fitness_func

# ile chromsomów w populacji
# ile genow ma chromosom
sol_per_pop = 50
num_genes = len(col_edges) * len(row_edges)

# ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
# ile pokolen
# ilu rodzicow zachowac (kilka procent)
num_parents_mating = 20
num_generations = 100
keep_parents = 2

# jaki typ selekcji rodzicow?
# sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

# w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

# mutacja ma dzialac na ilu procent genow?
# trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 5

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

# tutaj dodatkowo wyswietlamy sume wskazana przez jedynki
prediction = np.sum(solution)
print("Predicted output based on the best solution : {prediction}".format(
    prediction=prediction))

solution_print = solution.reshape((len(col_edges), len(row_edges))).tolist()

# Wyświetlenie najlepszego rozwiązania
print("Najlepsze rozwiązanie:")
for row in solution_print:
    int_row = [int(x) for x in row]
    print(int_row)

ga_instance.plot_fitness()
