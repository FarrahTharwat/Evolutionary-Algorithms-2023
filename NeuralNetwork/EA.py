from deap import base, creator, tools, algorithms
from NeuralNetwork.NN import create_model
from checkers.game import Game
from checkers.constants import RED, WHITE
# Constants
POPULATION_SIZE = 100
NUM_GENERATIONS = 50

# Evaluation function
def evaluate(individual):
    game = Game(None)
    for move in individual:
        row, col = move  # Extract row and column from the move
        game.select(row, col)  # Apply the move to the game board
    winner = game.winner()  # Check if there is a winner
    if winner == RED:
        fitness_score = 1  # Assign a fitness score for winning
    elif winner == WHITE:
        fitness_score = -1  # Assign a fitness score for losing
    else:
        fitness_score = 0  # Assign a fitness score for draw
    return fitness_score,

# Register genetic operators
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initRepeat, creator.Individual, create_model, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Evolutionary algorithm
def evolve():
    population = toolbox.population(n=POPULATION_SIZE)

    for gen in range(NUM_GENERATIONS):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)

        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

        best_individual = tools.selBest(population, 1)[0]
        print(f"Generation {gen + 1}: Best Fitness = {best_individual.fitness.values[0]}")

    best_individual = tools.selBest(population, 1)[0]
    return best_individual

# Main function
def main():
    best_individual = evolve()
    print("Best individual:", best_individual)

if __name__ == "__main__":
    main()
