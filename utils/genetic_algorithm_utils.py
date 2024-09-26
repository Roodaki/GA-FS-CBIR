# utils/genetic_algorithm_utils.py

import random
from deap import base, creator, tools
import numpy as np
from src.constants import (
    GA_POPULATION_SIZE,
    GA_NUMBER_OF_GENERATIONS,
    GA_CROSSOVER_PROBABILITY,
    GA_MUTATION_PROBABILITY,
    GA_MUTATION_INDEPENDENCE_PROBABILITY,
    CSV_FILE_PATH,
    GA_PRECISION_WEIGHT,
    TOURNAMENT_SIZE,
    CROSSOVER_INDP_PROBABILITY,
    LEAF_SIZE,
    K_NEIGHBORS,
)
from src.knn_image_retrieval import load_histograms_from_csv, retrieve_similar_images
from src.evaluation import calculate_metrics, load_ground_truth_labels

# Create a weighted multi-objective fitness function
creator.create(
    "FitnessWeighted",
    base.Fitness,
    weights=(GA_PRECISION_WEIGHT, -(1 - GA_PRECISION_WEIGHT)),
)  # Modify to two weights

creator.create("Individual", list, fitness=creator.FitnessWeighted)


def initialize_population(number_of_individuals, number_of_features):
    """Initialize a population of individuals with random feature selections.

    Args:
        number_of_individuals (int): Number of individuals in the population.
        number_of_features (int): Total number of features available.

    Returns:
        list: A list of initialized individuals.
    """
    population = []
    for _ in range(number_of_individuals):
        # Randomly initialize a binary vector indicating selected features
        individual = [random.randint(0, 1) for _ in range(number_of_features)]
        population.append(creator.Individual(individual))
    return population


def evaluate_individual(individual, histograms, target_labels):
    """Evaluate the fitness of an individual based on precision and feature count.

    Args:
        individual (Individual): The individual to evaluate.
        histograms (np.ndarray): Histograms of all images.
        target_labels (dict): Ground truth labels for the images.

    Returns:
        tuple: A tuple containing the weighted fitness value and average precision.
    """
    selected_features = [
        index for index, feature in enumerate(individual) if feature == 1
    ]

    if len(selected_features) == 0:
        # If no features are selected, return a low precision score and high feature count
        return (
            -100.0,
            0.0,
        )  # Return a very negative fitness score for an invalid individual

    # Filter histograms based on the selected features
    reduced_histograms = histograms[:, selected_features]

    # Initialize precision accumulation variables
    total_precision = 0.0
    total_images = len(target_labels)

    # Evaluate precision for each query image
    for i in range(total_images):
        query_histogram = reduced_histograms[i]
        retrieved_indices = retrieve_similar_images(query_histogram, reduced_histograms)

        query_filename = f"{i}.jpg"
        query_label = target_labels.get(query_filename, None)

        if query_label is None:
            continue

        retrieved_filenames = [
            f"{index}.jpg" for index in retrieved_indices if index != i
        ]
        relevant_images_count = sum(
            1 for f in target_labels if target_labels[f] == query_label
        )

        # True Positives: Correct class images retrieved
        tp = sum(1 for f in retrieved_filenames if target_labels.get(f) == query_label)
        true_positives = tp

        # False Positives: Incorrect class images retrieved
        false_positives = len(retrieved_filenames) - tp

        # Calculate precision for this image
        precision, _, _, _, _ = calculate_metrics(
            true_positives, false_positives, relevant_images_count, total_images
        )

        # Accumulate precision across all query images
        total_precision += precision

    # Compute the average precision across all images
    avg_precision = total_precision / total_images

    # Calculate the weighted fitness
    feature_count = len(selected_features)
    max_possible_features = len(individual)

    # Normalize the number of features (to ensure it's on a comparable scale to precision)
    feature_ratio = feature_count / max_possible_features

    # Weighted sum of precision and the feature count (inverted because fewer features is better)
    fitness_value = (GA_PRECISION_WEIGHT * avg_precision) - (
        (1 - GA_PRECISION_WEIGHT) * feature_ratio
    )

    # Return fitness value as a tuple (weighted fitness, average precision)
    return (fitness_value, avg_precision)


def run_genetic_algorithm():
    """Run the genetic algorithm for feature selection."""
    # Load histograms and labels
    histograms = load_histograms_from_csv(CSV_FILE_PATH)
    target_labels = load_ground_truth_labels()

    number_of_features = histograms.shape[1]
    population = initialize_population(GA_POPULATION_SIZE, number_of_features)

    # Register the genetic algorithm components
    toolbox = base.Toolbox()
    toolbox.register("mate", tools.cxUniform, indpb=CROSSOVER_INDP_PROBABILITY)
    toolbox.register(
        "mutate", tools.mutFlipBit, indpb=GA_MUTATION_INDEPENDENCE_PROBABILITY
    )
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register(
        "evaluate",
        evaluate_individual,
        histograms=histograms,
        target_labels=target_labels,
    )

    # Evolve the population
    for generation in range(GA_NUMBER_OF_GENERATIONS):
        # Evaluate the individuals
        fitnesses = list(map(toolbox.evaluate, population))
        for individual, fitness in zip(population, fitnesses):
            individual.fitness.values = (
                fitness  # fitness should be a tuple with two values
            )

        # Print details for each individual in this generation
        print(f"Generation {generation}:")
        for i, individual in enumerate(population):
            # Count the number of selected features
            num_selected_features = sum(individual)
            weighted_fitness = individual.fitness.values[0]
            precision = individual.fitness.values[1]
            print(
                f"  Individual {i + 1}: Selected Features = {num_selected_features}, "
                f"Weighted Fitness = {weighted_fitness:.4f}, Precision = {precision:.4f}"
            )
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CROSSOVER_PROBABILITY:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < GA_MUTATION_PROBABILITY:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Replace the old population by the offspring
        population[:] = offspring

        # Calculate best fitness while checking for valid fitness values
        valid_fitness = [
            ind.fitness.values[0] for ind in population if ind.fitness.valid
        ]
        if valid_fitness:  # Ensure there are valid fitness values to consider
            best_fitness = max(valid_fitness)
            print(
                f"  Generation {generation}: Best Weighted Fitness = {best_fitness:.4f}\n"
            )
        else:
            print(f"  Generation {generation}: No valid fitness values.\n")

    # Return the best individual after all generations
    best_individual = tools.selBest(population, 1)[0]
    print(
        f"Best Individual: {best_individual}, Fitness: {best_individual.fitness.values}"
    )
