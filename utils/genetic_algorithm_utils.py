import random
import csv
from deap import base, creator, tools
import multiprocessing
from src.constants import (
    GA_POPULATION_SIZE,
    GA_NUMBER_OF_GENERATIONS,
    GA_CROSSOVER_PROBABILITY,
    GA_BASE_MUTATION_PROBABILITY,
    GA_MUTATION_INDEPENDENCE_PROBABILITY,
    CSV_FILE_PATH,
    GA_PRECISION_WEIGHT,
    IMAGE_FILE_EXTENSION,
    TOURNAMENT_SIZE,
    CROSSOVER_INDP_PROBABILITY,
    GA_FEATURE_SELECTION_OUTPUT_FILE,
    GA_RESULTS_CSV_FILE,
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


def log_selected_features(generation, population):
    """Logs selected feature indices for each individual in the population to a txt file."""
    with open(GA_FEATURE_SELECTION_OUTPUT_FILE, "a") as file:
        file.write(f"Generation {generation}:\n")
        for individual in population:
            selected_features = [
                i for i, feature in enumerate(individual) if feature == 1
            ]
            file.write(f"{selected_features}\n")


def log_ga_results(generation, sorted_population):
    """Logs GA results to a CSV file and prints them to the console."""
    with open(GA_RESULTS_CSV_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        print(f"\nGeneration {generation} Results:")
        print(
            f"{'Individual':<12} {'Fitness':<20} {'Num Features':<15} {'Mean Precision':<15}"
        )
        print("=" * 70)

        # For each individual in sorted population
        for i, individual in enumerate(sorted_population):
            num_selected_features = sum(
                individual
            )  # Count the number of selected features
            avg_precision = individual.fitness.values[1]
            fitness_value = individual.fitness.values[0]
            row = [
                generation,
                fitness_value,
                f"{i + 1}",
                num_selected_features,
                avg_precision,
            ]
            writer.writerow(row)

            # Print the formatted output
            print(f"{row[2]:<12} {row[1]:<20.4f} {row[3]:<15} {row[4]:<15.4f}")


def initialize_population(number_of_individuals, number_of_features, max_attempts=1000):
    """Initialize a population of unique individuals with random feature selections."""
    population = set()
    attempts = 0
    while len(population) < number_of_individuals and attempts < max_attempts:
        individual = tuple(random.randint(0, 1) for _ in range(number_of_features))
        if individual not in population:
            population.add(individual)
        attempts += 1

    if len(population) < number_of_individuals:
        print(
            f"Warning: Only {len(population)} unique individuals were generated after {max_attempts} attempts."
        )

    # Convert tuples back to Individual objects
    return [creator.Individual(list(ind)) for ind in population]


def evaluate_individual(individual, histograms, target_labels):
    """Evaluate the fitness of an individual based on precision and feature count."""
    selected_features = [
        index for index, feature in enumerate(individual) if feature == 1
    ]
    if len(selected_features) == 0:
        return (
            -100.0,
            0.0,
        )  # Return a very negative fitness score for an invalid individual

    reduced_histograms = histograms[:, selected_features]
    total_precision = 0.0
    total_images = min(len(target_labels), reduced_histograms.shape[0])

    for i in range(total_images):
        query_histogram = reduced_histograms[i]
        _, retrieved_indices = retrieve_similar_images(
            query_histogram, reduced_histograms
        )
        query_filename = f"{i}{IMAGE_FILE_EXTENSION}"
        query_label = target_labels.get(query_filename, None)
        if query_label is None:
            continue

        retrieved_filenames = [
            f"{index}{IMAGE_FILE_EXTENSION}"
            for index in retrieved_indices
            if index != i
        ]
        relevant_images_count = sum(
            1 for f in target_labels if target_labels[f] == query_label
        )

        tp = sum(1 for f in retrieved_filenames if target_labels.get(f) == query_label)
        precision, _, _, _, _ = calculate_metrics(
            tp, len(retrieved_filenames) - tp, relevant_images_count, total_images
        )
        total_precision += precision

    avg_precision = total_precision / total_images
    feature_count = len(selected_features)
    max_possible_features = len(individual)
    feature_ratio = feature_count / max_possible_features

    fitness_value = (GA_PRECISION_WEIGHT * avg_precision) - (
        (1 - GA_PRECISION_WEIGHT) * feature_ratio
    )
    return fitness_value, avg_precision


def run_genetic_algorithm():
    """Run the genetic algorithm for feature selection with parallel evaluation."""
    histograms = load_histograms_from_csv(CSV_FILE_PATH)
    target_labels = load_ground_truth_labels()
    number_of_features = histograms.shape[1]
    population = initialize_population(GA_POPULATION_SIZE, number_of_features)

    # Initialize the toolbox
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

    # CSV header
    with open(GA_RESULTS_CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Generation",
                "Best Weighted Fitness",
                "Individual",
                "Number of Features",
                "Mean Precision",
            ]
        )

    # Create a pool of workers
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        toolbox.register("map", pool.map)  # Register parallel map

        # Evolve the population
        for generation in range(GA_NUMBER_OF_GENERATIONS):
            # Evaluate the entire population in parallel
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for individual, fitness in zip(population, fitnesses):
                individual.fitness.values = fitness

            # Sort population based on fitness and log results
            sorted_population = sorted(
                population, key=lambda ind: ind.fitness.values[0], reverse=True
            )
            best_fitness = sorted_population[0].fitness.values[0]
            print(
                f"Generation {generation}: Best Weighted Fitness = {best_fitness:.4f}"
            )

            # Log selected features and GA results
            log_selected_features(generation, sorted_population)
            log_ga_results(generation, sorted_population)

            # Select, clone, and apply crossover and mutation to produce offspring
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover on offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < GA_CROSSOVER_PROBABILITY:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation on offspring
            for mutant in offspring:
                if random.random() < GA_BASE_MUTATION_PROBABILITY:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Update the population for the next generation
            population[:] = offspring

    # Log the best individual after all generations
    best_individual = tools.selBest(population, 1)[0]
    selected_features_indices = [
        i for i, feature in enumerate(best_individual) if feature == 1
    ]
    num_selected_features = len(selected_features_indices)
    final_avg_precision = best_individual.fitness.values[1]

    print(
        f"Best Individual: {best_individual}, Fitness: {best_individual.fitness.values}"
    )
    print(f"\tNumber of Selected Features: {num_selected_features}")
    print(f"\tIndices of Selected Features: {selected_features_indices}")
    print(f"\tFinal Mean Precision: {final_avg_precision:.4f}")
