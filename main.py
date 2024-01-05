import random

import matplotlib.pyplot as plt

from utility import parse_input


def evaluate(clauses, weights, evaluation):
    not_fulfilled_cnt = sum(
        1 for clause in clauses
        if not any(
            (lit < 0 and evaluation[abs(lit) - 1] == 0) or
            (lit > 0 and evaluation[lit - 1] == 1)
            for lit in clause
        )
    )
    weights_sum = sum(weight for i, weight in enumerate(weights) if evaluation[i] == 1)

    return not_fulfilled_cnt, weights_sum


def fitness(not_fulfilled_cnt, weight_sum, gene_len: int):
    fulfilled_cnt = gene_len - not_fulfilled_cnt
    return ((fulfilled_cnt + 1) / (not_fulfilled_cnt + 1)) * weight_sum
    # return weight_sum/ (not_fulfilled_cnt + 1)
    # return fulfilled_cnt + weight_sum


class Gene:
    def __init__(self, length, clauses, weights):
        self.length = length
        self.evaluation = [random.randint(0, 1) for _ in range(length)]
        self.clauses, self.weights = clauses, weights
        self.not_fulfilled_cnt, self.weights_sum = evaluate(clauses, weights, self.evaluation)
        self.fitness = fitness(self.not_fulfilled_cnt, self.weights_sum, len(weights))

    def mutate(self):
        mutation_index = random.randint(0, self.length - 1)
        self.evaluation[mutation_index] = 1 - self.evaluation[mutation_index]
        self.not_fulfilled_cnt, self.weights_sum = evaluate(self.clauses, self.weights, self.evaluation)
        self.fitness = fitness(self.not_fulfilled_cnt, self.weights_sum, len(self.weights))

    def __str__(self):
        return f"Best = {round(self.fitness, 2)}, NF: {self.not_fulfilled_cnt}, W: {self.weights_sum}, ({self.evaluation})"

    def __repr__(self):
        return f"Gene({round(self.fitness, 2)})"


class GeneticAlgorithm:
    def __init__(self, population_size=100, initial_mutation_rate=0.01, mutation_scaler=1.,
                 generations=100, verbose=True, elites=0.):
        self.population_size = population_size
        self.initial_mutation_rate = initial_mutation_rate
        self.mutation_scaler = mutation_scaler
        self.generations = generations
        self.verbose = verbose
        self.elites_cnt = int(population_size * elites)

        self.mutation_rate = initial_mutation_rate
        self.fitness_log = []
        self.clauses = None
        self.weights = None
        self.best_valid = None
        self.population = None

    def adjust_rates(self):
        self.mutation_rate *= self.mutation_scaler

    def select_parent(self):
        total_fitness = sum(gene.fitness for gene in self.population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for gene in self.population:
            current += gene.fitness
            if current > pick:
                return gene

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, len(self.weights) - 1)
        child = Gene(len(self.weights), self.clauses, self.weights)
        child.evaluation = parent1.evaluation[:crossover_point] + parent2.evaluation[crossover_point:]
        return child

    def mutate_population(self):
        for gene in self.population:
            if random.random() < self.mutation_rate:
                gene.mutate()

    def run(self, clauses, weights):
        self.mutation_rate = self.initial_mutation_rate
        self.fitness_log = []
        self.clauses = clauses
        self.weights = weights
        self.best_valid = None
        self.population = [Gene(len(weights), clauses, weights) for _ in range(self.population_size)]

        for generation in range(self.generations):
            self.adjust_rates()
            new_population = []
            if self.elites_cnt > 0:
                elites = sorted(self.population, key=lambda gene: gene.fitness, reverse=True)[:self.elites_cnt]
                new_population = elites

            while len(new_population) < self.population_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent1, parent2)
                new_population.extend([child1, child2])

            self.population = new_population
            self.mutate_population()

            best_gene = max(self.population, key=lambda gene: gene.fitness)
            valid_genes = [x for x in self.population if x.not_fulfilled_cnt == 0]
            self.fitness_log.append(best_gene.fitness)
            if len(valid_genes) != 0:
                self.best_valid = max(valid_genes, key=lambda gene: gene.fitness)
            if self.verbose:
                print(f"{generation}: {str(best_gene)}, Mutation: {self.mutation_rate}")

        return self.best_valid
