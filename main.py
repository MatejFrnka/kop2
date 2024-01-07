import random
from cachetools import cached, LRUCache
from typing import List

import numpy as np
from cachetools.keys import hashkey

cache = LRUCache(maxsize=3_000)


@cached(cache=cache, key=lambda clauses, weights, evaluation: hashkey(tuple(evaluation)))
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


class Gene:
    def __init__(self, length, clauses, weights):
        self.length = length
        self.evaluation = [random.randint(0, 1) for _ in range(length)]
        self.clauses, self.weights = clauses, weights
        self._not_fulfilled_cnt, self._weights_sum = evaluate(clauses, weights, self.evaluation)

    def mutate(self):
        mutation_index = random.randint(0, self.length - 1)
        self.evaluation[mutation_index] = 1 - self.evaluation[mutation_index]
        self._not_fulfilled_cnt, self._weights_sum = evaluate(self.clauses, self.weights, self.evaluation)

    def get_not_fulfilled_cnt(self):
        return self._not_fulfilled_cnt

    def get_weights_sum(self):
        return self._weights_sum

    def __str__(self):
        return f"NF: {self._not_fulfilled_cnt}, W: {self._weights_sum}, ({self.evaluation})"

    def __repr__(self):
        return self.__str__()


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
        self.log = []
        self.clauses = None
        self.weights = None
        self.best_valid = None
        self.population = None

    def get_config(self):
        return {
            "population_size": self.population_size,
            "initial_mutation_rate": self.initial_mutation_rate,
            "mutation_scaler": self.mutation_scaler,
            "generations": self.generations,
            "elites_cnt": self.elites_cnt,
        }

    def fitness(self, gene: Gene, generation: int):
        clauses_cnt = len(self.clauses)
        fulfilled_cnt = clauses_cnt - gene.get_not_fulfilled_cnt()
        return (fulfilled_cnt / (gene.get_not_fulfilled_cnt() + 1)) * gene.get_weights_sum()

        # gen_pct = generation / self.generations
        # base = fulfilled_cnt / (gene.get_not_fulfilled_cnt() + 1)
        # base_pow = base ** (1 + gen_pct)
        # return base_pow * np.log(gene.get_weights_sum())

        # if fulfilled_cnt == 0:
        #     return gene.get_weights_sum() + sum(gene.weights)
        # return fulfilled_cnt * 1000

        # return weight_sum / (not_fulfilled_cnt + 1)
        # return fulfilled_cnt + weight_sum

    def adjust_rates(self):
        self.mutation_rate *= self.mutation_scaler

    def select_parent(self, generation):
        total_fitness = sum(self.fitness(gene, generation) for gene in self.population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for gene in self.population:
            current += self.fitness(gene, generation)
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
        self.log = []
        self.clauses = clauses
        self.weights = weights
        self.best_valid = None
        self.population: List[Gene] = [Gene(len(weights), clauses, weights) for _ in range(self.population_size)]
        cache.clear()

        for generation in range(self.generations):
            self.adjust_rates()
            new_population = []
            if self.elites_cnt > 0:
                elites = sorted(self.population, key=lambda gene: self.fitness(gene, generation), reverse=True)[
                         :self.elites_cnt]
                new_population = elites

            while len(new_population) < self.population_size:
                parent1 = self.select_parent(generation)
                parent2 = self.select_parent(generation)
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent1, parent2)
                new_population.extend([child1, child2])

            self.population = new_population
            self.mutate_population()

            genes_fitness = [(self.fitness(gene, generation), gene) for gene in self.population]
            best_gene_fitness, best_gene = max(genes_fitness, key=lambda d: d[0])
            valid_genes = [x for x in self.population if x.get_not_fulfilled_cnt() == 0]
            best_valid_gene = None if len(valid_genes) == 0 else max(valid_genes, key=lambda d: d.get_weights_sum())
            self.log.append(
                {
                    'best_gene_fitness': best_gene_fitness,
                    'best_gene_not_fulfilled': best_gene.get_not_fulfilled_cnt(),
                    'best_gene_weight': best_gene.get_weights_sum(),
                    'best_valid_fitness': None if best_valid_gene is None else self.fitness(best_valid_gene,
                                                                                            generation),
                    'best_valid_weight': None if best_valid_gene is None else best_valid_gene.get_weights_sum(),
                    'valid_count': len(valid_genes),
                    'avg_fitness': np.mean([k[0] for k in genes_fitness]),
                    'median_fitness': np.median([k[0] for k in genes_fitness]),
                    'std_fitness': np.std([k[0] for k in genes_fitness]),
                    'avg_not_fulfilled': np.mean([w.get_not_fulfilled_cnt() for w in self.population]),
                    'avg_weight': np.mean([w.get_weights_sum() for w in self.population]),
                    'generation': generation
                })
            if len(valid_genes) != 0:
                self.best_valid = max(valid_genes, key=lambda gene: self.fitness(gene, generation))
            if self.verbose:
                print(f"{generation}: Best: {best_gene_fitness}, {str(best_gene)}, Mutation: {self.mutation_rate}")

        return self.best_valid
