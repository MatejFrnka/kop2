import random

import matplotlib.pyplot as plt


def parse_input(input_str):
    lines = input_str.split('\n')
    weights = []
    clauses = []
    for line in lines:
        if line.startswith('w'):
            weights = [int(x) for x in line.split()[1:-1]]
        elif not line.startswith('c') and not line.startswith('p') and line.strip():
            clauses.append([int(x) for x in line.split()[:-1]])
    return clauses, weights


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
    def __init__(self, clauses, weights, population_size=100, initial_mutation_rate=0.01, mutation_scaler=1.,
                 generations=100, verbose=True, elites=0.):
        self.clauses = clauses
        self.weights = weights
        self.population_size = population_size
        self.mutation_rate = initial_mutation_rate
        self.mutation_scaler = mutation_scaler
        self.generations = generations
        self.population = [Gene(len(weights), clauses, weights) for _ in range(population_size)]
        self.best_valid = None
        self.fitness_log = []
        self.verbose = verbose
        self.elites_cnt = int(population_size * elites)

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

    def run(self):
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

        return max(self.population, key=lambda gene: gene.fitness)


if __name__ == "__main__":
    path = 'wuf20-91/wuf20-91-M/wuf20-01.mwcnf'
    input_str = open(path, 'r').read()
    c, w = parse_input(input_str)
    best = [0 if int(f) < 0 else 1 for f in
            "1 -2 -3 4 -5 6 -7 -8 -9 10 -11 -12 13 14 15 -16 17 -18 -19 20".split(" ")]
    print(best)
    print(evaluate(c, w, best))

    ga = GeneticAlgorithm(c, w, population_size=100, initial_mutation_rate=0.3, mutation_scaler=0.98, generations=100,
                          elites=0., verbose=False)
    ga.run()
    print(ga.best_valid)
    print(f"Weights: {ga.best_valid.weights_sum}")

    plt.plot(ga.fitness_log)
    plt.show()
