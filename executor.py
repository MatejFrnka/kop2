import os
from pathlib import Path
import random

from matplotlib import pyplot as plt

from main import GeneticAlgorithm
from utility import parse_input


class Executor:
    def __init__(self, dataset_name, dataset_type):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type

    def instances_folder_path(self):
        return Path(self.dataset_name) / f"{self.dataset_name}-{self.dataset_type}"

    def results_file_path(self):
        return Path(self.dataset_name) / f"{self.dataset_name}-{self.dataset_type}-opt.dat"

    def solution(self, instance_file):
        instance_name = instance_file[1:-6]
        with open(self.results_file_path(), 'r') as file:
            for line in file:
                data = line.split()
                if data[0] == instance_name:
                    return int(data[1])
        # If the instance name is not found, return None
        return None

    def execute(self, genetic_algorithm: GeneticAlgorithm, repeat = 1):
        files = os.listdir(self.instances_folder_path())
        file = random.choice(files)
        input_str = open(Path(self.instances_folder_path()) / file, 'r').read()
        c, w = parse_input(input_str)
        print(f"Best: {self.solution(file)}")
        for i in range(repeat):
            found_gene = genetic_algorithm.run(c, w)
            found = None if found_gene is None else found_gene.weights_sum
            print(f"{file}: Found: {found}, Best: {self.solution(file)}")
            plt.plot(ga.fitness_log)
            plt.title(f"{i}, {file}: Found: {found}, Best: {self.solution(file)}")
            plt.show()


if __name__ == "__main__":
    random.seed(5)
    e = Executor('wuf20-91', 'Q')
    ga = GeneticAlgorithm(population_size=800, initial_mutation_rate=0.4, mutation_scaler=0.99, generations=100,
                          elites=0.1, verbose=True)
    e.execute(ga, 10)
    # path = 'wu/f20-91/wuf20-91-M/wuf20-01.mwcnf'
    # input_str = open(path, 'r').read()
    # c, w = parse_input(input_str)
    # best = [0 if int(f) < 0 else 1 for f in
    #         "1 -2 -3 4 -5 6 -7 -8 -9 10 -11 -12 13 14 15 -16 17 -18 -19 20".split(" ")]
    # print(best)
    # print(evaluate(c, w, best))
    #

    # ga.run()
    # print(ga.best_valid)
    # print(f"Weights: {ga.best_valid.weights_sum}")
    #

