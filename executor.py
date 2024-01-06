import os
from pathlib import Path
import random

import pandas as pd
from matplotlib import pyplot as plt

from main import GeneticAlgorithm, Gene
from utility import parse_input


class Executor:
    def __init__(self, dataset_name, dataset_type, verbose=True, plots=False):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.data = pd.DataFrame()
        self.plots = plots
        self.verbose = verbose

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
                    return int(data[1]), [int(int(x) > 0) for x in data[2:-1]]
        # If the instance name is not found, return None
        return None

    def execute(self, genetic_algorithm: GeneticAlgorithm, repeat=1, limit=None):
        files = os.listdir(self.instances_folder_path())
        for i, file in enumerate(files):
            if limit < i:
                break
            self.execute_file(file, genetic_algorithm, repeat)

    def execute_file(self, file, genetic_algorithm: GeneticAlgorithm, repeat=1):
        file_path = Path(self.instances_folder_path()) / file
        input_str = open(file_path, 'r').read()
        c, w = parse_input(input_str)
        for i in range(repeat):
            found_gene: Gene = genetic_algorithm.run(c, w)

            log_df = pd.DataFrame(ga.log)
            log_df['it'] = i
            log_df['category'] = self.dataset_type
            log_df['dataset_name'] = self.dataset_name
            log_df['dataset_file'] = file

            for cnf in ga.get_config().keys():
                log_df[f"config_{cnf}"] = ga.get_config()[cnf]

            self.data = pd.concat([self.data, log_df], ignore_index=True)

            found = None if found_gene is None else found_gene.get_weights_sum()
            if self.verbose:
                print(f"{file}: Found: {found}, Best: {self.solution(file)}")
            if self.plots:
                plt.plot([p['best_gene_fitness'] for p in ga.log])
                plt.title(f"{i}, {file}: Found: {found}, Best: {self.solution(file)}")
                plt.show()


if __name__ == "__main__":
    # random.seed(4)
    e = Executor('wuf20-91', 'M', verbose=True)
    ga = GeneticAlgorithm(population_size=800, initial_mutation_rate=0.4, mutation_scaler=0.97, generations=5,
                          elites=0.01, verbose=False)
    e.execute(ga, 1, limit=5)
    e.data.to_csv("wuf20-91_test.csv")
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
