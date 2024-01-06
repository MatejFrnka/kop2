import os
import time
from pathlib import Path
import random

import pandas as pd
from matplotlib import pyplot as plt

from main import GeneticAlgorithm, Gene
from utility import parse_input

from tqdm import tqdm


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
        for i, file in tqdm(enumerate(files), desc="Files", position=0, leave=True):
            if limit <= i:
                break
            self.execute_file(file, genetic_algorithm, repeat)

    def execute_file(self, file, genetic_algorithm: GeneticAlgorithm, repeat=1):
        file_path = Path(self.instances_folder_path()) / file
        input_str = open(file_path, 'r').read()
        c, w = parse_input(input_str)
        for i in range(repeat):
            tic = time.time()
            found_gene: Gene = genetic_algorithm.run(c, w)
            toc = time.time() - tic

            log_df = pd.DataFrame(genetic_algorithm.log)
            log_df['it'] = i
            log_df['category'] = self.dataset_type
            log_df['dataset_name'] = self.dataset_name
            log_df['dataset_file'] = file
            log_df['time'] = toc
            log_df['solution_weights'] = self.solution(file)[0]

            for cnf in genetic_algorithm.get_config().keys():
                log_df[f"config_{cnf}"] = genetic_algorithm.get_config()[cnf]

            self.data = pd.concat([self.data, log_df], ignore_index=True)

            found = None if found_gene is None else found_gene.get_weights_sum()
            if self.verbose:
                print(f"{file}: Found: {found}, Best: {self.solution(file)}")
            if self.plots:
                plt.plot([p['best_gene_fitness'] for p in genetic_algorithm.log])
                plt.title(f"{i}, {file}: Found: {found}, Best: {self.solution(file)}")
                plt.show()


def measure():
    # random.seed(4)
    datasets = ['wuf20-91']
    problems = ['M', 'N', 'Q', 'R']
    algos = [
        GeneticAlgorithm(population_size=500, initial_mutation_rate=0.15, mutation_scaler=1, generations=60,
                         elites=0, verbose=False)
    ]

    df_all = pd.DataFrame()

    repeat = 6
    limit = 5
    with tqdm(total=len(datasets) * len(problems) * len(algos), desc='Overall Progress') as progress_bar:
        for dataset in datasets:
            for problem in problems:
                for ga in algos:
                    e = Executor('wuf20-91', 'M', verbose=False)
                    e.execute(ga, repeat, limit)
                    e.data.to_csv(f"{dataset}_{problem}_{time.time()}.csv")
                    df_all = pd.concat([df_all, e.data], ignore_index=True)
                    progress_bar.update(1)
    df_all.to_csv(f"{dataset}_{time.time()}.csv")


if __name__ == "__main__":
    measure()
    # e = Executor('wuf20-91', 'M', verbose=True)
    # ga = GeneticAlgorithm(population_size=800, initial_mutation_rate=0.1, mutation_scaler=1, generations=80,
    #                       elites=0.01, verbose=True)
    # e.execute(ga, 1, limit=1)
