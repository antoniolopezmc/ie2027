"""Mine diverse top-k subgroup lists (phenotypes) with the DSLM algorithm of the subgroups library, for each
configuration and each combination of hyperparameters of the grid.

Usage:
    python mine_phenotypes.py <view> <configuration|all> [number_of_processes]

Example:
    python mine_phenotypes.py clinical all 4

It requires the candidate subgroups (python mine_subgroups.py <view> all, in Experiments/subgroups).
Outputs: results/<view>/mining/execution_log.txt and, in results/<view>/mining/<configuration>/:
    - lists/<beta>_<pos_o>_<neg_o>.txt: diverse top-k subgroup lists (DSLM output).
    - log/log_<beta>_<pos_o>_<neg_o>.txt: log with the quality and the coverage.
"""

import sys
from datetime import datetime
from typing import Union
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from subgroups.algorithms import DSLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils

def mine_diverse_topk_subgroup_lists(dataset : pd.DataFrame,
                                     subgroups_input_file_path : Union[str, Path],
                                     target_attribute_name : str,
                                     target_attribute_value : str,
                                     maximum_number_of_subgroup_lists : int,
                                     maximum_number_of_subgroups_per_subgroup_list : int,
                                     beta_parameter : float,
                                     maximum_positive_overlap : float,
                                     maximum_negative_overlap : float,
                                     diverse_topk_subgroup_lists_file_path : Union[str, Path]) -> None:
    # Run the DSLM algorithm.
    model = DSLM(input_file_path = str(subgroups_input_file_path),
                 max_sl = maximum_number_of_subgroup_lists,
                 sl_max_size = maximum_number_of_subgroups_per_subgroup_list,
                 beta = float(beta_parameter),
                 maximum_positive_overlap = float(maximum_positive_overlap),
                 maximum_negative_overlap = float(maximum_negative_overlap),
                 output_file_path = str(diverse_topk_subgroup_lists_file_path))
    model.fit(dataset.reset_index(drop=True), (target_attribute_name, target_attribute_value))

def run_combination(arguments : tuple) -> None:
    view, configuration, b, pos_o, neg_o = arguments
    df = utils.get_configuration_dataset(utils.read_mining_view(view), configuration).reset_index(drop=True)
    path = utils.mining_path(view, configuration)
    (path / "lists").mkdir(parents=True, exist_ok=True)
    (path / "log").mkdir(parents=True, exist_ok=True)
    name = str(b) + "_" + str(pos_o) + "_" + str(neg_o)
    TP = int(utils.get_target_mask(df).sum())
    FP = len(df) - TP
    start_time = datetime.now()
    mine_diverse_topk_subgroup_lists(df, utils.candidates_path(view, configuration), utils.TARGET[0], utils.TARGET[1],
                                     utils.MAXIMUM_NUMBER_OF_SUBGROUP_LISTS, utils.MAXIMUM_NUMBER_OF_SUBGROUPS_PER_SUBGROUP_LIST,
                                     b, pos_o, neg_o, path / "lists" / (name + ".txt"))
    end_time = datetime.now()
    subgroup_lists = utils.read_subgroup_lists(path / "lists" / (name + ".txt"))
    measures = utils.subgroup_lists_quality(df, subgroup_lists)
    with open(path / "log" / ("log_" + name + ".txt"), "w") as log_file:
        log_file.write("Configuration: " + configuration + "\n")
        log_file.write("Number of instances: " + str(len(df)) + "\n")
        log_file.write("Number of attributes: " + str(len(df.columns)) + "\n")
        log_file.write("Number of instances in which the target is true: " + str(TP) + "\n")
        log_file.write("Number of instances in which the target is false: " + str(FP) + "\n")
        log_file.write("beta_parameter = " + str(b) + "\n")
        log_file.write("maximum_positive_overlap = " + str(pos_o) + "\n")
        log_file.write("maximum_negative_overlap = " + str(neg_o) + "\n")
        log_file.write("---\n")
        for index, subgroup_list in enumerate(subgroup_lists):
            log_file.write("## Subgroup list " + str(index + 1) + " (" + str(len(subgroup_list)) + " subgroups) ##\n")
            for position, description in enumerate(subgroup_list):
                log_file.write("s" + str(position + 1) + ": " + utils.description_to_str(description) + "\n")
        log_file.write("---\n")
        log_file.write("quality = " + str(measures["quality"]) + "\n")
        log_file.write("coverage = " + str(measures["coverage"]) + "\n")
        log_file.write("diversity = " + str(measures["diversity"]) + "\n")
        log_file.write("number_of_subgroups = " + str(measures["number_of_subgroups"]) + "\n")
        log_file.write("---\n")
        log_file.write("Start time: " + str(start_time) + "\n")
        log_file.write("End time: " + str(end_time) + "\n")

if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        print("Usage: python mine_phenotypes.py <view> <configuration|all> [number_of_processes]")
        exit(1)
    view = sys.argv[1]
    configurations = utils.CONFIGURATIONS if sys.argv[2] == "all" else [sys.argv[2]]
    number_of_processes = int(sys.argv[3]) if len(sys.argv) == 4 else 1
    sys.stdout = utils.Tee(utils.PHENOTYPES_RESULTS_PATH / view / "mining" / "execution_log.txt")
    print("Execution: " + str(datetime.now()) + " (" + " ".join(sys.argv) + ")")
    for configuration in configurations:
        if not utils.candidates_path(view, configuration).exists():
            print("The candidate subgroups of '" + configuration + "' do not exist. Execute Experiments/subgroups/mine_subgroups.py first.")
            exit(1)
    tasks = [(view, configuration, b, pos_o, neg_o) for configuration in configurations
             for b, pos_o, neg_o in product(utils.GRID["beta"], utils.GRID["maximum_positive_overlap"], utils.GRID["maximum_negative_overlap"])]
    print("Number of DSLM executions: " + str(len(tasks)))
    start_time = datetime.now()
    with Pool(number_of_processes) as pool:
        for index, _ in enumerate(pool.imap_unordered(run_combination, tasks)):
            if (index + 1) % 20 == 0:
                print(str(index + 1) + "/" + str(len(tasks)) + " (" + str(datetime.now() - start_time) + ")", flush=True)
    print("Total time: " + str(datetime.now() - start_time))
