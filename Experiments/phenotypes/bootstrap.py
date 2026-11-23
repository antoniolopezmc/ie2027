"""Bootstrap replicates of the phenotype mining (E4, stability).

Usage:
    python bootstrap.py <view> <configuration|global|all> <number_of_replicates> [number_of_processes]

'global' = without_context and with_context. It requires results/<view>/best_models.csv. In each replicate, the
instances of the configuration dataset are sampled with replacement (same size), and the candidate subgroups and the
subgroup lists (DSLM algorithm of the subgroups library) are mined with the best hyperparameters of the configuration.
Outputs (in results/<view>/bootstrap/):
    - <configuration>/candidates_<replicate>.txt: candidate subgroups of the replicate (input of DSLM).
    - <configuration>/b_<replicate>.txt: diverse top-k subgroup lists of the replicate (DSLM output).
    - execution_log.txt: standard output of the execution.
"""

import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils
from mine_phenotypes import mine_diverse_topk_subgroup_lists

def bootstrap_path(view : str, configuration : str) -> Path:
    return utils.bootstrap_path(view) / configuration

def run_replicate(arguments : tuple) -> None:
    view, configuration, replicate, b, pos_o, neg_o = arguments
    df = utils.get_configuration_dataset(utils.read_mining_view(view), configuration).reset_index(drop=True)
    rng = np.random.default_rng(utils.RANDOM_SEED + replicate)
    sample = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
    path = bootstrap_path(view, configuration)
    candidates_path = path / ("candidates_" + str(replicate) + ".txt")
    utils.generate_candidate_subgroups(sample, utils.TARGET, candidates_path)
    mine_diverse_topk_subgroup_lists(sample, candidates_path, utils.TARGET[0], utils.TARGET[1],
                                     utils.MAXIMUM_NUMBER_OF_SUBGROUP_LISTS, utils.MAXIMUM_NUMBER_OF_SUBGROUPS_PER_SUBGROUP_LIST,
                                     b, pos_o, neg_o, path / ("b_" + str(replicate) + ".txt"))

if __name__ == '__main__':
    if len(sys.argv) not in (4, 5):
        print("Usage: python bootstrap.py <view> <configuration|global|all> <number_of_replicates> [number_of_processes]")
        exit(1)
    view = sys.argv[1]
    if sys.argv[2] == "all":
        configurations = utils.CONFIGURATIONS
    elif sys.argv[2] == "global":
        configurations = ["without_context", "with_context"]
    else:
        configurations = [sys.argv[2]]
    number_of_replicates = int(sys.argv[3])
    number_of_processes = int(sys.argv[4]) if len(sys.argv) == 5 else 1
    best_models = pd.read_csv(utils.best_models_path(view)).set_index("configuration")
    sys.stdout = utils.Tee(utils.bootstrap_path(view) / "execution_log.txt")
    print("Execution: " + str(datetime.now()) + " (" + " ".join(sys.argv) + ")")
    tasks = []
    for configuration in configurations:
        bootstrap_path(view, configuration).mkdir(parents=True, exist_ok=True)
        for replicate in range(number_of_replicates):
            tasks.append((view, configuration, replicate, best_models.loc[configuration, "beta"],
                          best_models.loc[configuration, "maximum_positive_overlap"], best_models.loc[configuration, "maximum_negative_overlap"]))
    print("Number of replicates to mine: " + str(len(tasks)))
    start_time = datetime.now()
    with Pool(number_of_processes) as pool:
        for index, _ in enumerate(pool.imap_unordered(run_replicate, tasks)):
            if (index + 1) % 10 == 0:
                print(str(index + 1) + "/" + str(len(tasks)) + " (" + str(datetime.now() - start_time) + ")", flush=True)
    print("Total time: " + str(datetime.now() - start_time))
