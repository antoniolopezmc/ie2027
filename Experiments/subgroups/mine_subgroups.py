"""Mine the candidate subgroups of each configuration (input of the DSLM algorithm).

Usage:
    python mine_subgroups.py <view> <configuration|all>

Example:
    python mine_subgroups.py clinical all

The candidate subgroups are all the subgroups with WRAcc >= 0.01, support >= 10 and at most 3 selectors (see
utils.generate_candidate_subgroups). The VLSD algorithm of the subgroups library has no parameter to limit the size
of the descriptions and, without it, it did not finish with this mining view.
The quality of the subgroups is computed with the WRAcc quality measure of the subgroups library.

Outputs: results/<view>/execution_log.txt and results/<view>/<configuration>.txt, with the input format of DSLM:
    Description: [...], Target: culture_susceptibility = 'R' ; <bitset of positive instances> ; <bitset of negative instances>
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python mine_subgroups.py <view> <configuration|all>")
        exit(1)
    view = sys.argv[1]
    configurations = utils.CONFIGURATIONS if sys.argv[2] == "all" else [sys.argv[2]]
    sys.stdout = utils.Tee(utils.SUBGROUPS_RESULTS_PATH / view / "execution_log.txt")
    print("Execution: " + str(datetime.now()) + " (" + " ".join(sys.argv) + ")")
    for configuration in configurations:
        df = utils.get_configuration_dataset(utils.read_mining_view(view), configuration).reset_index(drop=True)
        TP = int(utils.get_target_mask(df).sum())
        print("#### " + configuration + " ####")
        print("Number of instances: " + str(len(df)))
        print("Number of attributes: " + str(len(df.columns)))
        print("INFO: number of instances in which the target is true: " + str(TP))
        print("INFO: number of instances in which the target is false: " + str(len(df) - TP))
        start_time = datetime.now()
        utils.candidates_path(view, configuration).parent.mkdir(parents=True, exist_ok=True)
        number_of_candidates = utils.generate_candidate_subgroups(df, utils.TARGET, utils.candidates_path(view, configuration))
        print("Number of subgroups mined: " + str(number_of_candidates))
        print("Time: " + str(datetime.now() - start_time))
