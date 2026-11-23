# -*- coding: utf-8 -*-

"""Shared functions for the experiments of the paper "Space-aware antimicrobial resistance phenotyping in hospital environments using diverse top-k subgroup list discovery" (IE 2027).
"""

import sys
from itertools import combinations
from math import sqrt
from pathlib import Path
from re import compile
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.contingency_tables import StratifiedTable
from subgroups.core import Operator, Pattern, Selector, Subgroup
from subgroups.quality_measures import WRAcc

########################
# CONSTANTS OF THE STUDY
########################

ROOT_PATH = Path(__file__).resolve().parent.parent
DATASETS_PATH = ROOT_PATH / "datasets"
EXPERIMENTS_PATH = ROOT_PATH / "Experiments"
# Results of the scripts (candidate subgroups and diverse top-k subgroup lists).
SUBGROUPS_RESULTS_PATH = EXPERIMENTS_PATH / "subgroups" / "results"
PHENOTYPES_RESULTS_PATH = EXPERIMENTS_PATH / "phenotypes" / "results"

# Target: vancomycin-resistant Enterococcus (intermediate isolates are considered resistant).
TARGET = ("culture_susceptibility", "R")

# Spatial-organizational context of the patient when the culture was carried out.
# - 'icu_when_culture': ICU in which the patient resided (or last resided) before the culture; 'NO_ICU' if none.
# - 'service_when_culture': clinical service in charge of the patient when the culture was carried out.
# - 'hours_from_admission_to_first_icu': trajectory until the first ICU ('NO_ICU_BEFORE_CULTURE' if none).
UNIT_ATTRIBUTE = "icu_when_culture"
CONTEXT_ATTRIBUTES = ["icu_when_culture", "service_when_culture", "hours_from_admission_to_first_icu"]
UNITS = ["MICU", "SICU", "TSICU", "CCU", "CSRU", "NO_ICU"]

# Configurations (modes) of the space-aware phenotyping.
# - 'without_context': global mining, only patient attributes (baseline).
# - 'with_context': global mining, patient and context attributes.
# - 'unit_<U>': mining only with the instances of unit U and only patient attributes.
CONFIGURATIONS = ["without_context", "with_context"] + ["unit_" + u for u in UNITS]

# Hyperparameter grid of DSLM.
GRID = {
    "beta": [0.0, 0.15, 0.3, 0.5, 0.75, 1.0],
    "maximum_positive_overlap": [0.1, 0.3, 0.6],
    "maximum_negative_overlap": [0.1, 0.3, 0.6],
}
MAXIMUM_NUMBER_OF_SUBGROUP_LISTS = 3
MAXIMUM_NUMBER_OF_SUBGROUPS_PER_SUBGROUP_LIST = 10

# Candidate subgroups.
CANDIDATES_MAXIMUM_DEPTH = 3
CANDIDATES_MINIMUM_WRACC = 0.01
CANDIDATES_MINIMUM_SUPPORT = 10

RANDOM_SEED = 100

#######
# PATHS
#######

class Tee:
    """Write the standard output in the console and in a log file (execution logs of the scripts)."""

    def __init__(self, file_path : Union[str, Path]):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(file_path, "a")
        self.stdout = sys.stdout

    def write(self, text : str):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

def candidates_path(view : str, configuration : str) -> Path:
    """Candidate subgroups of a configuration (Experiments/subgroups/results/<view>/<configuration>.txt)."""
    return SUBGROUPS_RESULTS_PATH / view / (configuration + ".txt")

def mining_path(view : str, configuration : str) -> Path:
    """Diverse top-k subgroup lists of the grid (Experiments/phenotypes/results/<view>/mining/<configuration>/)."""
    return PHENOTYPES_RESULTS_PATH / view / "mining" / configuration

def best_models_path(view : str) -> Path:
    return PHENOTYPES_RESULTS_PATH / view / "best_models.csv"

def best_subgroup_lists_path(view : str, best_models : pd.DataFrame, configuration : str) -> Path:
    """File with the subgroup lists of the best hyperparameters of a configuration ('best_models' indexed by configuration)."""
    row = best_models.loc[configuration]
    name = str(row["beta"]) + "_" + str(row["maximum_positive_overlap"]) + "_" + str(row["maximum_negative_overlap"])
    return mining_path(view, configuration) / "lists" / (name + ".txt")

def cross_validation_path(view : str) -> Path:
    return PHENOTYPES_RESULTS_PATH / view / "cross_validation"

def bootstrap_path(view : str) -> Path:
    return PHENOTYPES_RESULTS_PATH / view / "bootstrap"

def results_path(experiment : str, kind : str) -> Path:
    """Folder 'Experiments/<experiment>/results/<kind>' (kind: 'tables', 'figures', 'mining'...). It is created if needed."""
    path = EXPERIMENTS_PATH / experiment / "results" / kind
    path.mkdir(parents=True, exist_ok=True)
    return path

##########
# DATASETS
##########

def read_mining_view(view : str = "clinical") -> pd.DataFrame:
    """Read a mining view ('clinical' or 'all-specimens'). All the attributes are nominal."""
    return pd.read_csv(DATASETS_PATH / ("mining-view-" + view + ".csv"), dtype=str)

def read_auxiliary_data(view : str = "clinical") -> pd.DataFrame:
    """Read the auxiliary data (identifiers and flags not used for mining), aligned by row with the mining view."""
    return pd.read_csv(DATASETS_PATH / ("auxiliary-data-" + view + ".csv"))

def get_configuration_dataset(dataset : pd.DataFrame, configuration : str) -> pd.DataFrame:
    """Return the dataset of a configuration. The index of the original dataset is kept."""
    if configuration == "without_context":
        return dataset.drop(columns=CONTEXT_ATTRIBUTES)
    elif configuration == "with_context":
        return dataset.copy()
    elif configuration.startswith("unit_"):
        unit = configuration[len("unit_"):]
        return dataset[dataset[UNIT_ATTRIBUTE] == unit].drop(columns=CONTEXT_ATTRIBUTES)
    raise ValueError("Unknown configuration: " + configuration)

def get_target_mask(dataset : pd.DataFrame, target : tuple = TARGET) -> np.ndarray:
    return (dataset[target[0]] == target[1]).to_numpy()

##############
# DESCRIPTIONS
##############

# A description is a tuple of selectors (attribute, value), sorted by attribute.

def description_to_str(description : tuple) -> str:
    return "[" + ", ".join(attribute + " = '" + value + "'" for attribute, value in description) + "]"

def description_to_subgroup(description : tuple, target : tuple = TARGET) -> Subgroup:
    """Subgroup object of the subgroups library."""
    return Subgroup(Pattern([Selector(attribute, Operator.EQUAL, value) for attribute, value in description]),
                    Selector(target[0], Operator.EQUAL, target[1]))

def subgroup_to_str(description : tuple, target : tuple = TARGET) -> str:
    """Format used by the subgroups library: "Description: [...], Target: ..."."""
    return str(description_to_subgroup(description, target))

_regex_selector = compile("(?P<attribute>[^,\\[\\]]+?) = '(?P<value>[^']*)'")

def str_to_description(string : str) -> tuple:
    return tuple(sorted((m.group("attribute").strip(), m.group("value")) for m in _regex_selector.finditer(string)))

def covers(dataset : pd.DataFrame, description : tuple) -> np.ndarray:
    """Boolean mask of the instances covered by a description."""
    mask = np.ones(len(dataset), dtype=bool)
    for attribute, value in description:
        mask &= (dataset[attribute] == value).to_numpy()
    return mask

def split_description(description : tuple, context_attributes : list = CONTEXT_ATTRIBUTES) -> (tuple, tuple):
    """Split a description into its patient part and its context part."""
    patient = tuple(s for s in description if s[0] not in context_attributes)
    context = tuple(s for s in description if s[0] in context_attributes)
    return patient, context

def wracc(dataset : pd.DataFrame, description : tuple, target : tuple = TARGET) -> float:
    """WRAcc of a description, computed with the WRAcc quality measure of the subgroups library."""
    y = get_target_mask(dataset, target)
    mask = covers(dataset, description)
    if mask.sum() == 0:
        return 0.0
    return WRAcc().compute({WRAcc.TRUE_POSITIVES : int((mask & y).sum()), WRAcc.FALSE_POSITIVES : int((mask & ~y).sum()),
                            WRAcc.TRUE_POPULATION : int(y.sum()), WRAcc.FALSE_POPULATION : int((~y).sum())})

####################
# CANDIDATE SUBGROUPS
####################

def generate_candidate_subgroups(dataset : pd.DataFrame,
                                 target : tuple,
                                 file_path : Union[str, Path],
                                 maximum_depth : int = CANDIDATES_MAXIMUM_DEPTH,
                                 minimum_wracc : float = CANDIDATES_MINIMUM_WRACC,
                                 minimum_support : int = CANDIDATES_MINIMUM_SUPPORT) -> int:
    """Exhaustive depth-limited subgroup discovery (WRAcc with optimistic estimate pruning), equivalent to the VLSD
    algorithm of the subgroups library with a maximum description size (VLSD has no such parameter and, without it, it
    did not finish with this mining view). The quality of each subgroup is computed with the WRAcc quality measure of the
    subgroups library. The optimistic estimate is (tp / N) * (1 - TP / N): the WRAcc of a refinement that keeps all its
    positive instances and none of its negative instances. The candidates are written with the input format of DSLM:
    '<subgroup> ; <bitset of positive instances> ; <bitset of negative instances>'.
    It returns the number of candidates written.
    """
    y = get_target_mask(dataset, target)
    N = len(dataset)
    p0 = y.mean()
    TP, FP = int(y.sum()), int(N - y.sum())
    quality_measure = WRAcc()
    selectors = []
    for attribute in sorted(dataset.columns.drop(target[0])):
        for value in sorted(dataset[attribute].unique()):
            mask = (dataset[attribute] == value).to_numpy()
            if mask.sum() >= minimum_support:
                selectors.append((attribute, value, mask))
    candidates = []
    def search(start : int, used_attributes : frozenset, description : list, mask : Union[np.ndarray, None], depth : int):
        for i in range(start, len(selectors)):
            attribute, value, selector_mask = selectors[i]
            if attribute in used_attributes:
                continue
            new_mask = selector_mask if mask is None else (mask & selector_mask)
            n = new_mask.sum()
            if n < minimum_support:
                continue
            tp = (new_mask & y).sum()
            # Optimistic estimate of WRAcc for any refinement: all the positive instances are kept.
            if (tp / N) * (1 - p0) < minimum_wracc:
                continue
            new_description = description + [(attribute, value)]
            if quality_measure.compute({WRAcc.TRUE_POSITIVES : int(tp), WRAcc.FALSE_POSITIVES : int(n - tp), WRAcc.TRUE_POPULATION : TP, WRAcc.FALSE_POPULATION : FP}) >= minimum_wracc:
                candidates.append((tuple(new_description), new_mask))
            if depth < maximum_depth:
                search(i + 1, used_attributes | {attribute}, new_description, new_mask, depth + 1)
    search(0, frozenset(), [], None, 1)
    with open(file_path, "w") as output_file:
        for description, mask in candidates:
            positives = "".join("1" if b else "0" for b in (mask & y))
            negatives = "".join("1" if b else "0" for b in (mask & ~y))
            output_file.write(subgroup_to_str(description, target) + " ; " + positives + " ; " + negatives + "\n")
    return len(candidates)

########################
# DIVERSE SUBGROUP LISTS
########################

# The diverse top-k subgroup lists are mined with the DSLM algorithm of the subgroups library
# (from subgroups.algorithms import DSLM) in Experiments/phenotypes. These functions read its output file.

_regex_sl_header = compile("^## Subgroup list (?P<n_subgroups>\\(.*\\)) ##$")
_regex_subgroup = compile("^s(?P<number>[0-9]+): Description: (?P<description>.+), Target: (?P<target>.+)$")

def read_subgroup_lists(file_path : Union[str, Path]) -> list:
    """Read the output file of DSLM. It returns a list of subgroup lists, each one a list of descriptions."""
    subgroup_lists = []
    with open(file_path, "r") as input_file:
        for line in input_file:
            line = line.rstrip("\n")
            if _regex_sl_header.fullmatch(line):
                subgroup_lists.append([])
            else:
                match_object = _regex_subgroup.fullmatch(line)
                if match_object:
                    subgroup = Subgroup.generate_from_str("Description: " + match_object.group("description") + ", Target: " + match_object.group("target"))
                    subgroup_lists[-1].append(tuple((selector.attribute_name, str(selector.value)) for selector in subgroup.description))
    return subgroup_lists

def subgroup_lists_quality(dataset : pd.DataFrame, subgroup_lists : list, target : tuple = TARGET) -> dict:
    """Quality measures of a set of diverse subgroup lists.
    - quality: mean over the lists of the sum of the WRAcc of their subgroups.
    - coverage: proportion of instances covered by at least one subgroup of at least one list.
    - diversity: 1 - mean pairwise Jaccard similarity between the sets of selectors of the lists.
    - context_proportion: proportion of subgroups with at least one context selector.
    """
    covered = np.zeros(len(dataset), dtype=bool)
    qualities, selector_sets, n_subgroups, n_context = [], [], 0, 0
    for subgroup_list in subgroup_lists:
        qualities.append(sum(wracc(dataset, d, target) for d in subgroup_list))
        selector_sets.append(set(s for d in subgroup_list for s in d))
        for d in subgroup_list:
            covered |= covers(dataset, d)
            n_subgroups += 1
            n_context += int(len(split_description(d)[1]) > 0)
    similarities = [len(a & b) / len(a | b) for a, b in combinations(selector_sets, 2) if len(a | b) > 0]
    return {
        "number_of_lists": len(subgroup_lists),
        "number_of_subgroups": n_subgroups,
        "mean_subgroups_per_list": n_subgroups / max(len(subgroup_lists), 1),
        "quality": float(np.mean(qualities)) if qualities else 0.0,
        "coverage": covered.mean(),
        "diversity": 1 - float(np.mean(similarities)) if similarities else float("nan"),
        "context_proportion": n_context / max(n_subgroups, 1),
    }

#########################################
# SUBGROUP LISTS USED AS CLASSIFIERS (E1)
#########################################

class SubgroupListClassifier:
    """Probabilistic classifier from one or several subgroup lists. In each list, an instance is assigned to the first
    subgroup that covers it (or to the default rule) and its score is the (Laplace-smoothed) proportion of positive
    instances of that subgroup in the training data, considering its position in the list. With several lists, the
    scores are averaged.
    """

    def __init__(self, subgroup_lists : list, target : tuple = TARGET):
        self.subgroup_lists = subgroup_lists
        self.target = target
        self.probabilities_ = []

    def fit(self, dataset : pd.DataFrame):
        y = get_target_mask(dataset, self.target)
        self.probabilities_ = []
        for subgroup_list in self.subgroup_lists:
            remaining = np.ones(len(dataset), dtype=bool)
            probabilities = []
            for description in subgroup_list:
                covered = covers(dataset, description) & remaining
                probabilities.append(((covered & y).sum() + 1) / (covered.sum() + 2))
                remaining &= ~covered
            probabilities.append(((remaining & y).sum() + 1) / (remaining.sum() + 2))
            self.probabilities_.append(probabilities)
        return self

    def predict_proba(self, dataset : pd.DataFrame) -> np.ndarray:
        if len(self.subgroup_lists) == 0:
            return np.full(len(dataset), np.nan)
        scores = np.zeros(len(dataset))
        for subgroup_list, probabilities in zip(self.subgroup_lists, self.probabilities_):
            score = np.full(len(dataset), probabilities[-1])
            assigned = np.zeros(len(dataset), dtype=bool)
            for description, probability in zip(subgroup_list, probabilities[:-1]):
                covered = covers(dataset, description) & ~assigned
                score[covered] = probability
                assigned |= covered
            scores += score
        return scores / len(self.subgroup_lists)

##########################################
# CONTEXT DEPENDENCE OF A PHENOTYPE (E2)
##########################################

def context_dependence(dataset : pd.DataFrame,
                       description : tuple,
                       target : tuple = TARGET,
                       unit_attribute : str = UNIT_ATTRIBUTE,
                       units : list = UNITS,
                       minimum_covered : int = 10) -> dict:
    """Context dependence of a phenotype (subgroup).

    The context selectors of the description are removed (patient part P). For each unit u in which P covers at least
    'minimum_covered' instances, the association between P and resistance is measured with the odds ratio OR_u (2x2
    table: covered/not covered x R/S, inside u; Haldane correction when a cell is 0). Then:
    - pooled_or: Mantel-Haenszel odds ratio across units (p_pooled: test of OR = 1).
    - p_heterogeneity: Breslow-Day test of homogeneity of OR_u across units.
    - i2: proportion of the variability of log(OR_u) due to heterogeneity (Cochran's Q, Woolf weights), in [0, 1].
      This is the degree of context dependence: 0 = same effect in every unit; close to 1 = effect depends on unit.
    - active_units: units where the lower bound of the 95% CI of OR_u is greater than 1.
    The final label (TRANSVERSAL / SPACE_SPECIFIC / ...) is assigned by 'label_context_dependence' after correcting
    the p-values for multiple testing.
    """
    patient_description, context_description = split_description(description)
    y = get_target_mask(dataset, target)
    result = {"description": description_to_str(description),
              "patient_description": description_to_str(patient_description),
              "context_description": description_to_str(context_description)}
    # Gain of the context part (only for descriptions with context selectors).
    if len(context_description) > 0:
        result["wracc"] = wracc(dataset, description, target)
        result["wracc_without_context"] = wracc(dataset, patient_description, target) if patient_description else 0.0
        result["context_gain"] = result["wracc"] - result["wracc_without_context"]
        covered, covered_patient = covers(dataset, description), covers(dataset, patient_description)
        result["precision"] = y[covered].mean() if covered.sum() else np.nan
        result["precision_without_context"] = y[covered_patient].mean() if covered_patient.sum() else np.nan
        result["precision_gain"] = result["precision"] - result["precision_without_context"]
    if len(patient_description) == 0:
        result.update({"units_observed": 0, "label_raw": "CONTEXT_ONLY"})
        return result
    mask = covers(dataset, patient_description)
    tables, log_ors, weights = [], [], []
    for unit in units:
        in_unit = (dataset[unit_attribute] == unit).to_numpy()
        a = int((mask & y & in_unit).sum()); b = int((mask & ~y & in_unit).sum())
        c = int((~mask & y & in_unit).sum()); d = int((~mask & ~y & in_unit).sum())
        result["n_" + unit] = a + b
        if (a + b) < minimum_covered or (c + d) == 0:
            result["or_" + unit] = np.nan
            continue
        table = np.array([[a, b], [c, d]], dtype=float)
        if (table == 0).any():
            table = table + 0.5
        log_or = np.log(table[0, 0] * table[1, 1] / (table[0, 1] * table[1, 0]))
        se = sqrt((1 / table).sum())
        result["or_" + unit] = float(np.exp(log_or))
        result["or_low_" + unit] = float(np.exp(log_or - 1.96 * se))
        result["or_high_" + unit] = float(np.exp(log_or + 1.96 * se))
        result["precision_" + unit] = a / (a + b)
        tables.append(table); log_ors.append(log_or); weights.append(1 / se ** 2)
    k = len(tables)
    result["units_observed"] = k
    result["active_units"] = ", ".join(u for u in units if result.get("or_low_" + u, 0) > 1)
    result["number_of_active_units"] = sum(1 for u in units if result.get("or_low_" + u, 0) > 1)
    if k == 0:
        result["label_raw"] = "NOT_OBSERVED"
        return result
    if k == 1:
        result["label_raw"] = "OBSERVED_IN_ONE_UNIT"
        return result
    log_ors, weights = np.array(log_ors), np.array(weights)
    fixed_effect = (weights * log_ors).sum() / weights.sum()
    q = float((weights * (log_ors - fixed_effect) ** 2).sum())
    result["q"] = q
    result["i2"] = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    result["p_q"] = float(chi2.sf(q, k - 1))
    stratified_table = StratifiedTable(tables)
    result["pooled_or"] = float(stratified_table.oddsratio_pooled)
    low, high = stratified_table.oddsratio_pooled_confint()
    result["pooled_or_low"], result["pooled_or_high"] = float(low), float(high)
    result["p_pooled"] = float(stratified_table.test_null_odds().pvalue)
    result["p_heterogeneity"] = float(stratified_table.test_equal_odds().pvalue)
    result["label_raw"] = "TESTED"
    return result

def label_context_dependence(results : pd.DataFrame, alpha : float = 0.05) -> pd.DataFrame:
    """Assign the final label of each phenotype after a Benjamini-Hochberg correction of the p-values.
    - SPACE_SPECIFIC: significant heterogeneity of the effect across units and the effect is not significant
      (OR > 1) in all the units where the phenotype is observed (it is active only in some units).
    - SPACE_MODULATED: significant heterogeneity, but the effect is significant (OR > 1) in all the units where the
      phenotype is observed (same direction everywhere, different magnitude).
    - TRANSVERSAL: no significant heterogeneity and significant pooled effect (OR > 1).
    - PROTECTIVE_TRANSVERSAL: no significant heterogeneity and significant pooled effect (OR < 1).
    - NOT_ASSOCIATED: neither significant heterogeneity nor significant pooled effect.
    - CONTEXT_DRIVEN: the description has context selectors and its patient part is NOT_ASSOCIATED or
      PROTECTIVE_TRANSVERSAL (the risk comes from the context, not from the patient profile).
    - OBSERVED_IN_ONE_UNIT / CONTEXT_ONLY / NOT_OBSERVED: the measure cannot be computed.
    """
    from statsmodels.stats.multitest import multipletests
    results = results.copy()
    tested = results["label_raw"] == "TESTED"
    results["p_heterogeneity_bh"] = np.nan
    results["p_pooled_bh"] = np.nan
    if tested.sum() > 0:
        results.loc[tested, "p_heterogeneity_bh"] = multipletests(results.loc[tested, "p_heterogeneity"], method="fdr_bh")[1]
        results.loc[tested, "p_pooled_bh"] = multipletests(results.loc[tested, "p_pooled"], method="fdr_bh")[1]
    def label(row):
        if row["label_raw"] != "TESTED":
            return row["label_raw"]
        if row["p_heterogeneity_bh"] < alpha:
            return "SPACE_MODULATED" if row["number_of_active_units"] == row["units_observed"] else "SPACE_SPECIFIC"
        if row["p_pooled_bh"] < alpha:
            return "TRANSVERSAL" if row["pooled_or"] > 1 else "PROTECTIVE_TRANSVERSAL"
        return "NOT_ASSOCIATED"
    results["label"] = results.apply(label, axis=1)
    if "context_description" in results.columns:
        context_driven = (results["context_description"] != "[]") & results["label"].isin(["NOT_ASSOCIATED", "PROTECTIVE_TRANSVERSAL"])
        results.loc[context_driven, "label"] = "CONTEXT_DRIVEN"
    return results

##########
# STATISTICS
##########

def wilson_interval(successes : int, n : int, z : float = 1.96) -> (float, float):
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half
