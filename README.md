# Space-aware antimicrobial resistance phenotyping in hospital environments using diverse top-k subgroup list discovery

This repository contains all the needed resources in order to reproduce the research from the paper "Space-aware antimicrobial resistance phenotyping in hospital environments using diverse top-k subgroup list discovery" (IE 2027). It starts from the output of the SQL query in [Initial-query/query.txt](Initial-query/query.txt) over MIMIC-III v1.4.

The MIMIC-III data cannot be distributed: MIMIC-III v1.4 is only available through [PhysioNet](https://physionet.org/content/mimiciii/1.4/) to credentialed users who sign its data use agreement. Put the output of the query in `datasets/mimic-iii-dataset.csv` (the `COPY ... TO` statement at the end of the query writes it to `C:\Users\Public\Documents\mimic-iii-dataset.csv`, so change that path or move the file).

For the same reason, the input/output files that contain data at the level of the instances are not included in this repository (see [.gitignore](.gitignore)), although the code generates them.

## Structure

```
Initial-query/query.txt                      SQL query over MIMIC-III v1.4
datasets/                                    mimic-iii-dataset.csv (input) and the mining views (output of the preprocessing)
Preprocessing/notebook.ipynb                 preprocessing: mining views
Experiments/
  utils.py                                   shared functions and constants of the study
  subgroups/
    mine_subgroups.py                        candidate subgroups (WRAcc, subgroups library)
    results/clinical/
      <configuration>.txt                    candidate subgroups of each configuration (input of DSLM)
      execution_log.txt
  phenotypes/
    mine_phenotypes.py                       DSLM (subgroups library) with the hyperparameter grid
    select_best_model.ipynb                  best hyperparameters of each configuration
    cross_validation.py                      10-fold cross-validation grouped by patient
    bootstrap.py                             bootstrap replicates
    results/clinical/
      mining/<configuration>/lists/          DSLM output of each combination of hyperparameters
      mining/<configuration>/log/            log of each combination (quality, coverage, diversity)
      mining/execution_log.txt
      grid_results.csv                       quality, coverage and diversity of all the combinations
      best_models.csv                        best hyperparameters of each configuration
      figures/                               hyperparameter figures
      cross_validation/                      folds.csv, fold_<f>/ (candidates and DSLM output), predictions.csv, execution_log.txt
      bootstrap/<configuration>/             candidates_<b>.txt and b_<b>.txt (DSLM output) of each replicate; execution_log.txt
  E0-motivation/                             notebook.ipynb + results/{tables, figures}
  E1-context-and-quality/                    notebook.ipynb + results/{tables, figures}
  E2-context-dependence/                     notebook.ipynb + results/{tables, figures}
  E3-baselines/                              notebook.ipynb + results/{tables, figures}
  E4-robustness/                             notebook.ipynb + results/{tables, mining}
```

## 1. Initial query

The first step consists of executing the SQL query in [Initial-query/query.txt](Initial-query/query.txt) over the MIMIC-III v1.4 database, stored in a PostgreSQL database. It produces `datasets/mimic-iii-dataset.csv`.

## 2. Preprocessing

[Preprocessing/notebook.ipynb](Preprocessing/notebook.ipynb) generates the mining views:

- `datasets/mining-view-clinical.csv`: main analysis (clinical cultures).
- `datasets/mining-view-all-specimens.csv`: sensitivity analysis (clinical cultures and swabs).
- `datasets/auxiliary-data-*.csv`: identifiers and flags aligned by row with the mining views (not used for mining).

## 3. Experiments

The shared functions are in [Experiments/utils.py](Experiments/utils.py): constants of the study (target, context attributes, configurations, grid), paths of the results, candidate subgroup generation, quality measures (WRAcc of the subgroups library), reading of the DSLM output files, subgroup lists as classifiers and the context dependence measure.

Context attributes (spatial-organizational context when the culture was carried out): `icu_when_culture`, `service_when_culture` and `hours_from_admission_to_first_icu`. Configurations (modes):

- `without_context`: global mining with the patient attributes (baseline).
- `with_context`: global mining with the patient and the context attributes.
- `unit_<U>`: mining with the instances of the unit U and the patient attributes (U = MICU, SICU, TSICU, CCU, CSRU, NO_ICU).

Execute the files in the following order:

1. [Experiments/E0-motivation/notebook.ipynb](Experiments/E0-motivation/notebook.ipynb): resistance rate by unit (Wilson CIs, chi-square, Cramer's V), effect of the swabs and adjusted analysis (logistic regression, likelihood-ratio test).
2. In `Experiments/subgroups`: `python mine_subgroups.py clinical all` (candidate subgroups).
3. In `Experiments/phenotypes`:
   1. `python mine_phenotypes.py clinical all <number_of_processes>`: DSLM with the hyperparameter grid for each configuration (about 22 minutes with 4 processes).
   2. [select_best_model.ipynb](Experiments/phenotypes/select_best_model.ipynb): best hyperparameters of each configuration (highest quality, and highest coverage in case of a tie).
   3. `python cross_validation.py clinical <number_of_processes>`: 10-fold cross-validation grouped by patient (E1 and E3; about 10 minutes).
   4. `python bootstrap.py clinical global 100 <number_of_processes>`: bootstrap replicates (E4; about 1 hour).
4. [Experiments/E1-context-and-quality/notebook.ipynb](Experiments/E1-context-and-quality/notebook.ipynb) (RQ1): quality, coverage and diversity of the phenotypes, and predictive capacity (AUC-ROC, AUC-PR) with and without context and per unit.
5. [Experiments/E2-context-dependence/notebook.ipynb](Experiments/E2-context-dependence/notebook.ipynb) (RQ2): context dependence measure (definition in the notebook), labels (space-specific / transversal), heatmap units x phenotypes and phenotypic antibiogram by unit.
6. [Experiments/E3-baselines/notebook.ipynb](Experiments/E3-baselines/notebook.ipynb): antibiogram by unit, top-k subgroups, decision tree, logistic regression and gradient boosting + SHAP.
7. [Experiments/E4-robustness/notebook.ipynb](Experiments/E4-robustness/notebook.ipynb) (RQ3): Fisher tests with Holm and Benjamini-Hochberg, bootstrap stability of the phenotypes and of the context dependence labels, and sensitivity analyses (it runs DSLM again for each analysis; about 15 minutes).
8. [Experiments/E5-clinical-validation/notebook.ipynb](Experiments/E5-clinical-validation/notebook.ipynb): questionnaire for the clinical validation and analysis of the answers.

Each script writes its standard output in an `execution_log.txt` file next to its results. Each notebook writes its tables (CSV) in `results/tables/` and its figures (PDF) in `results/figures/`.

## Requirements

See [requirements.txt](requirements.txt).
