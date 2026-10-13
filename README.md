# Pima Indians Diabetes Classification

Predicting whether a patient has diabetes from eight diagnostic measurements. The project covers data-quality checks, exploratory analysis, a baseline logistic regression, feature selection, hyperparameter tuning and an alternative Random Forest model, with a focus on **recall** (catching diabetic patients) rather than accuracy alone.

> Module 7 assignment: Classification, Model Optimization and Evaluation.

## Project overview

| Step | What I did |
|---|---|
| 1. Analyze the dataset | Quality check (missing values, duplicates, hidden zeros), four plots, with an interpretation for each |
| 2. Baseline classifier | Stratified 80/20 split, logistic regression in a pipeline, accuracy / precision / recall / F1 / AUC |
| 3. Improve the model | Feature selection (`SelectKBest`), hyperparameter tuning (`GridSearchCV`), tuned Random Forest, feature importances, confusion-matrix comparison |

## Dataset

- **Source:** Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases), loaded from a public GitHub copy: <https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset>
- **Size:** 768 patients, 8 predictors, 1 binary target (`Outcome`: 1 = diabetes, 0 = no diabetes)
- **Predictors:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Class balance:** 500 non-diabetic (65.1%) and 268 diabetic (34.9%)

### Data-quality finding: hidden missing values

There are no `NaN` values and no duplicates, but five columns use `0` for missing measurements, which is physiologically impossible:

| Column | Zeros | Share |
|---|---|---|
| Insulin | 374 | 48.7% |
| SkinThickness | 227 | 29.6% |
| BloodPressure | 35 | 4.6% |
| BMI | 11 | 1.4% |
| Glucose | 5 | 0.7% |

These zeros were converted to `NaN` and filled with the **median** inside the modelling pipeline (the median is robust to the strong skew and outliers, especially in Insulin). Imputation is fitted on the training data only.

## Methods

- **Split:** 80/20 stratified (`random_state=42`): 614 training and 154 test patients, with the same share of diabetic patients (about 35%) in both.
- **Pipeline:** median imputation, standard scaling, then the classifier, so no information from the test set leaks into preprocessing.
- **Feature selection:** `SelectKBest` (ANOVA F-test) with k = 3 to 8, scored by 5-fold cross-validated F1 on the training set.
- **Hyperparameter tuning:** `GridSearchCV` (5-fold stratified CV, scoring = F1, training data only).
  - Logistic regression: `C`, `penalty` (l1/l2), `class_weight`
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `class_weight`
- **Metrics:** accuracy, precision, recall, F1 and AUC (computed from predicted probabilities), plus confusion matrices.

## Results (154 test patients, 54 diabetic)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| 1. Baseline logistic regression | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |
| 2. + Feature selection (all 8 kept) | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |
| 3. Tuned logistic regression | 0.721 | 0.585 | 0.704 | 0.639 | 0.811 |
| 4. Tuned Random Forest | 0.727 | 0.591 | 0.722 | 0.650 | 0.818 |

Confusion matrices (rows = actual, columns = predicted; `[[TN, FP], [FN, TP]]`):

| Model | Matrix |
|---|---|
| Baseline | `[[82, 18], [27, 27]]` |
| Tuned logistic regression | `[[73, 27], [16, 38]]` |
| Tuned Random Forest | `[[73, 27], [15, 39]]` |

## Key findings

1. **Accuracy hides the real problem.** The baseline reaches 70.8% accuracy, barely above the 65% obtained by always predicting "no diabetes", and it misses 27 of the 54 diabetic patients (recall 0.50).
2. **Tuning made the biggest difference.** Balanced class weights (selected by cross-validation, with C = 0.1 and an L2 penalty) raised recall from 0.50 to 0.70: 11 more diabetic patients caught, at the cost of about 9 more false alarms. AUC stayed about the same (0.813 to 0.811), so the gain comes mainly from shifting the decision threshold, not from ranking patients better. For screening, this trade-off is reasonable.
3. **Feature selection changed nothing.** Cross-validated F1 was nearly flat for k = 3 to 8 (0.636 to 0.655) and highest with all 8 features, so all were kept. Removing the weakest features (BloodPressure and DiabetesPedigreeFunction, k = 6) gave small, mixed changes within noise: recall 0.50 to 0.54 and F1 0.545 to 0.558, but AUC 0.813 to 0.802.
4. **Random Forest is about the same as tuned logistic regression.** It caught one more diabetic patient (39 vs 38), which is too small a difference to be confident about on 154 test patients. Its cross-validated F1 was slightly higher (0.705 vs 0.679).
5. **Glucose is the strongest predictor** across the box plots, correlations (0.49 with the outcome), ANOVA F-scores (217.8) and Random Forest importances (about 0.36).

## Limitations

- Small test set (154 patients): one patient changes accuracy by about 0.65 percentage points.
- A single train/test split, so results could shift with a different split. Repeated cross-validation would be more reliable.
- Heavy imputation for Insulin (about 49% missing) and SkinThickness (about 30% missing).
- The data covers only women of Pima Indian heritage aged 21 or older, so the results may not generalize to other populations.

## How the notebook differs from the course notebook

The notebook follows the structure of the Module 7 Classification Model Pipeline notebook, with changes that make the evaluation more reliable: a stratified split, imputation and scaling inside a pipeline, hyperparameter tuning fitted on the training data only, AUC computed from predicted probabilities, and a seeded Random Forest. It also adds feature selection, which the assignment requires.

## How to run

1. Open `Module_7_Assignment_Pima_Indians_Diabetes_Classification.ipynb` in Google Colab (File, then Upload notebook) or in Jupyter.
2. Install the dependencies if running locally: `pip install -r requirements.txt`
3. Run all cells from top to bottom. The data is loaded directly from GitHub, so an internet connection is required. The Random Forest grid search can take a couple of minutes.

Results may differ slightly across scikit-learn versions.

## Repository contents

```
.
├── Module_7_Assignment_Pima_Indians_Diabetes_Classification.ipynb   # full analysis with explanations
├── README.md                                                        # this project report
└── requirements.txt                                                 # Python dependencies
```

## References

- Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261-265. <https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/>
- npradaschnor. (n.d.). *Pima-Indians-Diabetes-Dataset* [Data set]. GitHub. <https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset>
- Scikit-learn developers. (n.d.). *scikit-learn documentation* (Pipelines, `SimpleImputer`, `SelectKBest`, `RandomForestClassifier`). <https://scikit-learn.org/stable/>
