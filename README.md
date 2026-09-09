# CardioModel Shift

## Cross-Population Heart Disease Classification Audit

CardioModel Shift investigates whether machine-learning models trained using one patient population maintain their performance when evaluated using a different patient population.

The project uses cohorts from the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), which contains data collected in Cleveland, Hungary, Switzerland and Long Beach. The analysis will begin by assessing whether the cohorts have sufficiently compatible features and outcome definitions for a fair comparison.

## Research question

How much does the performance of a heart-disease classification model change when it is tested on a patient cohort different from the one used for training?

## Planned analysis

- Audit data quality and cohort compatibility
- Harmonise shared patient variables
- Train an interpretable Logistic Regression model
- Compare it with a Random Forest model
- Evaluate within-cohort and cross-cohort performance
- Compare ROC-AUC, average precision, sensitivity and specificity
- Assess probability calibration
- Explain model behaviour using feature importance
- Visualise performance changes across cohorts

## Project status

In development.

## Tools

Python, pandas, scikit-learn, Matplotlib, Seaborn and Jupyter.

## Responsible-use statement

This is an educational analysis of public historical data. It is not a clinical diagnostic system and should not be used to make decisions about patient care.