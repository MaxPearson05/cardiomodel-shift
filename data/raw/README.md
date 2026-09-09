# Raw data

## Source

UCI Machine Learning Repository: [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

Dataset citation:

Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. (1989). *Heart Disease*. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

The repository describes four patient cohorts collected in Cleveland, Hungary, Switzerland and Long Beach.

## Files used

- `processed.cleveland.data`
- `processed.hungarian.data`

These processed files use the standard 14-variable structure associated with the UCI Heart Disease collection.

## Excluded source file

The unprocessed `cleveland.data` file was excluded because the warning supplied with the dataset identifies it as corrupted. The processed Cleveland file is identified as usable for the 14-variable analysis.

## Outcome definition

The Cleveland cohort records disease-severity values from 0 to 4, while the Hungarian cohort records values 0 and 1. For this project, 0 represents no recorded heart disease and all values greater than 0 represent recorded heart disease.

The resulting outcome represents disease presence rather than future cardiovascular risk.

## Data handling

Raw source files are retained locally without modification and are excluded from version control. Download instructions and provenance are published so the analysis can be reproduced from the original source.