# Predicting H5Nx Hotspots in the USA Using Species Distribution Modeling

This repository contains the code, data-processing workflows, and documentation used to develop and evaluate predictive models of H5Nx avian influenza risk in the United States (U.S.).

The study integrates climatic, wildlife, and poultry-related information to characterize the spatial distribution of H5Nx outbreak risk and generate predictive risk maps across the contiguous United States.

## Overview

The modeling framework combines two predictive modeling approaches:

- **Logistic Regression (LR)**
- **Random Forest (RF)**

To address class imbalance between locations with and without reported H5Nx infections, two data-augmentation approaches were investigated:

- **Stochastic Variational Inference (SVI)**
- **Empirical Distribution (ED)**

The combination of the two predictive models and two data-augmentation approaches resulted in four model configurations:

1. **LR + SVI**
2. **LR + ED**
3. **RF + SVI**
4. **RF + ED**

The models were used to estimate spatial H5Nx risk and generate predictive risk maps for the United States.

---

## Data

Three primary data sources were used in the analysis.

### 1. Avian Influenza Occurrence Data

Avian influenza occurrence data were obtained from the **Food and Agriculture Organization of the United Nations (FAO) EMPRES-i** platform:

https://empres-i.apps.fao.org/general

The dataset contains reported avian influenza occurrences in wild and domestic animals, including geographic coordinates and observation dates. The data used in this study cover the period from **January 1, 2004, through December 31, 2024**.

H5Nx outbreak observations were used as the presence observations for model development.

---

### 2. U.S. Poultry Facility Data

Locations of poultry facilities in the United States were obtained from the **United States Department of Agriculture (USDA) Food Safety and Inspection Service (FSIS) Meat, Poultry and Egg Product Inspection Directory**:

https://www.fsis.usda.gov/inspection/establishments/meat-poultry-and-egg-product-inspection-directory

The dataset used in this study represents poultry facility locations available as of **2024**.

Poultry-related spatial predictors were derived from these locations, including:

- Distance to the nearest poultry facility
- Number of nearby poultry facilities

---

### 3. Climate Data

Historical bioclimatic data were obtained from **WorldClim version 2.1**:

https://www.worldclim.org/data/worldclim21.html

The historical climate dataset represents the **1970–2000** baseline period and contains 19 bioclimatic variables.

Following data preprocessing and assessment of multicollinearity, **10 predictors** were retained for model development:

1. Mean Diurnal Range (BIO2)
2. Annual Temperature Range (BIO7)
3. Mean Temperature of Wettest Quarter (BIO8)
4. Precipitation Seasonality (BIO15)
5. Precipitation of Wettest Quarter (BIO16)
6. Precipitation of Warmest Quarter (BIO18)
7. Distance from the Nearest Poultry Facility
8. Number of Nearby Poultry Facilities
9. Distance from the Nearest Infected Wild Animal
10. Number of Nearby Wild Animal Infections

---

## Data Augmentation

The reported H5Nx occurrence data contain substantially fewer presence observations than background/absence observations. To address this class imbalance during model development, two data-augmentation approaches were investigated.

### Stochastic Variational Inference (SVI)

The SVI-based approach represents the standardized predictor space probabilistically and uses a two-component Gaussian formulation to generate additional synthetic positive observations.

The model uses the 10 selected predictors as a 10-dimensional representation of the predictor space. Synthetic observations are generated from the estimated multivariate Gaussian distribution associated with the positive class.

The final analysis used **500 synthetic positive observations**. The number of synthetic observations was selected after comparing model results using larger augmentation sizes; model results were essentially unchanged between 500 and 1,000 synthetic observations.

### Empirical Distribution (ED)

The ED approach generates additional positive observations based on the empirical distribution of the observed predictor variables.

The SVI and ED approaches were evaluated using the same Logistic Regression and Random Forest models to investigate how the choice of data-augmentation method affects predictive performance and geographic generalizability.

---

## Predictive Models

Four model configurations were developed and evaluated:

| Model | Data Augmentation |
|---|---|
| LR + SVI | Stochastic Variational Inference |
| LR + ED | Empirical Distribution |
| RF + SVI | Stochastic Variational Inference |
| RF + ED | Empirical Distribution |

The Logistic Regression models provide a statistical modeling framework, whereas the Random Forest models capture potentially nonlinear relationships and interactions among predictors.

---

## Model Evaluation

Model performance was evaluated using multiple complementary validation strategies.

### Random Train-Test Split

A random train-test split was initially used to provide a conventional assessment of predictive performance.

### Spatially Blocked 5-Fold Cross-Validation

Spatially blocked 5-fold cross-validation was used to assess **geographic generalizability** and reduce the potential influence of spatial dependence between training and validation observations.

This approach provides a more stringent assessment of whether model performance is maintained when predictions are made for geographically separated observations.

### Independent Temporal Validation

Data from **2024** were used as an independent temporal validation period to evaluate model performance on a subsequent outbreak period that was not used for model training.

---

## Performance Metrics

To provide a comprehensive assessment of model performance, particularly under class imbalance, the following metrics were evaluated:

- ROC-AUC
- Accuracy
- Precision
- Recall / Sensitivity
- F1-score
- Specificity
- Balanced Accuracy
- Brier Score
- Log Loss

The use of multiple metrics allows model performance to be assessed beyond accuracy and ROC-AUC alone.

---

## Spatial Risk Maps

The predictive models were used to generate spatial risk maps representing the predicted probability of H5Nx occurrence across the study area.

An ensemble risk map was generated by combining predictions from selected model configurations based on predictive performance and spatial generalizability.

Individual model risk maps are also provided in the repository and/or Supporting Information associated with the manuscript.

---

## Feature Importance

Permutation-based feature importance was used to examine the relative importance of predictors in the Random Forest models.

The analysis focuses on environmental, wildlife-related, and poultry-related predictors associated with predicted H5Nx risk.

Feature-importance results should be interpreted as **model-based associations rather than causal effects**. In particular, correlated predictors may influence permutation-importance rankings.

---

## Exploratory Spatial Clustering

A **Gaussian Mixture Model (GMM)** was used to characterize the spatial distribution of reported H5Nx outbreaks.

The GMM analysis was used as an **exploratory and descriptive spatial analysis** to identify areas where observed outbreaks were spatially concentrated and to facilitate qualitative comparison with predicted risk patterns.

Importantly, GMM clustering is **not considered an independent validation of model performance**, because the outbreak observations used for clustering were also included in the data used to develop the predictive models.

Geographic and temporal generalizability were instead assessed using spatially blocked cross-validation and independent temporal validation.

---

## Repository Structure

The repository is organized to separate data, preprocessing, modeling, validation, and visualization workflows.

```text
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── derived/
├── src/
│   ├── data_processing/
│   ├── augmentation/
│   ├── models/
│   ├── validation/
│   └── visualization/
├── notebooks/
├── results/
│   ├── tables/
│   ├── figures/
│   └── risk_maps/
└── LICENSE
```

> **Note:** The folder structure above should be updated to match the actual organization of this repository.

---

## Software Requirements

The analysis was implemented in Python.

The required Python packages are listed in `requirements.txt`. Key packages used in the analysis include:

```text
numpy
pandas
scipy
scikit-learn
torch
pyro-ppl
rasterio
plotly
matplotlib
```

Additional packages may be required depending on the specific preprocessing, modeling, and visualization scripts.

To install the required dependencies:

```bash
pip install -r requirements.txt
```

It is recommended to use a dedicated Python virtual environment or Conda environment.

---

## Reproducing the Analysis

The general workflow for reproducing the analysis is:

1. Download the required source datasets from their respective official sources.
2. Place the datasets in the appropriate data directories.
3. Run the data preprocessing scripts.
4. Perform predictor selection and preprocessing.
5. Generate synthetic positive observations using the SVI and ED procedures.
6. Train the Logistic Regression and Random Forest models.
7. Evaluate the models using random train-test splitting.
8. Perform spatially blocked 5-fold cross-validation.
9. Perform independent temporal validation using the 2024 data.
10. Generate performance metrics and model-comparison tables.
11. Generate feature-importance results.
12. Generate individual and ensemble risk maps.
13. Perform exploratory GMM spatial clustering.

The repository documentation and scripts provide additional information about the required input files and computational workflow.

Because the external datasets may be updated by their respective providers, the repository documents the source, version/date, and download information for the datasets used in the study where available.

---

## Data Availability and Use

The primary datasets used in this study were obtained from publicly accessible external sources:

- **FAO EMPRES-i:** Avian influenza occurrence data
- **USDA FSIS:** U.S. poultry establishment data
- **WorldClim:** Historical bioclimatic data

Users should consult the respective data providers for current terms of use, licensing, and redistribution conditions before downloading or redistributing these datasets.

Where redistribution of the original source data is not permitted, this repository provides the information necessary to identify and obtain the source data and reproduce the associated data-processing workflow.

---

## Citation

If you use the code or analysis workflow provided in this repository, please cite the associated publication:

> **[TODO: Add the final manuscript citation here]**

For example:

```text
Author(s). Title. PLOS ONE. Year.
DOI: [TODO: Add DOI]
```

---

## Reproducibility

The code in this repository is provided to facilitate transparency and reproducibility of the analyses reported in the associated publication.

The repository includes computational workflows used for:

- Data preprocessing
- Predictor selection
- SVI data augmentation
- ED data augmentation
- Logistic Regression modeling
- Random Forest modeling
- Random train-test validation
- Spatially blocked 5-fold cross-validation
- Independent temporal validation
- Feature-importance analysis
- Risk-map generation
- Exploratory GMM spatial clustering

The repository version corresponding to the published study should be archived in a persistent repository and assigned a DOI to facilitate long-term access and reproducibility.

---

## License

**[TODO: Add the selected open-source license here]**

The code should be released under an appropriate open-source license to facilitate reuse.

The license for the code does **not** automatically apply to the external datasets, which remain subject to the terms and conditions established by their respective data providers.

---

## Acknowledgements

The authors acknowledge the organizations that provide the publicly accessible datasets used in this study:

- Food and Agriculture Organization of the United Nations (FAO) EMPRES-i
- United States Department of Agriculture (USDA) Food Safety and Inspection Service (FSIS)
- WorldClim

---

## Contact

For questions regarding the code or analysis workflow, please contact:

**[TODO: Add corresponding author/repository contact information]**
