# Predicting H5Nx Hotspots in the USA Using Species Distribution Modeling

In this repository code and data for building a prediction model for Avian Influenza in the USA is implemented. Three datasets are used for this purpose:

* **Avian influenza case counts:** This dataset which is available from [Empres-i](https://empres-i.apps.fao.org/general) includes geocoordinates of avian influenza cases of wild and domestic birds along with their observation date. This dataset has been collected from 2004-01-01 to 2024-12-31.
* **Poultry locations in the USA as of 2024:** This dataset was collected from [US Department of Agriculture (USDA)](https://www.fsis.usda.gov/inspection/establishments/meat-poultry-and-egg-product-inspection-directory), and includes geocoordinates of poultry facilities in America, as of 2024.
* **Climate and environmental factors:** This data is available from [WorldClim](https://www.worldclim.org/data/worldclim21.html). It includes 19 climate factors which were collected from 1970 to 2000. These files are in .tif format and can be converted to .csv format using the "tif2csv.py" file in this repository.

After removing multicollinearity from the datasets, 10 factors were retained for predicting avian influenza hotspots in the USA using 2 species distribution modeling methods:

* Logistic Regressoin (LR)
* Random Forest Classifier (RF)

The H5Nx cases collected from Empres-i were treated as the presence data. However, the small volume of the case count dataset caused a hige inbalance between the presence and absence data. To metigate this, synthetic presence data was generated using two methods:

* Stochastic Variational Inference (SVI)
* Empirical Distribution (ED)

Therefore, 4 models are implemented and compared for predicting avian influenza hotspots: LR+SVI, LR+ED, RF+SVI, RF+ED. The code provided in this repository implements and evaluates these 4 predictive models.

### Model Evaluation

Model performance was evaluated using multiple complementary validation strategies.

**Random Train-Test Split**

A random train-test split was initially used to provide a conventional assessment of predictive performance.

**Spatially Blocked 5-Fold Cross-Validation**

Spatially blocked 5-fold cross-validation was used to assess **geographic generalizability** and reduce the potential influence of spatial dependence between training and validation observations.

This approach provides a more stringent assessment of whether model performance is maintained when predictions are made for geographically separated observations.

**Independent Temporal Validation**

Data from **2024** were used as an independent temporal validation period to evaluate model performance on a subsequent outbreak period that was not used for model training.

---

### Software Requirements

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
pathlib
```

Additional packages may be required depending on the specific preprocessing, modeling, and visualization scripts.

---
