import math
from copy import deepcopy

import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import webbrowser
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import rasterio
from matplotlib.colors import LogNorm
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from itertools import product
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.io as pio
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss
)
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.inspection import permutation_importance
import shap
import copy
def show_data_on_map5 (data_df, **kwargs):
  # Ensure your probability column is sorted if you want high-values on top
  data_df = data_df.sort_values(by=kwargs['color'])
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
  
  fig = go.Figure(go.Scattermapbox(
      lat = data_df['latitude'],
      lon = data_df['longitude'],
      mode = 'markers',
      marker = dict(
          size = 5,                 # Small marker size matches your square-like grain
          opacity = 0.85,           # Slight transparency lets overlapping values blend
          color = data_df[kwargs['color']], # Maps color mapping scale dynamically

          # This matches the purple -> pink -> orange -> bright yellow scale shown
          colorscale = 'inferno',   # Alternatively, use custom: [[0, 'indigo'], [0.5, 'magenta'], [1, 'yellow']]
          cmin = 0.1,               # Locks the scale floor to 0.1 as shown in your colorbar
          cmax = 1,               # Locks the scale ceiling to matches your image colorbar

          colorbar = dict(
              title = kwargs['color'],
 #             titleside = 'top',
              thickness = 20,
              len = 0.9
          )
      ),
      text = data_df[kwargs['color']], # Hover data display
      hoverinfo = 'text'
  ))

  # Configure the canvas to exactly replicate the solid light grey vector US landmass
  fig.update_layout(
      mapbox = dict(
          #style="carto-positron",       # Keeps background completely clean and copyright-free
          style="white-bg",
          center = dict(lat=37.0902, lon=-95.7129), # Centered perfectly over USA
          zoom = 3.5,

          # Build out the light-grey US polygon backdrop layer manually via GeoJSON
          layers = [{ "sourcetype": "raster",
                      "source": [f"https://cartocdn.com{{z}}/{{x}}/{{y}}.png?key={carto_key}"],
                      "below": "traces"
                  }]
      ),

      annotations=[dict(
        text='<a href="https://carto.com/attributions" style="color: #444; text-decoration: none;">© CARTO</a> | <a href="https://openstreetmap.org" style="color: #444; text-decoration: none;">© OpenStreetMap</a>',
        xref="paper", yref="paper",
        x=0.99, y=0.01,          # Coordinates: Bottom-right corner
        showarrow=False,
        align="right",
        bgcolor="rgba(255, 255, 255, 0.7)", # Slightly transparent white backing for readability
        borderpad=4
      )],

      margin = {"r":0, "t":30, "l":0, "b":0},
      width = 1100,
      height = 700
  )

  fig.show()

def modify_locations(df_to_modify, bio_us_data_df):
    n = len(df_to_modify)
    m = len(bio_us_data_df)
    df_to_modify_copy = deepcopy(df_to_modify)
    df_to_modify_copy_with_locations_only = df_to_modify_copy[['latitude', 'longitude']]
    for i in range(n):
        target_location = df_to_modify_copy_with_locations_only.iloc[i]
        locations = bio_us_data_df[['latitude', 'longitude']]
        diffs = target_location - locations
        distances = np.linalg.norm(diffs, axis=-1)
        min_distance_index = np.argmin(distances)
        optimal_latitude = bio_us_data_df.loc[min_distance_index, 'latitude']
        optimal_longitude = bio_us_data_df.loc[min_distance_index, 'longitude']
        df_to_modify_copy.iloc[i, df_to_modify_copy.columns.get_loc('latitude')] = optimal_latitude
        df_to_modify_copy.iloc[i, df_to_modify_copy.columns.get_loc('longitude')] = optimal_longitude

    return df_to_modify_copy


def z_normalizing(df, colns_to_ignore, colns_to_restore):
    # normalize vars using Z-score normalization
    scaler = StandardScaler()
    z_score_normalized_complete_bio_us_data = scaler.fit_transform(df.drop(columns=colns_to_ignore))
    # remove keys in df.columns that are in colns_to_ignore
    colns = []
    for key in list(df.columns):
        if key not in colns_to_ignore:
            colns.append(key)
    z_score_normalized_complete_bio_us_data_df = pd.DataFrame(z_score_normalized_complete_bio_us_data, columns=colns)
    for key in colns_to_restore:
        z_score_normalized_complete_bio_us_data_df.loc[:, key] = df[key]

    return z_score_normalized_complete_bio_us_data_df


def add_distance_to_poultry_coln(df):
    """
    add a column <distance_to_nearest_poultry> to df that represents the distance between the location and its
    nearest poultry

    :param df: the bio us data
    :return: void
    """
    n = len(df)
    df['distance_to_nearest_poultry'] = 0
    poultry_spots_locations = df[df['poultry_observed'] == 1][['latitude', 'longitude']]
    #df['distance_to_nearest_poultry'] = [np.min(np.linalg.norm((df[df.index.isin([i])][['latitude','longitude']].to_numpy() - poultry_spots_locations).to_numpy(dtype=np.float64), axis=1)) for i in df.index]
    dummy = []
    for i in range(n):
        row = df.iloc[i]
        # check if there is any nearby poultry
        location = row[['latitude', 'longitude']].to_numpy()
        location_diff_vector = (location - poultry_spots_locations).to_numpy(dtype=np.float64)
        location_distances = np.linalg.norm(location_diff_vector, axis=1)
        min_d = np.min(location_distances, axis=0)
    #    df.iloc[i, df.columns.get_loc('distance_to_nearest_poultry')] = min_d
        dummy.append(min_d)
    df['distance_to_nearest_poultry'] = dummy

def add_num_of_poultry_coln(df, threshold):
    """
    add a column <num_of_nearby_poultry> to df that represents the number of poultry for a given location

    :param df: the bio us data
    :param threshold: the distance cutoff under which a poultry is deemed as nearby
    :return: void
    """
    n = len(df)
    df['num_of_nearby_poultry'] = 0
    poultry_spots_locations = df[df['poultry_observed'] == 1][['latitude', 'longitude']]
    dummy = []
    for i in range(n):
        row = df[df.index.isin([i])] #df.iloc[i]
        # check if there is any nearby poultry
        location = row[['latitude', 'longitude']].to_numpy()
        location_diff_vector = (location - poultry_spots_locations).to_numpy(dtype=np.float64)
        location_distances = np.linalg.norm(location_diff_vector, axis=1)
        nearby_location_distances = location_distances[location_distances < threshold]
        #df.iloc[i, df.columns.get_loc('num_of_nearby_poultry')] = len(nearby_location_distances)
        dummy.append(len(nearby_location_distances))
    df['num_of_nearby_poultry'] = dummy

def add_distance_to_infected_wild_animals(df, h5n1_cases):
    """
    add a column <distance_to_infected_wild_animals> to df that represents the distance between the location and its
    nearest infected wild animals

    :param df: the bio us data
    :param h5n1_cases: the h5n1 cases df
    :return: void
    """
    n = len(df)
    df['distance_to_nearest_infected_wild_animals'] = 0
    infected_wild_animals_spots_locations = h5n1_cases[h5n1_cases['species'].str.contains('Wild', na=False)][['latitude', 'longitude']]
    #df['distance_to_nearest_infected_wild_animals'] = [np.min(np.linalg.norm((df[df.index.isin([i])][['latitude', 'longitude']].to_numpy() - infected_wild_animals_spots_locations).to_numpy(dtype=np.float64), ord=2, axis=1), axis=0) for i in df.index]
    dummy = []
    for i in range(n):
        row = df.iloc[i]
        # check if there is any nearby poultry
        location = row[['latitude', 'longitude']].to_numpy()
        location_diff_vector = (location - infected_wild_animals_spots_locations).to_numpy(dtype=np.float64)
        location_distances = np.linalg.norm(location_diff_vector, ord=2, axis=1)
        min_d = np.min(location_distances, axis=0)
   #     df.iloc[i, df.columns.get_loc('distance_to_nearest_infected_wild_animals')] = min_d
        dummy.append(min_d)
    df['distance_to_nearest_infected_wild_animals'] = dummy


def add_num_of_infected_wild_animals(df, h5n1_cases, threshold):
    """
    add a column <num_of_infected_wild_animals> to df that represents the number of nearby infected wild animals

    :param df: the bio us data
    :param h5n1_cases: the h5n1 cases df
    :param threshold: the distance cutoff under which it is deemed as nearby
    :return: void
    """
    n = len(df)
    df['num_of_nearby_infected_wild_animals'] = 0
    infected_wild_animals_spots_locations = h5n1_cases[h5n1_cases['species'].str.contains('Wild', na=False)][['latitude', 'longitude']]
    dummy = []
    for i in range(n):
        row = df.iloc[i]
        # check if there is any nearby poultry
        location = row[['latitude', 'longitude']].to_numpy()
        location_diff_vector = (location - infected_wild_animals_spots_locations).to_numpy(dtype=np.float64)
        location_distances = np.linalg.norm(location_diff_vector, axis=1)
        nearby_location_distances = location_distances[location_distances < threshold]
        dummy.append(len(nearby_location_distances))
    df['num_of_nearby_infected_wild_animals'] = dummy

def calculate_contribution_level(X_train, X_test, y_train, y_test, var, total_AUC, model):
   X_train_dropped = X_train.drop(var, axis=1, inplace=False)
   X_test_dropped = X_test.drop(var, axis=1, inplace=False)
   model = copy.deepcopy(model)
   model.fit(X_train_dropped, y_train)
   y_pred= model.predict(X_test_dropped)
   # Calculate AUC
   AUC = roc_auc_score(y_test, y_pred)
   print(f"AUC by dropping {var} = {AUC}")
   return total_AUC - AUC


def calculate_shap_contribution_level(model, X_train, X_test):
    # Create SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Plot summary of feature importance
    shap.summary_plot(shap_values, X_test)

    # Convert SHAP values to a DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)
    # Get a summary statistics for the SHAP values
    summary_stats = shap_df.describe()
    print(summary_stats)


def calculate_permutation_importance(model, X_train, X_test, y_train, y_test):
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test)
    print(perm_importance.importances_mean)


def roc_and_auc(y_true, y_pred, model_name):
  # Compute roc curve and auc value
  fpr, tpr, thresholds = roc_curve(y_true, y_pred)
  roc_auc = auc(fpr, tpr)
  print(f"total AUC = {roc_auc}")

  # Plot the roc curve
  plt.figure()
  plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
  plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f"{model_name} ROC and AUC")
  plt.legend(loc='lower right')
  plt.show()


def _calculate_correlation(var1, var2, df):
  df_filtered = df[[var1, var2]]
  corr_matrix = df_filtered.corr()
  print(f"{var1} and {var2} have correlation matrix = {corr_matrix}")


def create_spatial_blocks(data_df, lat_col='latitude', lon_col='longitude', block_size=2.0):
    """
    Assign each observation to a geographic spatial block.
    block_size is expressed in degrees.
    """

    df = data_df.copy()

    df['lat_block'] = np.floor(df[lat_col] / block_size).astype(int)
    df['lon_block'] = np.floor(df[lon_col] / block_size).astype(int)
    df['spatial_block'] = (df['lat_block'].astype(str) + '_' + df['lon_block'].astype(str))

    return df

def assign_spatial_folds(data_df, n_splits=5):
    """
    Assign observations to spatial folds.
    All observations within the same spatial block
    remain in the same fold.
    """

    df = data_df.copy()
    groups = df["spatial_block"]
    group_kfold = GroupKFold(n_splits=n_splits)
    df["spatial_fold"] = -1
    for fold, (_, test_idx) in enumerate(group_kfold.split(df, groups=groups)):
        df.iloc[test_idx, df.columns.get_loc("spatial_fold")] = fold

    return df






