import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import xy
import os
from tqdm import tqdm
import math
from copy import deepcopy

import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import rasterio
from matplotlib.colors import LogNorm
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from itertools import product
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.inspection import permutation_importance
import copy
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

# some helper functions
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

#---------------------------------------------------------------------
bio_us_data_df = pd.read_csv('./bio_us_data_df.csv')
bio_us_data_df = bio_us_data_df[['longitude','latitude','wc2.1_10m_bio_2','wc2.1_10m_bio_7','wc2.1_10m_bio_8','wc2.1_10m_bio_15','wc2.1_10m_bio_16','wc2.1_10m_bio_18']]

us_h5n1_cases_df = pd.read_csv("./Dataset/empresi_h5nx.csv")
us_h5n1_cases_df = us_h5n1_cases_df[us_h5n1_cases_df['country'] == 'United States of America']
us_h5n1_cases_df = us_h5n1_cases_df[~pd.isnull(us_h5n1_cases_df['species'])]
us_h5n1_cases_df = us_h5n1_cases_df[['locality', 'species', 'latitude', 'longitude', 'observation_date']]

us_poultry_df = pd.read_csv("./Dataset/USDA_Poultry.csv", sep='\t', encoding='utf-16')
us_poultry_df = us_poultry_df.rename(columns={'Latitude (generated)':'latitude', 'Longitude (generated)':'longitude'})

simulated_present_samples = pd.read_csv('./svi.csv')

#---------------------------------------------------------------------
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

def calculate_permutation_importance(model, X_train, X_test, y_train, y_test):
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test)
    print(perm_importance.importances_mean)

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

def show_data_on_map (data_df, **kwargs):
  # Ensure your probability column is sorted if you want high-values on top
  data_df = data_df.sort_values(by=kwargs['color'])

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

  # Configure the canvas to use CARTO Positron
  fig.update_layout(
      mapbox = dict(
          style = "carto-positron",                 # Switched to CARTO Positron basemap
          center = dict(lat=37.0902, lon=-95.7129), # Centered perfectly over USA
          zoom = 3.5
      ),

      annotations=[dict(
        text='<a href="https://carto.com" style="color: #444; text-decoration: none;">© CARTO</a> | <a href="https://openstreetmap.org" style="color: #444; text-decoration: none;">© OpenStreetMap</a>',
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

#---------------------------------------------------------------
def max_ent(X, y, model_name='Model B'):
    """
    we use a logistic regression model as an efficient replacement of maximum entropy model because we don't need to
    transform our feature x_1, .., x_19
    :param X: the features of all samples
    :param y: the label of all samples: 0 for absence for h5n1 and 1 for presence for h5n1
    :return:
    """
    # split dataset (X:features, y:label) into 2 categories: train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # build up the max_entropy model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print(model.coef_)
    print(model.intercept_)

    # make a prediction
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Calculate AUC
    AUC = roc_auc_score(y_test, y_pred)
    print(f"total AUC = {AUC}")

    # The reviewer asked for all of these:
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Sensitivity = Recall
    sensitivity = recall_score(y_test, y_pred)
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp)
    # Balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # -----------------------------
    # Display results
    # -----------------------------

    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-score:             {f1:.4f}")
    print(f"Sensitivity:          {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"Balanced Accuracy:    {balanced_accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
    #---------------------------------------------
    # Calculate AUC and draw ROC Curve
    roc_and_auc(y_test, y_pred, model_name)

    # Calculate permutation importances
    calculate_permutation_importance(model, X_train, X_test, y_train, y_test)

    model.densify()
    y_prob_all = model.predict_proba(X)

    return y_prob_all[:,1], model
#---------------------------------------------------------------
def max_ent_for_us(new_zs=None, model_name='LR with SVI - Random Train-Test Split'):
    # modify locations
    h5n1_us_data_df_with_locations_modified = modify_locations(us_h5n1_cases_df, bio_us_data_df)

    wild_h5n1_us_data_df_with_locations_modified = h5n1_us_data_df_with_locations_modified[~h5n1_us_data_df_with_locations_modified['species'].str.contains('Wild', case=False, na=False)]
    wild_h5n1_us_data_df_locations_only = wild_h5n1_us_data_df_with_locations_modified[['latitude', 'longitude']]
    wild_h5n1_us_data_df_locations_only['observed'] = 1
    # load us poultry data
    poultry_locations_df = us_poultry_df[['latitude', 'longitude']]
    poultry_locations_df_locations_modified = modify_locations(poultry_locations_df, bio_us_data_df)
    poultry_locations_df_locations_modified['poultry_observed'] = 1
    # merge bio_us_data with us h5n1 observed cases
    bio_us_data_df_with_hn51_obs = pd.merge(bio_us_data_df, wild_h5n1_us_data_df_locations_only,
                                                on=['latitude', 'longitude'], how='outer')
    # merge bio_us_data with us poultry locations data
    bio_us_data_df_with_hn51_obs_and_poultry_obs = pd.merge(bio_us_data_df_with_hn51_obs,
                                                                poultry_locations_df_locations_modified,
                                                                on=['latitude', 'longitude'], how='outer')
    bio_us_data_df_with_hn51_obs_and_poultry_obs = bio_us_data_df_with_hn51_obs_and_poultry_obs.fillna(0)

    # add distance_to_nearest_poultry column
    add_distance_to_poultry_coln(bio_us_data_df_with_hn51_obs_and_poultry_obs)
    # add # nearby poultry facilities column
    add_num_of_poultry_coln(bio_us_data_df_with_hn51_obs_and_poultry_obs, 2)
    # add distance to infected wild animals
    add_distance_to_infected_wild_animals(bio_us_data_df_with_hn51_obs_and_poultry_obs, h5n1_us_data_df_with_locations_modified)
    # add # nearby infected wild animals
    add_num_of_infected_wild_animals(bio_us_data_df_with_hn51_obs_and_poultry_obs, h5n1_us_data_df_with_locations_modified, 2)
    # drop poultry_observed column
    target_bio_us_data_df_with_hn51_obs = bio_us_data_df_with_hn51_obs_and_poultry_obs.drop(
        columns=['poultry_observed'])
    # normalize data
    z_normalized_bio_us_data_df = z_normalizing(target_bio_us_data_df_with_hn51_obs,
                                                ['latitude', 'longitude', 'observed'],
                                                ['latitude', 'longitude', 'observed'])

    z_normalized_bio_us_data_df = z_normalized_bio_us_data_df.sample(frac=1).reset_index(drop=True)

    if new_zs is not None:
        z_normalized_bio_us_data_df = pd.concat([z_normalized_bio_us_data_df, new_zs],ignore_index=True)

    # fit the logistic regression model
    z_normalized_bio_us_data_df.fillna(0, inplace=True)

    y_predict_all, max_ent_model = max_ent(
        z_normalized_bio_us_data_df.drop(columns=['latitude', 'longitude', 'observed']),
        z_normalized_bio_us_data_df['observed'], model_name)
    z_normalized_bio_us_data_df_pred = deepcopy(z_normalized_bio_us_data_df)
    z_normalized_bio_us_data_df_pred.loc[:, 'probability'] = y_predict_all
    z_normalized_bio_us_data_df_pred_true = z_normalized_bio_us_data_df_pred[
        (z_normalized_bio_us_data_df_pred['probability'] > 0.1) &
        (z_normalized_bio_us_data_df_pred['latitude'] < 10000) &
        (z_normalized_bio_us_data_df_pred['longitude'] < 10000)]

    # show on the map
    show_data_on_map (z_normalized_bio_us_data_df_pred_true, color='probability')

    return max_ent_model
#--------------------------------------------
max_ent_for_us(simulated_present_samples)
