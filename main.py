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

def roc_5_fold (fold_roc_data, fold_auc, model_name):
    """
    Plot mean ROC curve with ±1 SD shaded region.
    """

    mean_fpr = np.linspace(0, 1, 101)
    interpolated_tprs = []
    for fpr, tpr in fold_roc_data:
        interp_tpr = np.interp(mean_fpr, fpr, tpr)

        # ROC curves start at (0,0)
        interp_tpr[0] = 0.0
        interpolated_tprs.append(interp_tpr)

    interpolated_tprs = np.array(interpolated_tprs)

    # Mean and SD of TPR across folds
    mean_tpr = np.mean(interpolated_tprs, axis=0)
    sd_tpr = np.std(interpolated_tprs, axis=0, ddof=1)

    # Upper and lower bounds
    tpr_upper = np.minimum(mean_tpr + sd_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - sd_tpr, 0)

    # Mean AUC and SD
    mean_auc = np.mean(fold_auc)
    sd_auc = np.std(fold_auc, ddof=1)

    # Plot
    plt.figure(figsize=(7, 6))

    plt.plot(mean_fpr, mean_tpr, linewidth=2, label=(f"Mean ROC ", f"(AUC = {mean_auc:.2f} ± {sd_auc:.2f})"))

    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.2, label="± 1 SD")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(f"{model_name} - 5-Fold Spatial Cross-Validation")

    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    print(f"{model_name}: ", f"Mean AUC = {mean_auc:.3f}, ", f"SD = {sd_auc:.3f}")
    return mean_auc, sd_auc

def max_ent_for_us_5_fold (new_zs=None, n_splits= 5):
    #group_kfold = GroupKFold(n_splits=n_splits)
    fold_acc = []
    fold_auc = []
    curves_auc = []

    # load us poultry data
    poultry_locations_df = us_poultry_df[['latitude', 'longitude']]
    poultry_locations_df_locations_modified = modify_locations(poultry_locations_df, bio_us_data_df)
    poultry_locations_df_locations_modified['poultry_observed'] = 1

    # modify locations
    h5n1_us_data_df_with_locations_modified = modify_locations(us_h5n1_cases_df, bio_us_data_df)
    # h5n1_us_data_df = h5n1_us_data_df[h5n1_us_data_df['species'].str.contains('Wild', case=False, na=False)]
    # only focus on domestic cases
    wild_h5n1_us_data_df_with_locations_modified = h5n1_us_data_df_with_locations_modified[~h5n1_us_data_df_with_locations_modified['species'].str.contains('Wild', case=False, na=False)]
    wild_h5n1_us_data_df_locations_only = wild_h5n1_us_data_df_with_locations_modified[['latitude', 'longitude']]
    wild_h5n1_us_data_df_locations_only['observed'] = 1
    # merge bio_us_data with us h5n1 observed cases
    bio_us_data_df_with_hn51_obs = pd.merge(bio_us_data_df, wild_h5n1_us_data_df_locations_only, on=['latitude', 'longitude'], how='outer')
    # merge bio_us_data with us poultry locations data
    bio_us_data_df_with_hn51_obs = pd.merge(bio_us_data_df_with_hn51_obs, poultry_locations_df_locations_modified, on=['latitude', 'longitude'], how='outer')

    #---------------- My Code ---------------
    # Create spatial blocks BEFORE removing latitude/longitude
    bio_us_data_df_with_hn51_obs = create_spatial_blocks(bio_us_data_df_with_hn51_obs, lat_col='latitude', lon_col='longitude', block_size=2.0)
    spatial_groups = assign_spatial_folds (bio_us_data_df_with_hn51_obs)

    #spatial_groups = bio_us_data_df_with_hn51_obs['spatial_block'].reset_index(drop=True)

    #X = wild_h5n1_us_data_df_locations_only.copy() #.drop(columns=['latitude','longitude','observed','lat_block','lon_block','spatial_block'])
    #y = wild_h5n1_us_data_df_locations_only[['observed','spatial_groups']]

    for fold in tqdm(pd.unique(spatial_groups['spatial_fold'])):

        test_dummy = spatial_groups[spatial_groups['spatial_fold'] == fold].copy()
        train = spatial_groups.copy()
        train['observed'] = [0 if train['spatial_fold'][idx] == fold else train['observed'][idx] for idx in train.index]


        train = train.fillna(0)

        # add distance_to_nearest_poultry column
        add_distance_to_poultry_coln(train)
        # add # nearby poultry facilities column
        add_num_of_poultry_coln(train, 2)
        # add distance to infected wild animals
        add_distance_to_infected_wild_animals(train, h5n1_us_data_df_with_locations_modified)
        # add # nearby infected wild animals
        add_num_of_infected_wild_animals(train, h5n1_us_data_df_with_locations_modified, 2)
        # drop poultry_observed column
        target_bio_us_data_df_with_hn51_obs = train.drop(columns=['poultry_observed'])
        # normalize data
        z_normalized_bio_us_data_df = z_normalizing(target_bio_us_data_df_with_hn51_obs,['latitude', 'longitude', 'observed','lat_block','lon_block','spatial_block','spatial_fold'],['latitude', 'longitude', 'observed','lat_block','lon_block','spatial_block','spatial_fold'])

        train = z_normalized_bio_us_data_df[z_normalized_bio_us_data_df['spatial_fold'] != fold]
        test = z_normalized_bio_us_data_df[z_normalized_bio_us_data_df['spatial_fold'] == fold]
        test['observed'] = test_dummy['observed'] #[test_dummy['observed'][i] for i in test.index]
        test.fillna(0, inplace=True)


        #train = train.sample(frac=1).reset_index(drop=True)
        if new_zs is not None:
          train = pd.concat([train, new_zs],ignore_index=True)
        train.fillna(0, inplace=True)

        X_train = train.drop (columns= ['latitude','longitude','observed','lat_block','lon_block','spatial_block','spatial_fold'])
        X_test = test.drop (columns= ['latitude','longitude','observed','lat_block','lon_block','spatial_block','spatial_fold'])

        y_train = train['observed']
        y_test = test['observed']


        print("Training observations:", len(X_train))
        print("Testing observations:", len(X_test))

        # Build logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predictions for accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        fold_acc.append(accuracy)

        # Probability predictions for AUC
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"Fold {fold + 1} AUC: {auc:.4f}")
        fold_auc.append(auc)
        curves_auc.append({'y_test':y_test, 'y_prob':y_prob})

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

        # Brier score (lower is better)
        brier_score = brier_score_loss(y_test, y_prob)

        # Log loss / cross-entropy (lower is better)
        logloss = log_loss(y_test, y_prob)

        print(f"Accuracy:             {accuracy:.4f}")
        print(f"Precision:            {precision:.4f}")
        print(f"Recall:               {recall:.4f}")
        print(f"F1-score:             {f1:.4f}")
        print(f"Sensitivity:          {sensitivity:.4f}")
        print(f"Specificity:          {specificity:.4f}")
        print(f"Balanced Accuracy:    {balanced_accuracy:.4f}")
        print(f"Brier Score:          {brier_score:.4f}")
        print(f"Log Loss:              {logloss:.4f}")

        print("\nConfusion Matrix:")
        print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")

    return fold_acc, fold_auc, curves_auc

fold_acc, fold_auc, curves_auc = max_ent_for_us_5_fold (None)

mean_acc = np.mean(fold_acc)
sd_acc = np.std(fold_acc, ddof=1)

mean_auc = np.mean(fold_auc)
sd_auc = np.std(fold_auc, ddof=1)

print(f"Mean ACC: {mean_acc:.3f}")
print(f"SD: {sd_acc:.3f}")

print(f"Mean AUC: {mean_auc:.3f}")
print(f"SD: {sd_auc:.3f}")

fold_roc_data = []
fold_auc = []
for item in curves_auc:
  fpr, tpr, thresholds = roc_curve(item['y_test'], item['y_prob'])
  fold_auc_value = auc(fpr, tpr)
  fold_roc_data.append((fpr, tpr))
  fold_auc.append(fold_auc_value)

roc_5_fold (fold_roc_data, fold_auc, model_name='No Augmentation')

def max_ent(X, y, model_name='Model B', groups=None, test_size=0.2, use_spatial_split=True):
      if use_spatial_split:
          if groups is None:
              raise ValueError("Spatial groups must be provided when use_spatial_split=True.")

          splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

          train_idx, test_idx = next(splitter.split(X, y, groups=groups))

          X_train = X.iloc[train_idx]
          X_test = X.iloc[test_idx]

          y_train = y.iloc[train_idx]
          y_test = y.iloc[test_idx]

      else:
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

      # Build logistic regression model
      model = LogisticRegression()
      model.fit(X_train, y_train)

      print("Coefficients:")
      print(model.coef_)

      print("Intercept:")
      print(model.intercept_)

      # Predictions
      y_pred = model.predict(X_test)

      # IMPORTANT: probability, not hard class, for AUC
      y_prob = model.predict_proba(X_test)[:, 1]

      # Accuracy
      accuracy = accuracy_score(y_test, y_pred)
      print(f"Accuracy: {accuracy:.2f}")

      # AUC
      AUC = roc_auc_score(y_test, y_prob)
      print(f"Spatial test AUC = {AUC:.4f}")

      print(f"Accuracy:             {accuracy:.4f}")
  # The reviewer asked for all of these:
      precision = precision_score(y_test, y_pred)
      print(f"Precision:            {precision:.4f}")
      recall = recall_score(y_test, y_pred)
      print(f"Recall:               {recall:.4f}")
      f1 = f1_score(y_test, y_pred)
      print(f"F1-score:             {f1:.4f}")
      # Sensitivity = Recall
      sensitivity = recall_score(y_test, y_pred)
      print(f"Sensitivity:          {sensitivity:.4f}")
      # Confusion matrix
      tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
      print("\nConfusion Matrix:")
      print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
      # Specificity = TN / (TN + FP)
      specificity = tn / (tn + fp)
      print(f"Specificity:          {specificity:.4f}")
      # Balanced accuracy
      balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
      print(f"Balanced Accuracy:    {balanced_accuracy:.4f}")

      # Brier score (lower is better)
      brier_score = brier_score_loss(y_test, y_prob)
      print(f"Brier Score:          {brier_score:.4f}")
      # Log loss / cross-entropy (lower is better)
      logloss = log_loss(y_test, y_prob)
      print(f"Log Loss:              {logloss:.4f}")

      # ROC curve
      roc_and_auc(y_test, y_prob, model_name)

      # SHAP
      calculate_shap_contribution_level(model, X_train, X_test)

      # Permutation importance
      calculate_permutation_importance(model, X_train, X_test, y_train, y_test)
      model.densify()

      # Predictions for ALL observations
      y_prob_all = model.predict_proba(X)

      return y_prob_all[:, 1], model

def max_ent_for_us(new_zs=None, if_save_risk_map=False, model_name='Model B'):
    # modify locations
    h5n1_us_data_df_with_locations_modified = modify_locations(us_h5n1_cases_df, bio_us_data_df)
    # h5n1_us_data_df = h5n1_us_data_df[h5n1_us_data_df['species'].str.contains('Wild', case=False, na=False)]
    # only focus on domestic cases
    wild_h5n1_us_data_df_with_locations_modified = h5n1_us_data_df_with_locations_modified[~h5n1_us_data_df_with_locations_modified['species'].str.contains('Wild', case=False, na=False)]
    wild_h5n1_us_data_df_locations_only = wild_h5n1_us_data_df_with_locations_modified[['latitude', 'longitude']]
    wild_h5n1_us_data_df_locations_only['observed'] = 1
    # load us poultry data
    poultry_locations_df = us_poultry_df[['latitude', 'longitude']]
    poultry_locations_df_locations_modified = modify_locations(poultry_locations_df, bio_us_data_df)
    poultry_locations_df_locations_modified['poultry_observed'] = 1
    # merge bio_us_data with us h5n1 observed cases
    bio_us_data_df_with_hn51_obs = pd.merge(bio_us_data_df, wild_h5n1_us_data_df_locations_only, on=['latitude', 'longitude'], how='outer')
    # merge bio_us_data with us poultry locations data
    bio_us_data_df_with_hn51_obs_and_poultry_obs = pd.merge(bio_us_data_df_with_hn51_obs, poultry_locations_df_locations_modified, on=['latitude', 'longitude'], how='outer')

    # Create spatial blocks BEFORE removing latitude/longitude
    spatial_groups = create_spatial_blocks(bio_us_data_df_with_hn51_obs_and_poultry_obs, lat_col='latitude', lon_col='longitude', block_size=2.0)

    #spatial_groups = spatial_groups['spatial_block'].reset_index(drop=True)

    X = spatial_groups.copy()
    y = spatial_groups['observed']

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=spatial_groups['spatial_block'].reset_index(drop=True)))

    train = spatial_groups.copy()#.iloc[train_idx]
    test_dummy = spatial_groups[spatial_groups.index.isin([test_idx])].copy()
    train['observed'] = [0 if i in test_idx else train['observed'][i] for i in train.index]
    
    train = train.fillna(0)

    # add distance_to_nearest_poultry column
    add_distance_to_poultry_coln(train)
    # add # nearby poultry facilities column
    add_num_of_poultry_coln(train, 2)
    # add distance to infected wild animals
    add_distance_to_infected_wild_animals(train, h5n1_us_data_df_with_locations_modified)
    # add # nearby infected wild animals
    add_num_of_infected_wild_animals(train, h5n1_us_data_df_with_locations_modified, 2)
    # drop poultry_observed column
    train = train.drop(columns=['poultry_observed'])
    # normalize data
    z_normalized_bio_us_data_df = z_normalizing(train,
                                                ['latitude', 'longitude', 'observed','lat_block','lon_block','spatial_block'],
                                                ['latitude', 'longitude', 'observed','lat_block','lon_block','spatial_block'])

    #z_normalized_bio_us_data_df = z_normalized_bio_us_data_df.sample(frac=1).reset_index(drop=True)
    train = z_normalized_bio_us_data_df[z_normalized_bio_us_data_df.index.isin(train_idx)]
    train = train.fillna(0)
    test = z_normalized_bio_us_data_df[z_normalized_bio_us_data_df.index.isin(test_idx)]
    test['observed'] = test_dummy['observed']

    #z_normalized_bio_us_data_df.to_csv("/content/drive/MyDrive/SNOW_FOLDER/Layers/normalized_factors_with_h5nx.csv", index=False)
    # print("done")

    if new_zs is not None:
        train = pd.concat([train, new_zs],ignore_index=True)
        #z_normalized_bio_us_data_df.to_csv("/content/drive/MyDrive/SNOW_FOLDER/Layers/us_bio_and_poultry_and_observ_information_normalized.csv", index=False)

    # get EM us bio data
    # fit the logistic regression model
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    X_train = train.drop(columns= ['latitude', 'longitude', 'observed','lat_block','lon_block','spatial_block'])
    X_test = test.drop(columns= ['latitude', 'longitude', 'observed','lat_block','lon_block','spatial_block'])

    y_train = train['observed']
    y_test = test['observed']


#----------------- MaxEnt ------------------
    # Build logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Coefficients:")
    print(model.coef_)

    print("Intercept:")
    print(model.intercept_)

    # Predictions
    y_pred = model.predict(X_test)

    # IMPORTANT: probability, not hard class, for AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # AUC
    AUC = roc_auc_score(y_test, y_prob)
    print(f"Spatial test AUC = {AUC:.4f}")

    print(f"Accuracy:             {accuracy:.4f}")
# The reviewer asked for all of these:
    precision = precision_score(y_test, y_pred)
    print(f"Precision:            {precision:.4f}")
    recall = recall_score(y_test, y_pred)
    print(f"Recall:               {recall:.4f}")
    f1 = f1_score(y_test, y_pred)
    print(f"F1-score:             {f1:.4f}")
    # Sensitivity = Recall
    sensitivity = recall_score(y_test, y_pred)
    print(f"Sensitivity:          {sensitivity:.4f}")
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("\nConfusion Matrix:")
    print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp)
    print(f"Specificity:          {specificity:.4f}")
    # Balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy:    {balanced_accuracy:.4f}")

    # Brier score (lower is better)
    brier_score = brier_score_loss(y_test, y_prob)
    print(f"Brier Score:          {brier_score:.4f}")
    # Log loss / cross-entropy (lower is better)
    logloss = log_loss(y_test, y_prob)
    print(f"Log Loss:              {logloss:.4f}")

    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1-score:             {f1:.4f}")
    print(f"Sensitivity:          {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"Balanced Accuracy:    {balanced_accuracy:.4f}")
    print(f"Brier Score:          {brier_score:.4f}")
    print(f"Log Loss:              {logloss:.4f}")

    print("\nConfusion Matrix:")
    print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
    # Permutation importance
    calculate_permutation_importance(model, X_train, X_test, y_train, y_test)
    model.densify()

    X = z_normalized_bio_us_data_df.copy()
    if new_zs is not None:
      X = pd.concat([X, new_zs],ignore_index=True)
    # Predictions for ALL observations
    y_prob_all = model.predict_proba(X.drop(columns= ['latitude', 'longitude', 'observed','lat_block','lon_block','spatial_block']))

    z_normalized_bio_us_data_df_pred = deepcopy(X) #z_normalized_bio_us_data_df)
    z_normalized_bio_us_data_df_pred['probability'] = list(y_prob_all[:, 1])
    z_normalized_bio_us_data_df_pred_true = z_normalized_bio_us_data_df_pred[
        (z_normalized_bio_us_data_df_pred['probability'] > 0.1) &
        (z_normalized_bio_us_data_df_pred['latitude'] < 10000) &
        (z_normalized_bio_us_data_df_pred['longitude'] < 10000)]

    # save risk map
    if if_save_risk_map:
        z_normalized_bio_us_data_df_pred_all = z_normalized_bio_us_data_df_pred[(z_normalized_bio_us_data_df_pred['latitude'] < 10000) & (z_normalized_bio_us_data_df_pred['longitude'] < 10000)]
        z_normalized_bio_us_data_df_pred_all.to_csv('/content/drive/MyDrive/SNOW_FOLDER/Layers/risk_map.csv', index=False) #('/content/data/linear_logistic_regression_SVI_result/risk_map.csv')
    show_data_on_map5(z_normalized_bio_us_data_df_pred_true, color='probability')

def log_likelihood(z, max_ent_model):
    """ p(x|z) = 1/1+exp{beta_0 + beta1 * z_1 + .... beta_6 * z_6}

    :param z: a 6-dimensional feature: shape = N X 6
    :param max_ent_model: a maximum entropy model
    :return: log_likelihood of p(x|z) where z is the 6-dimensional latent variable and x is the 1-dimensional binary variable
    """

    #Todo：to uncomment it
    # coeffs = max_ent_model.coef_
    # intercept = max_ent_model.intercept_
    coeffs = torch.tensor([-0.86810891, 1.32568532, 0.44760417, 0.81212959, -0.9136064, -0.75873348, -6.71906164,
                           0.64581913, -3.04195532, -0.38696719]).T
    linear_combination = torch.matmul(z.T, coeffs) - 10.68359288  # shape = N X 1
    likelihoods = 1 / (1 + torch.exp2(-linear_combination))
    log_likelihoods = torch.log2(likelihoods)
    log_likelihood = torch.nansum(log_likelihoods)
    return log_likelihood


# generate z based on the prior on z
def mixed_gaussian_samples_generator(mean_1, sigma_1, mean_2, sigma_2, pi_1, num_samples):
    """

    :param mean_1: a 6-dimensional mean
    :param sigma_1: a 6 X 6 matrix
    :param mean_2: a 6-dimensional mean
    :param sigma_2: a 6 X 6 matrix
    :param pi_1: the probability that a latent variable belongs to cluster1
    :param num_samples: the number of samples to generate
    :return: the number of samples X 6
    """

    D = mean_1.shape[0]
    epsilon_1 = torch.randn(D, num_samples)
    epsilon_2 = torch.randn(D, num_samples)

    L1 = torch.linalg.cholesky(sigma_1)
    L2 = torch.linalg.cholesky(sigma_2)

    samples = pi_1 * torch.matmul(L1, epsilon_1) + (1 - pi_1) * torch.matmul(L2, epsilon_2)  # 6 * 6   6 * 1000 -> 6 * 1000
    samples += pi_1 * mean_1.unsqueeze(1) + (1 - pi_1) * mean_2.unsqueeze(1)       # broadcast 6 * 10000 + 6 * 1 -> 6 * 1000
    return samples


def calculate_proposed_posterior_prob(zs, params):
    mean_1 = params[0]
    sigma_1 = params[1]
    mean_2 = params[2]
    sigma_2 = params[3]
    pi_1 = params[4]

    proposed_multinorm_dist1 = MultivariateNormal(loc=mean_1, covariance_matrix=sigma_1)
    proposed_multinorm_dist2 = MultivariateNormal(loc=mean_2, covariance_matrix=sigma_2)

    n = zs.shape[1]
    res = torch.zeros(n)
    for i in range(n):
        z = zs[:, i]
        proposed_posterior_prob = pi_1 * proposed_multinorm_dist1.log_prob(z) + \
                                  (1 - pi_1) * proposed_multinorm_dist2.log_prob(z)
        res[i] = proposed_posterior_prob

    return res


def calculate_empirical_posterior_prob(zs, empirical_params):
    mean_1 = empirical_params[0]
    sigma_1 = empirical_params[1]
    mean_2 = empirical_params[2]
    sigma_2 = empirical_params[3]
    pi_1 = empirical_params[4]

    empirical_multinorm_dist1 = MultivariateNormal(loc=torch.tensor(mean_1.values), covariance_matrix=torch.tensor(sigma_1.values))
    empirical_multinorm_dist2 = MultivariateNormal(loc=torch.tensor(mean_2.values), covariance_matrix=torch.tensor(sigma_2.values))

    n = zs.shape[1]
    res = torch.zeros(n)
    for i in range(n):
        z = zs[:, i]
        empirical_posterior_prob = pi_1 * empirical_multinorm_dist1.log_prob(z) + \
                                  (1 - pi_1) * empirical_multinorm_dist2.log_prob(z)
        res[i] = empirical_posterior_prob

    return res


def compare_empirical_posterior_prob(zs, empirical_params):
    mean_1 = empirical_params[0]
    sigma_1 = empirical_params[1]
    mean_2 = empirical_params[2]
    sigma_2 = empirical_params[3]
    # pi_1 = empirical_params[4]

    empirical_multinorm_dist1 = MultivariateNormal(loc=torch.tensor(mean_1.values), covariance_matrix=torch.tensor(sigma_1.values))
    empirical_multinorm_dist2 = MultivariateNormal(loc=torch.tensor(mean_2.values), covariance_matrix=torch.tensor(sigma_2.values))

    n = zs.shape[1]
    res = torch.zeros(n)
    for i in range(n):
        z = zs[:, i]
        if empirical_multinorm_dist1.log_prob(z) > empirical_multinorm_dist2.log_prob(z):
            res[i] = 0
        else:
            res[i] = 1

    return res


def calculate_proposed_posterior_prob_by_cluster(zs, params, zs_clustering_result):
    mean_1 = params[0]
    sigma_1 = params[1]
    mean_2 = params[2]
    sigma_2 = params[3]

    proposed_multinorm_dist1 = MultivariateNormal(loc=mean_1, covariance_matrix=sigma_1)
    proposed_multinorm_dist2 = MultivariateNormal(loc=mean_2, covariance_matrix=sigma_2)

    n = zs.shape[1]
    sum = 0
    for i in range(n):
        z = zs[:, i]
        cluster_no = zs_clustering_result[i]
        if cluster_no == 0:
            proposed_posterior_prob = proposed_multinorm_dist1.log_prob(z)
        else:
            proposed_posterior_prob = proposed_multinorm_dist2.log_prob(z)
        sum += proposed_posterior_prob

    return sum


def _find_diagnals(M):
    """
    find out diagonals from a triangular matrix

    :param M: a D X D lower triangular matrix or a D X D upper triangular matrix
    :return: a list of diagonals of M
    """
    D = M.shape[0]
    res = torch.zeros(D)
    for i in range(D):
        res[i] = M[i][i]
    return res


def _help(zs, probs_list, pi_1):
    n = zs.shape[1]
    sum = 0
    for i in range(n):
        probs = probs_list[i]
        sum += probs[0] * pi_1 + probs[1] * (1-pi_1)
    return sum


# step3: define the objective
# Hyperparameters
n_iters = 5000
stepsize = 0.0001
num_samples_per_iter = 500

def objective(params, empirical_params, max_ent_model):
    mean_1 = params[0]
    sigma_1 = params[1]
    mean_2 = params[2]
    sigma_2 = params[3]
    pi_1 = params[4]


    # 1. generate latent variables sample
    zs = mixed_gaussian_samples_generator(mean_1, sigma_1, mean_2, sigma_2, pi_1, num_samples_per_iter)

    # 2. calculate the log-likelihood
    log_probability = log_likelihood(zs, max_ent_model)

    # 3. calculate the norm of logp(zs)
    multinormal_dist = MultivariateNormal(loc=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), covariance_matrix=torch.eye(10))
    n = zs.shape[1]
    log_p_of_zs = 0
    for i in range(n):
        z = zs[:, i]
        log_p_of_zs += multinormal_dist.log_prob(z)

    # # 4. calculate log probability of posterior p(z|x^(i)) ~ N(u_x^(i), sigma_x^(i))
    # probs_list = find_mixed_gaussian_prob_given_z(zs, z_normalized_mean_observed, z_normalized_mean_absent, z_normalized_corr_observed, z_normalized_corr_absent)


    # L1 = torch.linalg.cholesky(sigma_1)
    # L2 = torch.linalg.cholesky(sigma_2)
    # diagonals_of_L1 = _find_diagnals(L1)
    # diagonals_of_L2 = _find_diagnals(L2)

    # log_det_sigma1 = 2 * torch.sum(torch.log2(diagonals_of_L1))
    # log_det_sigma2 = 2 * torch.sum(torch.log2(diagonals_of_L2))
    #
    # sum_of_log_det_sigma = pi_1 * log_det_sigma1 + (1- pi_1) * log_det_sigma2

    # calculate the proposed posterior prob
    zs_clustering_results = compare_empirical_posterior_prob(zs, empirical_params)
    proposed_posterior_pro = calculate_proposed_posterior_prob_by_cluster(zs, params, zs_clustering_results)

    numerator = log_probability + log_p_of_zs - proposed_posterior_pro

    return (-1)*numerator / num_samples_per_iter


# step 4: optimize
# Set up optimizer.
init_mean_1 = torch.zeros(10, requires_grad=True)
init_mean_2 = torch.zeros(10, requires_grad=True)
init_sigma_1 = torch.eye(10, requires_grad=True)
init_sigma_2 = torch.eye(10, requires_grad=True)
pi_1 = torch.tensor(0.03)


params = (init_mean_1, init_sigma_1, init_mean_2, init_sigma_2, pi_1) # use params to generate data
optimizer = torch.optim.SGD(params, lr=stepsize, momentum=0.8)


def update(empirical_params, max_ent_model):
    optimizer.zero_grad()
    loss = objective(params, empirical_params, max_ent_model)
    loss.backward()
    optimizer.step()
    return loss

def callback(loss, t):
    if t % 25 == 0:
        print("Iteration {} lower bound {} with pi_1={}".format(t, loss, params[4]))

# Optimize and print loss in a loop
print("Optimizing variational parameters...")
def svi(empirical_params, max_ent_model):
    for t in range(0, n_iters):
        loss = update(empirical_params, max_ent_model)
        callback(loss, t)
        if loss <= 0:
            break
    return params

def oversampling_k_nearest(k):
    """
    generate the simulated values of explainable variables of the present
    :param k: the k-th nearest neighbors
    :return: the set of simulated values
    """
    df = pd.read_csv("data/layers/us_complete_layers_normalized.csv")
    # step 1: find out the records that are positive and negative respectively
    positive_features = df[df['observed'] == 1].drop(columns=['Unnamed: 0', 'latitude', 'longitude', 'observed'])
    negative_features = df[df['observed'] == 0].drop(columns=['Unnamed: 0', 'latitude', 'longitude', 'observed'])
    # step 2: find out the k-th nearest neighbors in negative_features
    res = None
    for i in range(len(positive_features)):
        pos_feature = positive_features.iloc[i]
        # find out k-th nearest neighbors in negative_features
        dis_vectors = pos_feature - negative_features
        distances = np.linalg.norm(dis_vectors, axis=1)
        distances_sorted_indices = np.argsort(distances)
        k_th_nearest_indices = distances_sorted_indices[:k]
        if res is None:
            res = negative_features.iloc[k_th_nearest_indices]
        else:
            res = pd.concat([res, negative_features.iloc[k_th_nearest_indices]], axis=0, ignore_index=True)
    return res

# learning parameters for each cluster
def get_VAE_for_bio_climatic_data_for_us_independent_vars(max_ent_model):
    """
    it is only for domestic species

    :param: max_ent_model: the maximum entropy model
    :return:
    """
    h5n1_us_data_df_with_locations_modified = modify_locations(us_h5n1_cases_df, bio_us_data_df)
    domestic_h5n1_us_data_df = h5n1_us_data_df_with_locations_modified[~h5n1_us_data_df_with_locations_modified['species'].str.contains('Wild', case=False, na=False)]
    domestic_h5n1_us_data_df_locations_only = domestic_h5n1_us_data_df[['latitude', 'longitude']]
    domestic_h5n1_us_data_df_locations_only['observed'] = 1
    # merge us bio data with h5n1 cases
    bio_us_data_df_with_hn51_obs = pd.merge(bio_us_data_df, domestic_h5n1_us_data_df_locations_only,
                                            on=['latitude', 'longitude'], how='outer')
    bio_us_data_df_with_hn51_obs = bio_us_data_df_with_hn51_obs.fillna(0)
    # load poultry obs
    poultry_locations_df = us_poultry_df[['latitude', 'longitude']]
    poultry_locations_df_modified = modify_locations(poultry_locations_df, bio_us_data_df)
    poultry_locations_df_modified['poultry_observed'] = 1
    # merge us bio data with poultry obs
    bio_us_data_df_with_hn51_obs_and_poultry_obs = pd.merge(bio_us_data_df_with_hn51_obs, poultry_locations_df_modified,
                                                            on=['latitude', 'longitude'], how='outer')
    bio_us_data_df_with_hn51_obs_and_poultry_obs = bio_us_data_df_with_hn51_obs_and_poultry_obs.fillna(0)

    # add distance_to_nearest_poultry column
    add_distance_to_poultry_coln(bio_us_data_df_with_hn51_obs_and_poultry_obs)
    # add #nearby poultry facilities column
    add_num_of_poultry_coln(bio_us_data_df_with_hn51_obs_and_poultry_obs, 2)
    # add distance to infected wild animals
    add_distance_to_infected_wild_animals(bio_us_data_df_with_hn51_obs_and_poultry_obs,
                                          h5n1_us_data_df_with_locations_modified)
    # add #nearby infected wild animals
    add_num_of_infected_wild_animals(bio_us_data_df_with_hn51_obs_and_poultry_obs,
                                     h5n1_us_data_df_with_locations_modified, 2)

    # drop poultry_observed column
    complete_bio_us_data_df = bio_us_data_df_with_hn51_obs_and_poultry_obs.drop(columns=['poultry_observed'])

    # find out the mean and covariance of normalized samples
    complete_bio_us_data_df_to_analyze = complete_bio_us_data_df.drop(columns=['latitude', 'longitude', 'observed'])
    z_score_normalized_complete_bio_us_data_df = \
        z_normalizing(complete_bio_us_data_df, ['latitude', 'longitude', 'observed'],  ['latitude', 'longitude', 'observed'])    # no locations
    z_score_normalized_complete_bio_us_data_df_observed = z_score_normalized_complete_bio_us_data_df[
                                                    z_score_normalized_complete_bio_us_data_df['observed']==1].drop(columns= ['latitude', 'longitude', 'observed'])
    z_score_normalized_complete_bio_us_data_df_absent = z_score_normalized_complete_bio_us_data_df[
        z_score_normalized_complete_bio_us_data_df['observed'] == 0].drop(columns= ['latitude', 'longitude', 'observed'])
    corr_matrix_observed = z_score_normalized_complete_bio_us_data_df_observed.corr()
    means_observed = z_score_normalized_complete_bio_us_data_df_observed.mean()
    corr_matrix_absent = z_score_normalized_complete_bio_us_data_df_absent.corr()
    means_absent = z_score_normalized_complete_bio_us_data_df_absent.mean()

    # start off svi
    # check oout
    print("means_absent", means_absent.shape[0])
    print("corr_matrix_absent", corr_matrix_absent.shape[0])
    print("means_observed", means_observed.shape[0])
    print("corr_matrix_observed", corr_matrix_observed.shape[0])
    params = svi([means_absent, corr_matrix_absent, means_observed,corr_matrix_observed, 0.97], max_ent_model)
    mean_1 = params[0].detach().numpy()
    sigma_1 = params[1].detach().numpy()
    mean_2 = params[2].detach().numpy()
    sigma_2 = params[3].detach().numpy()
    pi_1 = params[4].detach().item()

    print(mean_1)
    print(sigma_1)
    print(mean_2)
    print(sigma_2)
    print(pi_1)

    return mean_2, sigma_2, z_score_normalized_complete_bio_us_data_df.columns

