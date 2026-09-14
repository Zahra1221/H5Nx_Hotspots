import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.inspection import permutation_importance
import plotly
import plotly.express as px
import plotly.io as pio
from scipy.stats import multivariate_normal

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

simulated_present_samples = pd.read_csv('./ed.csv')

#---------------------------------------------------------------------
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
    
    wild_h5n1_us_data_df_with_locations_modified = h5n1_us_data_df_with_locations_modified[~h5n1_us_data_df_with_locations_modified['species'].str.contains('Wild', case=False, na=False)]
    wild_h5n1_us_data_df_locations_only = wild_h5n1_us_data_df_with_locations_modified[['latitude', 'longitude']]
    wild_h5n1_us_data_df_locations_only['observed'] = 1
    # merge bio_us_data with us h5n1 observed cases
    bio_us_data_df_with_hn51_obs = pd.merge(bio_us_data_df, wild_h5n1_us_data_df_locations_only, on=['latitude', 'longitude'], how='outer')
    # merge bio_us_data with us poultry locations data
    bio_us_data_df_with_hn51_obs = pd.merge(bio_us_data_df_with_hn51_obs, poultry_locations_df_locations_modified, on=['latitude', 'longitude'], how='outer')

    #-------------------------------
    # Create spatial blocks BEFORE removing latitude/longitude
    bio_us_data_df_with_hn51_obs = create_spatial_blocks(bio_us_data_df_with_hn51_obs, lat_col='latitude', lon_col='longitude', block_size=2.0)
    spatial_groups = assign_spatial_folds (bio_us_data_df_with_hn51_obs)

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
        #-------------------------------

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
        # Calibration measures
        # -----------------------------
        # Brier score (lower is better)
        brier_score = brier_score_loss(y_test, y_prob)
        # Log loss / cross-entropy (lower is better)
        logloss = log_loss(y_test, y_prob)
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
        print(f"Brier Score:          {brier_score:.4f}")
        print(f"Log Loss:              {logloss:.4f}")

        print("\nConfusion Matrix:")
        print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
        #----------------------------
    return fold_acc, fold_auc, curves_auc


#---------------------------------------------------------------------
print('MaxEnt (Logistic Regression) with SVI:')
fold_acc, fold_auc, curves_auc = max_ent_for_us_5_fold (simulated_present_samples)

mean_acc = np.mean(fold_acc)
sd_acc = np.std(fold_acc, ddof=1)

mean_auc = np.mean(fold_auc)
sd_auc = np.std(fold_auc, ddof=1)

print('Accuracies: ', fold_acc, 'AUC: ', fold_auc)

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

roc_5_fold (fold_roc_data, fold_auc, model_name='LR with ED - 5-Fold Spatial Cross-Validation')

#----------------------------------------------
def max_ent_for_us(new_zs=None):
    # modify locations
    h5n1_us_data_df_with_locations_modified = modify_locations(us_h5n1_cases_df, bio_us_data_df)
    # h5n1_us_data_df = h5n1_us_data_df[h5n1_us_data_df['species'].str.contains('Wild', case=False, na=False)]
    
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

    #-------------------------------
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

    if new_zs is not None:
        train = pd.concat([train, new_zs],ignore_index=True)
        #z_normalized_bio_us_data_df.to_csv("./Layers/us_bio_and_poultry_and_observ_information_normalized.csv", index=False)

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

    # Predictions
    y_pred = model.predict(X_test)
    # IMPORTANT: probability, not hard class, for AUC
    y_prob = model.predict_proba(X_test)[:, 1]

    # Permutation importance
    print("Permutation Importance:")
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

    # show on the map
    #z_normalized_bio_us_data_df_pred_true['type'] = [str(n) for n in z_normalized_bio_us_data_df_pred_true['probability']]
    show_data_on_map(z_normalized_bio_us_data_df_pred_true, color='probability')

#    return max_ent_model
#----------------------------------------------
max_ent_for_us(simulated_present_samples)
