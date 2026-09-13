import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

# SVI FUNCTIONS
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from markdown.util import deprecated
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
# from scipy.stats import multivariate_normal
from torch.distributions import MultivariateNormal, Normal
from scipy.stats import multivariate_normal

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# SVI FUNCTIONS
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from markdown.util import deprecated
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
# from scipy.stats import multivariate_normal
from torch.distributions import MultivariateNormal, Normal


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


#---------------------------------------------------------------------
# define the likelihood model p(x|z)
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

def svi(empirical_params, max_ent_model):
    for t in range(0, n_iters):
        loss = update(empirical_params, max_ent_model)
        callback(loss, t)
        if loss <= 0:
            break
    return params

# K-nearest neighbors uppersampling Algorithm
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
    wild_h5n1_us_data_df = h5n1_us_data_df_with_locations_modified[~h5n1_us_data_df_with_locations_modified['species'].str.contains('Wild', case=False, na=False)]
    wild_h5n1_us_data_df_locations_only = wild_h5n1_us_data_df[['latitude', 'longitude']]
    wild_h5n1_us_data_df_locations_only['observed'] = 1
    # merge us bio data with h5n1 cases
    bio_us_data_df_with_hn51_obs = pd.merge(bio_us_data_df, wild_h5n1_us_data_df_locations_only,
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

    # p(x=1|z) = p(z|x = 1) * p(x=1) / p(z)
    posterior = multivariate_normal.pdf(z_score_normalized_complete_bio_us_data_df.drop(
        columns=['latitude', 'longitude', 'observed']), mean=mean_2, cov=sigma_2)

    return mean_2, sigma_2, z_score_normalized_complete_bio_us_data_df.columns

#--------------------------------------------------------

# Augment observed zs using SVI model
mean_2, sigma_2, colns = get_VAE_for_bio_climatic_data_for_us_independent_vars(None)
# use mean_2 and sigma_2 to generate 18000 zs that have observed == 1
D = mean_2.shape[0]
new_samples_to_generate_n = 500 #1000 #18000
epsilon_1 = np.random.multivariate_normal(np.zeros(D), np.eye(D), new_samples_to_generate_n).T
L = np.linalg.cholesky(sigma_2)
new_zs_observed = np.matmul(L, epsilon_1)
mean_2_reshaped = mean_2[:, np.newaxis]
new_zs_observed += mean_2_reshaped

# use newly generated zs that have observed == 1 to improve max entropy model
# add columns for newly generated zs
i = 0
new_zs_dict = {}
for key in list(colns):
    if key in ['latitude', 'longitude', 'observed', 'likelihood']:
        continue
    new_zs_dict[key] = new_zs_observed[i,:]
    i += 1

new_zs_dict['observed'] = [1 for i in range(new_samples_to_generate_n)]
new_zs_dict['latitude'] = [10000 for i in range(new_samples_to_generate_n)]
new_zs_dict['longitude'] = [10000 for i in range(new_samples_to_generate_n)]
simulated_present_samples = pd.DataFrame(new_zs_dict)


simulated_present_samples.to_csv("./SVI.csv", index=False)


#---------------------------------------------------------------------
