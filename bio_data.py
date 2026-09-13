import pandas as pd
from tqdm import tqdm

def filter_for_us_bio_data(folder_of_bio_vars, drop_colinear_vars=False):
    aggregate_vio_vars_df = None
    for i in tqdm(range(1, 20)):
        bio_var_df = pd.read_csv(folder_of_bio_vars + "_" + str(i) + ".csv")

        bio_var_df = bio_var_df.rename(columns={'longitude':'x', 'latitude':'y'})

        # x is longitude, y is latitude
        bio_var_usd_df = bio_var_df[(bio_var_df['x'] > -125) & (bio_var_df['x'] < -67) & (bio_var_df['y'] > 24) & (bio_var_df['y'] < 50)]
   #     print(aggregate_vio_vars_df)
        if aggregate_vio_vars_df is None:
            aggregate_vio_vars_df = bio_var_usd_df
            aggregate_vio_vars_df = aggregate_vio_vars_df.rename(columns= {'value': 'wc2.1_10m_bio_'+str(i)})
        else:
            aggregate_vio_vars_df = pd.merge(aggregate_vio_vars_df, bio_var_usd_df, on=['x', 'y'], how='outer')
            aggregate_vio_vars_df = aggregate_vio_vars_df.rename(columns= {'value': 'wc2.1_10m_bio_'+str(i)})

    aggregate_vio_vars_df = aggregate_vio_vars_df.dropna()
    aggregate_vio_vars_df = aggregate_vio_vars_df.reset_index(drop=True)
    aggregate_vio_vars_df = aggregate_vio_vars_df.rename(columns={'x': 'longitude', 'y': 'latitude'})

    if drop_colinear_vars:

        # remove collinearity
        # Calculate correlation matrix
        aggregate_vio_vars_df_wo_locations = aggregate_vio_vars_df.drop(columns=['latitude', 'longitude'])
        corr_matrix = aggregate_vio_vars_df_wo_locations.corr()
        # Set the threshold for correlation
        threshold = 0.7
        # Find pairs of correlated features
        corr_pairs = corr_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
        corr_pairs = corr_pairs[corr_pairs['level_0'] != corr_pairs['level_1']]
        # Select features to drop
        to_drop = set()
        for i in range(len(corr_pairs)):
            if corr_pairs.iloc[i, 2] > threshold:
                if corr_pairs.iloc[i, 0] not in to_drop and corr_pairs.iloc[i, 1] not in to_drop:
                    to_drop.add(corr_pairs.iloc[i, 1])

        # Drop the selected features
        aggregate_vio_vars_df_reduced = aggregate_vio_vars_df.drop(columns=to_drop)
        return aggregate_vio_vars_df_reduced
    else:
        return aggregate_vio_vars_df

folder_of_bio_vars = "./WorldClim_CSV/wc2.1_10m_bio"
bio_us_data_df = filter_for_us_bio_data(folder_of_bio_vars, drop_colinear_vars=True)
bio_us_data_df.to_csv('./bio_us_data_df.csv', index=False)
