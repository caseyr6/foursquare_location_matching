import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def generate_pairs(
    df,
    n_neighbors = 1,
    train = True
):
    '''
    Function takes the train data set as an input.
    
    Creates n_neighbors pairs for each sample is the dataset. These pairwise combinations are generated by rank ordering samples based on distance from
    the current sample, then selecting the top n_neighbors of these.
    
    Returns a dataframe containing these samples, with ~10% being positive matches.
    '''
    
    orig_df = df.copy()
    
    # define and scale the location df
    location_data = df[['latitude', 'longitude']].values
    location_data = StandardScaler().fit_transform(location_data) 
    
    # define and fit the knn
    knn = NearestNeighbors(n_neighbors = min(n_neighbors + 1, len(df)), 
                           algorithm = 'kd_tree',
                           n_jobs = -1)
    knn.fit(location_data)
    
    # generate the neighbours array - index of closest n_neighbors+1 samples to each of the samples in the location data array, including that sample istelf
    neighbors_array = knn.kneighbors(location_data,
                                     return_distance=False)

    # create column dictionaries
    cols_1 = dict(zip(orig_df.columns, [f'{col}_1' for col in orig_df.columns]))
    cols_2 = dict(zip(orig_df.columns, [f'{col}_2' for col in orig_df.columns]))

    # create dataframes needed for final join
    df_1 = orig_df
    idxs = neighbors_array.flatten()
    df_2 = df_1.iloc[idxs].reset_index().rename(columns=cols_2)

    # change structure of df_1
    df_1['orig_index'] = df_1.index
    df_1 = pd.concat(knn.n_neighbors * [df_1],
                     ignore_index=True)
    df_1 = df_1.reset_index().rename(columns = cols_1)
    df_1 = df_1.sort_values(['orig_index', 'index'])
    df_1.drop(columns=['orig_index', 'index'],
              inplace=True)
    df_1.reset_index(drop=True, inplace=True)
    
    # create final df
    df = pd.concat([df_1, df_2],
                   axis=1)
    df = df.loc[df.id_1 != df.id_2].reset_index(drop=True)
    
    # add final match column
    if train:
        df['match'] = df['point_of_interest_1'] == df['point_of_interest_2']
        df.drop(columns=['point_of_interest_1', 'point_of_interest_2'], inplace=True)
    
    return df