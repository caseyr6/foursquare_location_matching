#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features

@author: maxhdarby
"""
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
import geopy.distance
from langdetect import detect, DetectorFactory
import datetime


def detect_(
    x,
    i
):
    
    try:
        op = detect(str(x[f'name_{i}']).replace(str(np.nan), '') + ' ' + str(x[f'address_{i}']).replace(str(np.nan), '') + ' ' + str(x[f'city_{i}']).replace(str(np.nan), '') + ' ' + str(x[f'state_{i}']).replace(str(np.nan), '') + ' ' + str(x[f'country_{i}']).replace(str(np.nan), ''))
    except:
        op = np.nan
    
    return op
    

def fuzzy_match(
    df
):
    
    df['name_ratio'] = df.apply(lambda x: fuzz.ratio(x['name_1'], x['name_2']) if( (pd.isnull(x['name_1'])==False) & (pd.isnull(x['name_2'])==False) ) else np.nan, axis=1)
    df['city_ratio'] = df.apply(lambda x: fuzz.ratio(x['city_1'], x['city_2']) if( (pd.isnull(x['city_1'])==False) & (pd.isnull(x['city_2'])==False) ) else np.nan, axis=1)
    df['state_ratio'] = df.apply(lambda x: fuzz.ratio(x['state_1'], x['state_2']) if( (pd.isnull(x['state_1'])==False) & (pd.isnull(x['state_2'])==False) ) else np.nan, axis=1)
    df['zip_ratio'] = df.apply(lambda x: fuzz.ratio(x['zip_1'], x['zip_2']) if( (pd.isnull(x['zip_1'])==False) & (pd.isnull(x['zip_2'])==False) ) else np.nan, axis=1)
    df['country_ratio'] = df.apply(lambda x: fuzz.ratio(x['country_1'], x['country_2']) if( (pd.isnull(x['country_1'])==False) & (pd.isnull(x['country_2'])==False) ) else np.nan, axis=1)
    df['url_ratio'] = df.apply(lambda x: fuzz.ratio(x['url_1'], x['url_2']) if( (pd.isnull(x['url_1'])==False) & (pd.isnull(x['url_2'])==False) ) else np.nan, axis=1)
    df['categories_ratio'] = df.apply(lambda x: fuzz.ratio(x['categories_1'], x['categories_2']) if( (pd.isnull(x['categories_1'])==False) & (pd.isnull(x['categories_2'])==False) ) else np.nan, axis=1)
    
    df['name_ratio_part'] = df.apply(lambda x: fuzz.partial_ratio(x['name_1'], x['name_2']) if( (pd.isnull(x['name_1'])==False) & (pd.isnull(x['name_2'])==False) ) else np.nan, axis=1)
    df['city_ratio_part'] = df.apply(lambda x: fuzz.partial_ratio(x['city_1'], x['city_2']) if( (pd.isnull(x['city_1'])==False) & (pd.isnull(x['city_2'])==False) ) else np.nan, axis=1)
    df['state_ratio_part'] = df.apply(lambda x: fuzz.partial_ratio(x['state_1'], x['state_2']) if( (pd.isnull(x['state_1'])==False) & (pd.isnull(x['state_2'])==False) ) else np.nan, axis=1)
    df['zip_ratio_part'] = df.apply(lambda x: fuzz.partial_ratio(x['zip_1'], x['zip_2']) if( (pd.isnull(x['zip_1'])==False) & (pd.isnull(x['zip_2'])==False) ) else np.nan, axis=1)
    df['country_ratio_part'] = df.apply(lambda x: fuzz.partial_ratio(x['country_1'], x['country_2']) if( (pd.isnull(x['country_1'])==False) & (pd.isnull(x['country_2'])==False) ) else np.nan, axis=1)
    df['url_ratio_part'] = df.apply(lambda x: fuzz.partial_ratio(x['url_1'], x['url_2']) if( (pd.isnull(x['url_1'])==False) & (pd.isnull(x['url_2'])==False) ) else np.nan, axis=1)
    df['categories_ratio_part'] = df.apply(lambda x: fuzz.partial_ratio(x['categories_1'], x['categories_2']) if( (pd.isnull(x['categories_1'])==False) & (pd.isnull(x['categories_2'])==False) ) else np.nan, axis=1)
    
    return df


def point_dist(
    df
):
    
    df['proximity'] = df.apply(lambda x: geopy.distance.geodesic((x['latitude_1'],x['longitude_1']), (x['latitude_2'],x['longitude_2'])).km, axis=1)
    
    return df


def same_lang(
    df
):
    
    DetectorFactory.seed = 0
    
    df['full_address_1_lang'] = df.apply(lambda x: detect_(x, 1), axis=1)
    df['full_address_2_lang'] = df.apply(lambda x: detect_(x, 2), axis=1)
    df['same_lang'] = np.where((df['full_address_1_lang'].isna()) | (df['full_address_2_lang'].isna()),
                               np.nan,
                               np.where(df['full_address_1_lang'] == df['full_address_2_lang'], 1, 0))
    
    return df


def len_strings(
    df
):
    
    df['len_name_diff'] = df['name_1'].str.len() - df['name_2'].str.len()
    df['len_address_diff'] = df['address_1'].str.len() - df['address_2'].str.len()
    df['len_city_diff'] = df['city_1'].str.len() - df['city_2'].str.len()
    df['len_state_diff'] = df['state_1'].str.len() - df['state_2'].str.len()
    df['len_zip_diff'] = df['zip_1'].str.len() - df['zip_2'].str.len()
    df['len_phone_diff'] = df['phone_1'].str.len() - df['phone_2'].str.len()
    df['len_categories_diff'] = df['categories_1'].str.len() - df['categories_2'].str.len()
    
    return df

def build_features(
    df
):
    
    org_columns = df.columns
    org_columns = org_columns.drop(['id_1','id_2','match'])
    
    df = fuzzy_match(df)
    print('fuzzy match complete @ {}'.format(datetime.datetime.now()))
        
    df = point_dist(df)
    print('point dist complete @ {}'.format(datetime.datetime.now()))
    
    df = same_lang(df)
    print('same lang complete @ {}'.format(datetime.datetime.now()))
    
    df = len_strings(df)
    print('len strings complete @ {}'.format(datetime.datetime.now()))
    
    df = df.drop(org_columns, axis = 1)
    
    return df 