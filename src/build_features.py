#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features

@author: maxhdarby
"""
from fuzzywuzzy import fuzz
import geopy.distance
from langdetect import detect, DetectorFactory


def fuzzy_match(
    df
    ):
    '''
    add function description here
    '''
    df = df.fillna('')
    for i in df.index:
        df.loc[i,'name_ratio'] = fuzz.ratio(df.loc[i,'name_1'],df.loc[i,'name_2'])
        df.loc[i,'city_ratio'] = fuzz.ratio(df.loc[i,'city_1'],df.loc[i,'city_2'])
        df.loc[i,'state_ratio'] = fuzz.ratio(df.loc[i,'state_1'],df.loc[i,'state_2'])
        df.loc[i,'zip_ratio'] = fuzz.ratio(df.loc[i,'zip_1'],df.loc[i,'zip_2'])
        df.loc[i,'country_ratio'] = fuzz.ratio(df.loc[i,'country_1'],df.loc[i,'country_2'])
        df.loc[i,'url_ratio'] = fuzz.ratio(df.loc[i,'url_1'],df.loc[i,'url_2'])
        df.loc[i,'categories_ratio'] = fuzz.ratio(df.loc[i,'categories_1'],df.loc[i,'categories_2'])
    
        df.loc[i,'name_ratio_part'] = fuzz.partial_ratio(df.loc[i,'name_1'],df.loc[i,'name_2'])
        df.loc[i,'city_ratio_part'] = fuzz.partial_ratio(df.loc[i,'city_1'],df.loc[i,'city_2'])
        df.loc[i,'state_ratio_part'] = fuzz.partial_ratio(df.loc[i,'state_1'],df.loc[i,'state_2'])
        df.loc[i,'zip_ratio_part'] = fuzz.partial_ratio(df.loc[i,'zip_1'],df.loc[i,'zip_2'])
        df.loc[i,'country_ratio_part'] = fuzz.partial_ratio(df.loc[i,'country_1'],df.loc[i,'country_2'])
        df.loc[i,'url_ratio_part'] = fuzz.partial_ratio(df.loc[i,'url_1'],df.loc[i,'url_2'])
        df.loc[i,'categories_ratio_part'] = fuzz.partial_ratio(df.loc[i,'categories_1'],df.loc[i,'categories_2'])
    return df


def point_dist(
    df
    ):
    '''
    add function description here
    '''
    for i in df.index:
        coord_1 = (df.loc[i,'latitude_1'],df.loc[i,'longitude_1'])
        
        coord_2 = (df.loc[i,'latitude_2'],df.loc[i,'longitude_2'])
        
        df['proximity'] = geopy.distance.geodesic(coord_1,coord_2).km
    return df

def same_lang(
    df
    ):
    DetectorFactory.seed = 0
    df['same_lang'] = 0
    df = df.fillna('')
    
    for i in df.index:
        full_address_1 = detect(df.loc[i,'name_1'] + ' ' + df.loc[i,'address_1'] + ' ' + df.loc[i,'city_1'] + ' ' + df.loc[i,'state_1'] + ' ' + df.loc[i,'country_1'])
        full_address_2 = detect(df.loc[i,'name_2'] + ' ' + df.loc[i,'address_2'] + ' ' + df.loc[i,'city_2'] + ' ' + df.loc[i,'state_2'] + ' ' + df.loc[i,'country_2'])
        if (full_address_1 == full_address_2):
            df.loc[i,'same_lang'] = 1
            df.loc[i,'lang_addr_1'] = full_address_1
            df.loc[i,'lang_addr_2'] = full_address_2
            
    
    return df

def len_strings(
        df
        ):
    df = df.fillna('')
    df['len_name_1'] = df['name_1'].str.len()
    df['len_name_2'] = df['name_2'].str.len()
    df['len_address_1'] = df['address_1'].str.len()
    df['len_address_2'] = df['address_2'].str.len()
    df['len_city_1'] = df['city_1'].str.len()
    df['len_city_2'] = df['city_2'].str.len()
    df['len_state_1'] = df['state_1'].str.len()
    df['len_state_2'] = df['state_2'].str.len()
    df['len_zip_1'] = df['zip_1'].str.len()
    df['len_zip_2'] = df['zip_2'].str.len()
    df['len_phone_1'] = df['phone_1'].str.len()
    df['len_phone_2'] = df['phone_2'].str.len()
    return df

def build_features(
        df
        ):
    org_columns = df.columns
    org_columns = org_columns.drop(['id_1','id_2','MATCH'])
    df = fuzzy_match(df)
    df = point_dist(df)
    df = same_lang(df)
    df = len_strings(df)
    
    df = df.drop(org_columns, axis = 1)
    return df
    