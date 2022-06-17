import re

def clean_text_field(
    df,
    col
):
    '''
    Function takes a dataframe and the object type column that requires cleaning as input.
    
    Two string cleaning operations are applied.
    
    Returns the original dataframe with the provided column cleaned.
    '''
    
    df[col] = df[col].str.lower() # map all text to lower case
    df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', str(x))) # remove any character that isn’t a word or a whitespace
    
    return df


def clean_data(
    df
):
    '''
    Function takes the raw train dataframe as input.
    
    Applies various cleaning operations.
    
    Returns a cleaned version of the train dataframe.
    '''
    
    for col in df.columns:
    
        # text cleaning function
        if col in ['name','address','city','state','zip','country','url','categories']:
            df = clean_text_field(df, col)
        else:
            pass

        # removing spaces
        if col in ['zip']:
            df[col] = df[col].str.replace(' ','')
        else:
            pass

        # rounding lat / lon values
        if col in ['latitude','longitude']:
            df[col] = round(df[col], 6)
        else:
            pass
    
    return df