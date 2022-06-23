import pandas as pd
import logging
from pathlib import Path
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# import functions from modules
import clean_data
import generate_pairs
import utils



###########################
# define required variables
###########################
'''
variables required for process defined here
'''

root_file_path = r'C:\Users\caseyrya\Dropbox\foursquare_location_matching_data'
log_file_location = 'process_logs'
log_file_name = 'location_matching.log'
data_location = 'data\model_testing'
data_file_name = 'test_raw.csv'



###################
# configure logging
###################
'''
configure the script logger
'''

# define log file path
log_file_path = os.path.join(root_file_path, log_file_location, log_file_name)

# Set log file configuration
logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%d/%m/%Y %I:%M:%S %p',
    level = logging.DEBUG,
    filename = log_file_path,
    filemode = 'a'
)

# clean log file to prevent it growing too large
assert Path(log_file_path).is_file(), f"Supplied `filename` parameter " \
                                          f"`{log_file_name}` is not a file."
if utils._file_len(log_file_path) >= 1010:
    num_lines_to_delete = utils._file_len(log_file_name) - 1000
    utils._file_del_n_lines(log_file_name, num_lines_to_delete)

# create the logger
logger = logging.getLogger()
logger.info('--------------- STARTED -----------------')
logger.info('LOGGING SUCCESSFULLY CONFIGURED\n')



#####################
# read raw train data
#####################
'''
read train.csv file from dropbox
'''

logger.info('READING DATA')
try:
    df = pd.read_csv(os.path.join(root_file_path, data_location, data_file_name))
    logger.info('data read')
except Exception as e:
    logger.exception('data read failed: {}'.format(e))
    logging.shutdown()
    raise SystemExit('Error: {}'.format(e))

logger.info('DATA SUCCESSFULLY READ\n')    



##################
# clean train data
##################
'''
perform basic cleaning operations on the train data set
'''

logger.info('CLEANING DATA')
try:
    df = clean_data.clean_data(df)
    logger.info('data cleaned')
except Exception as e:
    logger.exception('data cleaning failed: {}'.format(e))
    logging.shutdown()
    raise SystemExit('Error: {}'.format(e))

logger.info('DATA SUCCESSFULLY CLEANED\n')    



#####################
# generate pairs data
#####################
'''
generate the required pairs dataset
'''

logger.info('GENERATE PAIRS DATA')
try:
    df = generate_pairs.generate_pairs(df)
    logger.info('pairs data generated')
except Exception as e:
    logger.exception('pairs data generation failed: {}'.format(e))
    logging.shutdown()
    raise SystemExit('Error: {}'.format(e))
    
logger.info('PAIRS DATA SUCCESSFULLY GENERATED\n') 



#########################
# generate modelling data
#########################
'''
generate the required modelling data set
'''



############################
# evaluate model performance
############################
'''
generate model predictions and evaluate performance
'''

# import the model

# generate the predictions

# extract the standard, and comp specific results (add these to log file)



#################
# script complete
#################
'''
end script
'''

logger.info('--------------- FINISHED -----------------\n\n\n')












