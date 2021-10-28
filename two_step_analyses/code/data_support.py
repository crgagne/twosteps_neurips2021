import pandas as pd
import numpy as np

def get_data(directory, filename, filter_no_response=False):

    currentdirectory = str(directory) + '/' + str(filename)
    data_curr_subj = pd.read_csv(currentdirectory, names = list(range(0,15)))

    # Drop rows with instructions and so on
    data_curr_subj = data_curr_subj.dropna()
    data_curr_subj = data_curr_subj.reset_index(drop=True)

    # Rename columns in df according to README
    data_curr_subj.columns = ["trial_num",
                           "drift 1",
                           "drift 2",
                           "drift 3",
                           "drift 4",
                           "stage 1 response",
                           "stage 1 selected stimulus", # Note: (1/2 - note this is redundant with the response as the stage 1 options do not switch locations)
                           "stage 1 RT",
                           "transition (common = TRUE; rare = FALSE)",
                           "stage 2 response",
                           "stage 2 selected stimulus", # Note: (1/2 - note this is redundant with response as the stage 2 options also do not switch locations)
                           "stage 2 state", # (identity 2 or 3)
                           "stage 2 RT",
                           "reward", # (1= yes; 0=no)
                           "redundant task variable, always set to 1"]

    data_curr_subj = data_curr_subj.drop(columns=["redundant task variable, always set to 1"])

    # Filter out rows with -1 i.e. no response
    if filter_no_response:
        data_curr_subj = data_curr_subj[data_curr_subj['stage 1 response (left/right)'].values != '-1']
        data_curr_subj = data_curr_subj[data_curr_subj['stage 2 response (left/right)'].values != '-1']

    # Adds column with filename
    data_curr_subj['Subj_identity'] = [filename for x in range(data_curr_subj.shape[0])]

    return(data_curr_subj)
