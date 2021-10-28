# basics
import sys
import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import pickle
import datetime

sys.path.append('code')

# my imports
from  data_support import get_data
from model import fit_model

def main():
    '''
    Fitting the two main models to experiment 1 data:
        python fit_model_to_dataset.py --model dcvar_mb_mf_sticky --experiment 1 --n_subjects 0:253 --multi_starts 10 --subj_set no_missing
        python fit_model_to_dataset.py --model mean_mb_mf_sticky --experiment 1 --n_subjects 0:253 --multi_starts 10 --subj_set no_missing

    Fitting the two main models to experiment 2 data:
        python fit_model_to_dataset.py --model dcvar_mb_mf_sticky --experiment 2 --n_subjects 0:539 --multi_starts 10 --subj_set no_missing
        python fit_model_to_dataset.py --model mean_mb_mf_sticky --experiment 2 --n_subjects 0:539 --multi_starts 10 --subj_set no_missing
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mean_mb_mf_sticky')
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--n_subjects', type=str, default='0:1')
    parser.add_argument('--multi_starts', type=str, default=4)
    parser.add_argument('--subj_set', type=str, default='no_missing')
    parser.add_argument('--method', type=str, default='L-BFGS-B')

    args = parser.parse_args()

    # load meta-data
    if args.experiment==1:
        directory = 'data/all_twostep_data_study1'
    elif args.experiment==2:
        directory = 'data/all_twostep_data_study2'

    # choose set of participants to fit
    if args.subj_set=='no_missing':
        subj_dataframe = pd.read_csv('data/summary_data_no_missing_trials_exp'+str(args.experiment)+'.csv',index_col=0)
        potentially_missing_trials = False

    # print how participnts there are in the set
    filenames = [ID + '.csv' for ID in subj_dataframe['subj.x'].values]
    print(len(filenames))

    # get slice of participants
    slc = slice(*map(int, args.n_subjects.split(':')))

    # loop over participants
    for filename in filenames[slc]:

        # get participants data
        trial_data = get_data(directory,filename)
        ID = trial_data['Subj_identity'][0].split('.csv')[0]

        # get data into correct form
        data={}
        data['stage1_choices'] = (trial_data['stage 1 response'].values=='right').astype('int')
        data['stage2_choices'] = (trial_data['stage 2 response'].values=='right').astype('int')
        data['stage2_states'] = (trial_data['stage 2 state'].values-1).astype('int') # note the minus 1 here
        data['outcomes'] = (trial_data['reward'].values).astype('int')

        if potentially_missing_trials:
            data['stage1_choices'][trial_data['stage 1 response'].values=='-1']=-1
            data['stage2_choices'][trial_data['stage 2 response'].values=='-1']=-1

        # set up saving
        savedir = 'results/model_fits/experiment'+str(args.experiment)+'_'+args.model+'/'
        savename = savedir+ID+'.pkl'

        # check if saved file already exists
        if (not os.path.exists(savename)) or (args.subj_set=='refits'):

            try:
                # fit model (with graceful failure)
                print('fitting data for ... '+filename)

                fit_results = fit_model(data,model_name=args.model,multi_starts=int(args.multi_starts),method=args.method)

                # Save
                if not os.path.isdir(savedir):
                    os.mkdir(savedir)

                with open(savename, 'wb') as f:
                        pickle.dump(fit_results, f)
            except:
                print('failed for ... '+filename)

        else:
            print('already fit'+filename)


if __name__=='__main__':
    main()
