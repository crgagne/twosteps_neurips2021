# Overview

This repository contains the data and code needed to reproduce the main results from our paper:

Gagne, C., & Dayan, P. (2021). Two steps to risk sensitivity. 35th Conference on Neural Information Processing Systems.

The code is split into two folders, one for the two-step task analyses (Figures 1-4; Supplemental Figure 1) and the other for the p/f/n CVaR simulations (Figures 5-6; Supplemental Figure 2).

# Getting Set Up

First, clone this repository. Then install the necessary python packages in `requirements.txt`.
If you are using conda, you can run the following:

`conda env create -f environment.yml`


# Part 1: Two-Step Task Analyses

## Data from Gillan et al. 2016

The data obtained from Gillan et al. 2016 is contained in `two_step_analyses/data` for convenience. It is divided into two folders, study 1 and study 2, based on the original paper. The current paper combines data across both. The list of included participants (with IDs) from study 1 and 2 can be found in the 'summary_data' files.

## Fitting the CVaR- and mean-models to participants' data

The model fits for individual participants for the CVaR- and mean-models are already contained in `two_step_analyses/results/model_fits/`, but they can be reproduced by calling, for example:

`python fit_model_to_dataset.py --model dcvar_mb_mf_sticky --experiment 1 --n_subjects 0:253 --multi_starts 10 --subj_set no_missing`

In the command above `dcvar_mb_mf_sticky` refers to the CVaR model; this can be changed to `mean_mb_mf_sticky` for the mean model. The experiment input, either `1` or `2` is for different subsets of participants (see note about dataset above); both subsets were used for our analyses.

## Model code

The main code for both models is found in `two_step_analyses/code/model.py`.

## Figures 1-4 and Supplemental Figure 1

Figures 1-4 and Supplemental Figure 1 can be reproduced using the Jupyter notebooks found in `two_step_analyses/notebooks`. These rely on the model fit data found in `two_step_analyses/results`.

# Part 2: CVaR Gridworld Simulations

## Code for calculating CVaR optimal policies

The n/p/f CVaR optimal policies were calculated using dynamic programming (dp); the code for this is contained in `gridworld_simulations/code/dp.py` and `gridworld_simulations/code/dp_support.py`.

The optimal policies are calculated with respect to a Markov Decision Process (MDP), which for our analyses, was specified in the file `gridworld_simulations/code/task_gridworld.py`.

Examples of how to use this code are contained in the Jupyter notebooks that are used to plot the figures (see next section).

## Figures 5-6 and Supplemental Figure 2

These figures can be reproduced using the Jupyter notebooks in `gridworld_simulations/notebooks`.
