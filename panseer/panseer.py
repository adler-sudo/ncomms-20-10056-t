#!/usr/bin/env python

# productionize the panseer model build - code taken directly from the panseer jupyter notebooks

# import modules
import argparse
import sys
import os
import glob
import random

import pandas as pd
import numpy as np
import scipy
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.multitest import multipletests

import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Predict cancer given biopsy sample.')
    parser.add_argument('-f','--input_file',help='Matrix containing counts.')
    parser.add_argument('-r','--output_roc_curve_file',help='Desired path of ROC curve plot.')
    parser.add_argument('-p','--output_prediction_file',help='Desired path to the output production table.')
    parser.add_argument('-m','--metadata_file',help='Pre and post diagnosis metadata.')
    parser.add_argument('-b','--biochain_metadata_file',help='Biochain metadata file.')
    args = parser.parse_args()
    return args

class PanSeer:
    """
    PanSeer model build
    """
    def __init__(self):
        pass

    def train_model(self):
        pass

    def predict(self):
        pass

def process_metadata(input_file):
    metadata_df = pd.read_csv(input_file,sep='\t',header=0,index_col=0)
    return metadata_df

def process_data(input_file):
    amf_df_orig = pd.read_csv(input_file, sep="\t", header=0, index_col=0)
    amf_df_orig = remove_empty_markers(amf_df_orig)
    return amf_df_orig

def remove_empty_markers(df):
    """
    Removes markers that are missing in at least one sample.
    """
    # from original jupyter notebook - may need to refine but seems like we should be able to just dropna rather than defining all columns
    # amf_df_orig = df.copy()
    # amf_df_orig = amf_df_orig.loc[amf_df_orig[healthy_samples+post_diagnosis_samples+pre_diagnosis_samples].dropna(how='any').index]

    df = df.dropna(how='any')
    return df

# TODO: why can't we just use the sklearn train_test_split here?
# TODO: include different split sizes here
def generate_training_split(samples:list,random_state:int=0):
    """
    Splits the designated samples into leavein and leaveout.
    """
    random.seed(random_state)
    random.shuffle(samples)
    leavein = list(samples[:int(len(samples)/2)])
    leaveout = list(samples[int(len(samples)/2):])
    return leavein, leaveout

# TODO: probably more appropriate name would be identify_significant_markers
def conduct_ttest(df,ttest_df,healthy_samples,cancer_samples,column_name):
    """
    Calculates pvalue from ttest and outputs to dataframe.
    """
    pval = stats.ttest_ind(
        df.loc[:,healthy_samples].T,
        df.loc[:,cancer_samples].T,axis=0,
        equal_var=True,
        nan_policy='omit').pvalue
    
    ttest_df[column_name] = pval
    return ttest_df

def bh_correction(ttest_df):
    """
    Performs benjamini-hochberg correction
    """
    _, pvals_corrected, _, _ = multipletests(
        ttest_df.min(axis=1).fillna(1).values,
        alpha=0.05,
        method='fdr_bh')

    # TODO: i think we can remove that iloc portion
    # TODO: may need to look into exactly what this is trying to accomplish
    tissue_t_tests_min_by_marker = ttest_df.loc[pvals_corrected <= 0.05].iloc[:,:4].min(axis=1).copy()
    return tissue_t_tests_min_by_marker

def set_plot_characteristics():
    sns.set()
    sns.set_context('notebook', font_scale=1.2)
    sns.set_style('white')

# TODO: we should be able to simplify this
def compute_ensemble_performance(z_prob_list,negative_test_list,positive_pre_test_list,positive_post_test_list):
    ensemble_score = pd.concat([
        z_prob_list[index].loc[negative_test_list[index]+positive_pre_test_list[index]+positive_post_test_list[index]] for index in range(1000)],
        axis=1, 
        sort=True).mean(axis=1)
    return ensemble_score

def generate_roc_curve():
    pass

def generate_prediction_plot():
    pass

def output_predictions():
    pass

def main():
    # set random seed
    args = parse_args()
    
    amf_df_orig = process_data(args.input_file)
    tsh_metadata_df = process_metadata(args.metadata_file)
    bc_metadata_df = process_metadata(args.biochain_metadata_file)

    # generate ttest df which will identify significant markers (compares healthy to cancer for each tissue)
    ttest_df = pd.DataFrame(index=amf_df_orig.index)

    # TODO: maybe move this to function as well
    for cancer_tissue in bc_metadata_df.tissue.unique():
        # define the healthy and cancerous samples
        healthy_samples = bc_metadata_df.loc[(bc_metadata_df.status=='healthy') & (bc_metadata_df.tissue==cancer_tissue),:].index
        cancer_samples = bc_metadata_df.loc[(bc_metadata_df.status=='cancer') & (bc_metadata_df.tissue==cancer_tissue),:].index
        
        ttest_df = conduct_ttest(
            df=amf_df_orig,
            ttest_df=ttest_df,
            healthy_samples=healthy_samples,
            cancer_samples=cancer_samples,
            column_name=cancer_tissue
        )
    # perform bh correction and identify minimum pval across ALL tissues?
    tissue_t_tests_min_by_marker = bh_correction(ttest_df=ttest_df)
    
    # generate leavein, leaveout sets
    negative_leavein, negative_leavout = generate_training_split(list(tsh_metadata_df.loc[tsh_metadata_df.type=='healthy',:].index))
    positive_post_leavein, positive_post_leaveout = generate_training_split(list(tsh_metadata_df.loc[tsh_metadata_df.type=='post-diagnosis',:].index))
    positive_pre_leavein, positive_pre_leaveout = generate_training_split(list(tsh_metadata_df.loc[tsh_metadata_df.type=='pre-diagnosis',:].index))
    
    # build ensemble model
    negative_test_list = []
    positive_post_test_list = []
    positive_pre_test_list = []
    z_prob_list = []

    # Split the leave-in set into training and test set
    for random_state in range(5):
        random.seed(random_state)

        # TODO: would be better to store each individual predictor in a list then use to predict on all desired samples        
        # Set up LR classifier that will automatically fit the best C value using the training set
        clf = LogisticRegressionCV(penalty='l1', solver='liblinear', random_state=random_state,
                                Cs=[1.0, 5.0, 10.0, 50.0, 100.0], cv=3)
        
        negative_training, negative_test = generate_training_split(
            samples=negative_leavein,
            random_state=random_state)
        positive_post_training, positive_post_test = generate_training_split(
            samples=positive_post_leavein,
            random_state=random_state)
        positive_pre_training, positive_pre_test = generate_training_split(
            samples=positive_pre_leavein,
            random_state=random_state)
        
        positive_training = positive_post_training+positive_pre_training
        positive_test = positive_post_test+positive_pre_test

        all_training = negative_training + positive_training
        
        # Build the model and set parameters using the training set
        x = amf_df_orig.loc[tissue_t_tests_min_by_marker.index, all_training].transpose()
        y = tsh_metadata_df.loc[all_training,'status']
        clf.fit(x,y)

        # Compute the accuracy of this classifier on the leave-in and leave-out sets
        z = clf.predict(amf_df_orig.loc[tissue_t_tests_min_by_marker.index, :].transpose())
        z_prob = clf.predict_proba(amf_df_orig.loc[tissue_t_tests_min_by_marker.index, :].transpose())[:, 0]
        
        # Store the results
        negative_test_list.append(negative_test)
        positive_post_test_list.append(positive_post_test)
        positive_pre_test_list.append(positive_pre_test)
        z_prob_list.append(z_prob)

    # TODO: the generation of these doesn't exactly make sense to me
    # TODO: basically we are just trying to grab the scores and plot
    # ensemble_score_leavein = compute_ensemble_performance(
    #     z_prob_list=z_prob_list,
    #     negative_test_list=negative_test_list,
    #     positive_pre_test_list=positive_pre_test_list,
    #     positive_post_test_list=positive_post_test_list
    # )
    # TODO: THIS IS WHERE YOU LEFT OFF
    # ensemble_score_leaveout = compute_ensemble_performance(
    #     z_prob_list=z_prob_list,
    #     negativ
    # )
        
    print(z_prob_list)
if __name__ == '__main__':
    main()
