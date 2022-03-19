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
    # amf_df = df.copy()
    # amf_df = amf_df.loc[amf_df[healthy_samples+post_diagnosis_samples+pre_diagnosis_samples].dropna(how='any').index]

    df = df.dropna(how='any')
    return df

# TODO: why can't we just use the sklearn train_test_split here?
def generate_training_split(samples):
    """
    Splits the designated samples into leavein and leaveout.
    """
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

def generate_roc_curve():
    pass

def generate_prediction_plot():
    pass

def output_predictions():
    pass

def main():
    # set random seed
    random.seed(128)

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
    
    # TODO: this is where you left off
    negative_leavein, negative_leavout = generate_training_split(list(tsh_metadata_df.index))
    print(negative_leavein)

if __name__ == '__main__':
    main()
