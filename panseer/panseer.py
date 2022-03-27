#!/usr/bin/env python

# productionize the panseer model build - code taken directly from the panseer jupyter notebooks

# import modules
import argparse
import sys
import os
import glob
import random
import itertools

from statistics import mode
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
    parser.add_argument('-t','--num_trees',help='Number of trees in classifier.',type=int,default=5)
    parser.add_argument('-c','--conf_matrix_file',help='Desired path to the confusion matrix plot.',default='conf_matrix.png')
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

def construct_training_datasets(num_trees:int,samples:list) -> tuple[list,list]:
    """
    Define the indices of the training and validation sets. Returns two lists of lists, training and validation, of
    length num_trees. 

    Parameters:
    -----------
    num_trees : int
        Used to set the random_state using a loop
    samples : list
        Indices to be split into training and validation

    Returns:
    --------
    training_list : list
        List of lists of training indices
    validation_list : list
        List of lists of validation indices
    """
    training_list = []
    validation_list = []
    for random_state in range(num_trees):
        training_epoch, validation_epoch = generate_training_split(
            samples=samples,
            random_state=random_state)
        training_list.append(training_epoch)
        validation_list.append(validation_epoch)
    return training_list, validation_list

def combine_training_dataset_elementwise(*args):
    """
    Combines the elements of each list by index.
    """
    return list(zip(*args))


def build_ensemble_tree(num_trees:int,training_samples:list,metadata_df:pd.DataFrame,df:pd.DataFrame) -> list:
    """
    Build and store each of the models for our forest.

    Parameters:
    -----------
    num_trees : int
        Number of trees in our forest
    training_samples : list
        List of lists with indices of training samples
    metadata_df : pd.DataFrame
        Dataframe containing the sample labels
    df : pd.DataFrame
        Dataframe containing the counts data to be used in training. Markers in rows, samples in columns.

    Returns:
    --------
    lr_forest : list
        Each of the trees in our forest - corresponds to individual logistic regression classifier per index
    """
    lr_forest = []
    for i in range(num_trees):
        epoch_samples = list(itertools.chain.from_iterable(training_samples[i]))
        x = df.loc[:,epoch_samples].transpose()
        y = metadata_df.loc[epoch_samples,'status']

        # Set up LR classifier that will automatically fit the best C value using the training set
        clf = LogisticRegressionCV(
            penalty='l1', 
            solver='liblinear', 
            random_state=i,
            Cs=[1.0, 5.0, 10.0, 50.0, 100.0], 
            cv=3)

        clf.fit(x,y)
        
        lr_forest.append(clf)
        
    return lr_forest

def make_predictions(lr_forest:list,df:pd.DataFrame) -> list:
    """
    Make predictions using the logistic regression forest.

    Parameters:
    -----------
    lr_forest : list
        Contains each of the models.
    df : pd.DataFrame
        Dataframe containing the counts data to be used in training. Markers in rows, samples in columns.

    Returns:
    --------
    predictions : list
        Most common prediction across all models.
    """
    pred_matrix = []
    z_prob_array = np.zeros([len(lr_forest),df.shape[1]])
    for n,model in enumerate(lr_forest):
        preds = model.predict(df.transpose())
        pred_matrix.append(list(preds))
        # grab the prediction probability of class 1 of 2 
        z_prob = model.predict_proba(df.transpose())[:,0]
        z_prob_array[n] = z_prob
    # determine most common prediction
    predictions = [mode(a) for a in zip(*pred_matrix)]
    z_prob_mean = z_prob_array.mean(axis=0)
    # TODO: remove model here - just place holder now to access classes
    return predictions,model,z_prob_mean

def construct_confusion_matrix(y_pred:list,metadata_df:pd.DataFrame,model,confusion_matrix_file:str):
    """
    Constructs confusion matrix of healthy vs cancer predictions

    Parameters:
    -----------
    y_pred : list
        Predictions made by the forest model
    metadata_df : pd.DataFrame
        Dataframe containing the sample labels
    """
    cm = confusion_matrix(metadata_df.status,y_pred,labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    disp.plot()
    plt.savefig('test.png')


# TODO: we should be able to simplify this
def compute_ensemble_performance(z_prob_list,negative_test_list,positive_pre_test_list,positive_post_test_list):
    ensemble_score = pd.concat([
        z_prob_list[index].loc[negative_test_list[index]+positive_pre_test_list[index]+positive_post_test_list[index]] for index in range(1000)],
        axis=1, 
        sort=True).mean(axis=1)
    return ensemble_score

def construct_roc(metadata_df:pd.DataFrame,y_score:np.array,pre_samples:list,post_samples:list,output_file:str,pos_label:str='cancer') -> None:
    """
    Construct ROC curve for predictions.
    """
    metadata_df['y_score'] = y_score # TODO: reevaluate this piece - best solution could come up with for now

    # TODO: can turn this all into a separate function
    leavein_metadata_df = metadata_df.loc[post_samples,:]
    y_true = leavein_metadata_df.status
    y_score = leavein_metadata_df.y_score # TODO: need to come up with a different variable name here

    cur_fpr, cur_tpr, cur_cutoff = roc_curve(y_true,y_score,pos_label=pos_label)
    cur_auc = auc(cur_fpr,cur_tpr)
    
    plt.clf()
    plt.plot(cur_fpr, cur_tpr, label=r'Post-Diagnosis ROC (AUC = %0.2f)' % cur_auc)

    # TODO: can turn this all into a separate function
    leavein_metadata_df = metadata_df.loc[pre_samples,:]
    y_true = leavein_metadata_df.status
    y_score = leavein_metadata_df.y_score

    cur_fpr, cur_tpr, cur_cutoff = roc_curve(y_true,y_score,pos_label=pos_label)
    cur_auc = auc(cur_fpr,cur_tpr)

    plt.plot(cur_fpr, cur_tpr, label=r'Pre-Diagnosis ROC (AUC = %0.2f)' % cur_auc)
    plt.grid()
    # plt.xlim([-0.0, 1.0])
    # plt.ylim([-0.0, 1.0])
    # plt.xticks([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # plt.yticks([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.savefig('roc-curve.png')

    
def generate_prediction_plot():
    pass

def output_predictions():
    pass

def main():
    args = parse_args()

    NUM_TREES = args.num_trees
    
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
    # perform bh correction and reduce df to only markers meeting criteria
    tissue_t_tests_min_by_marker = bh_correction(ttest_df=ttest_df)
    amf_df_reduced_marker = amf_df_orig.loc[tissue_t_tests_min_by_marker.index,tsh_metadata_df.index]
    
    # generate leavein, leaveout sets
    negative_leavein, negative_leavout = generate_training_split(list(tsh_metadata_df.loc[tsh_metadata_df.type=='healthy',:].index))
    positive_post_leavein, positive_post_leaveout = generate_training_split(list(tsh_metadata_df.loc[tsh_metadata_df.type=='post-diagnosis',:].index))
    positive_pre_leavein, positive_pre_leaveout = generate_training_split(list(tsh_metadata_df.loc[tsh_metadata_df.type=='pre-diagnosis',:].index))
    
    # split leavein to training and validation lists for each of negative, post-diagnosis, and pre-diagnosis
    negative_training, negative_validation = construct_training_datasets(
        num_trees=NUM_TREES,
        samples=negative_leavein
    )
    positive_post_training, positive_post_validation = construct_training_datasets(
        num_trees=NUM_TREES,
        samples=positive_post_leavein
    )
    positive_pre_training, positive_pre_validation = construct_training_datasets(
        num_trees=NUM_TREES,
        samples=positive_pre_leavein
    )
    
    # training and validation sets combined
    all_training = combine_training_dataset_elementwise(
        negative_training,
        positive_post_training,
        positive_pre_training
    )
    all_validation = combine_training_dataset_elementwise(
        negative_validation,
        positive_post_validation,
        positive_pre_validation
    )

    # build the ensemble forest
    lr_forest = build_ensemble_tree(
        num_trees=NUM_TREES,
        training_samples=all_training,
        metadata_df=tsh_metadata_df,
        df=amf_df_reduced_marker
    )

    # TODO: remove model piece here - just a placeholder to get the classes
    predictions,model,z_prob_mean = make_predictions(
        lr_forest=lr_forest,
        df=amf_df_reduced_marker
    )

    construct_confusion_matrix(
        y_pred=predictions,
        metadata_df=tsh_metadata_df,
        model=model,
        confusion_matrix_file=args.conf_matrix_file
    )

    construct_roc(
        metadata_df=tsh_metadata_df.loc[amf_df_reduced_marker.columns,:],
        y_score=z_prob_mean,
        pre_samples=negative_leavein+positive_post_leavein,
        post_samples=negative_leavein+positive_pre_leavein,
        pos_label='cancer',
        output_file=args.output_roc_curve_file
    )

if __name__ == '__main__':
    main()
