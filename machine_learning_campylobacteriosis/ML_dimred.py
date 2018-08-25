#this is a script which takes a dataframe as a pickle and performs a bake-off between different machine learning techniques
from __future__ import division

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import cPickle as pickle
import time as t
import os.path
import sys
import warnings
import sys
import logging
import argparse
import multiprocessing
from matplotlib.backends.backend_pdf import PdfPages
from math import pi, ceil
from matplotlib import gridspec
from collections import Counter
from multiprocessing import Pool, Array
from functools import partial

from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, LeaveOneOut
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,  classification_report, roc_curve, auc, precision_recall_fscore_support, explained_variance_score 
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.metrics import sensitivity_specificity_support


from matplotlib.path import Path
from matplotlib.spines import Spine
from joblib import Parallel, delayed
import multiprocessing



#this is a local file downloaded from https://github.com/fengwangPhysics/matplotlib-chord-diagram
from matplotlibchord import chordDiagram


"""____________________________________________________________________________HERE COME THE DATA PROCESSING FUNCTIONS_________________________________________________________________________________________"""



def train_test_splitting(df):
    #Split the dataset into training and testing

    print set(df.label)
    X = df.iloc[:, :-1]
    y = list(df.label)
    le = LabelEncoder()
    le.fit( y)
    y = le.transform( y)
    unique_transormed_labels = list(set( y))
    
    global labeldict
    labeldict = dict( zip(  unique_transormed_labels, le.inverse_transform( unique_transormed_labels) ) )
    print "Labels are encoded as follows: ", labeldict
    
    print "Number of samples in complete dataset: ", Counter(y)
    
    #testing = {1: 100, 0: 100, 4: 100, 2: 100, 3: 100}
    #X = SelectKBest(chi2, k=5000).fit_transform(X, y)

    #X, y = RandomOverSampler().fit_sample(X, y)

    #rus = RandomUnderSampler(return_indices=True, ratio = "not minority", random_state = 0)
    #X, y, idx_resampled = rus.fit_sample(X,y)


        
    #print "Number of samples AFTER undersampling: ", Counter(y)
    
    global X_train
    global y_train
    global X_test
    global y_test
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, train_size = 0.75, stratify = y)



    rus = RandomUnderSampler(return_indices=True, ratio = "not minority", random_state = 0)
    X_test, y_test, idx_resampled = rus.fit_sample(X_test,y_test)
   
    


    filename = "NMF_dimred_object.p"
    
    print "STARTING DIMENSIONALITY REDUCTION WITH NMF DUMPING OBJECT IN" , filename
    
    nmf = NMF(n_components = 50, init = 'nndsvd', random_state=0, verbose = True)
    
    nmf.fit( X_train) 
    explained_variance = get_score(nmf, X_train)

    print "RECONSTRUCTION ERROR = ", nmf.reconstruction_err_ 

    print "EXPLAINED VARIANCE RATIO = ", explained_variance 
    
    pickle.dump(nmf, open(filename, "wb"))

    X_train = nmf.transform( X_train)
    X_test = nmf.transform( X_test)

    #X_train, y_train = SMOTE( k_neighbors = 10, kind = "svm" ,n_jobs = args.t).fit_sample(X_train, y_train)
    
    print "Number of samples in training dataset: ", Counter(y_train)

    print "Number of samples in test dataset: ", Counter(y_test)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)



def gridsearch_best_parameters(classifier):
    #tune optimal parameters for each indidivual model
    clf = classifier[0]
    grid_values = classifier[1]
    typeclf = classifier[2]
    
    prefit_switch = 0
    if args.prefitted:

        filename = "./fitted_classifiers/" + typeclf.replace(" ", "_") + "_optimised.p"
        if os.path.isfile(filename):
            prefit_switch = 1
    
    if prefit_switch == 1:
        print "READING IN OPTIMISED CLASSIFIER FROM {}".format(filename)
        optimised_clf = pickle.load( open(filename, "r"))
    
    else:
        print "STARTING TO OPTIMISE {}".format(typeclf)
        
        #scoring methods we will look at
        scoring = [ "accuracy", "f1_macro", "f1_micro", 'precision_micro', 'precision_macro',  'recall_macro', "recall_micro","f1_weighted"]
        
        
        #skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)  # an object to stratify the dataset by
        skf = StratifiedKFold(n_splits=5,  random_state=0)  # an object to stratify the dataset by
        #optimise parameters for f1 micro

        if "Perceptron" in typeclf:
            njobs = 1
        else:
            njobs = args.t
        grid= GridSearchCV(clf, param_grid = grid_values, scoring=scoring, cv=skf, refit = "accuracy", n_jobs = njobs)
        grid.fit(X_train, y_train)
        
        print "For the {} the best hyper-parameters are:".format( typeclf)
        for key, value in grid.best_params_.iteritems():
            print key, "=" ,value
        
        #what is the best parameter setting
        best_param_string = ""
        param_string = ""
        for key, value in grid.best_params_.iteritems():
            best_param_string += "{}  ".format( value)
            param_string += " "
            param_string += key
        
        #get the whole parameter tuning shabang into a dataframe
        index_list = []
        for params_dict in grid.cv_results_["params"]:
            row_header_string = ""
            for param, value in params_dict.iteritems():
                row_header_string += "{}  ".format( value)
            index_list.append(row_header_string)
        scores_df = pd.DataFrame(data = grid.cv_results_, index = index_list)
        
        
        std_error_df = scores_df[ [ 'std_test_accuracy', "std_test_f1_micro", "std_test_precision_micro", "std_test_recall_micro"]]
        scores_df = scores_df[ [ "mean_test_accuracy","mean_test_f1_micro",   "mean_test_precision_micro", "mean_test_recall_micro"]]
        
        
        
        std_error_df.columns = scores_df.columns
        
        #plot the parameter tuning
        plot_parameter_tuning( scores_df, std_error_df ,param_string, best_param_string, typeclf, pdf)
        
        optimised_clf = grid.best_estimator_
        
        if typeclf == "Random Forest Classifier":
            with open( args.n + 'feature_importances_rf.tsv', "w+") as outf:
                feature_importances = sorted(zip(map(lambda x: round(x, 4), optimised_clf.feature_importances_), list(df)),reverse=True)
                feature_importances = feature_importances[ :50 ]
                print "IMPORTANCES COMING" , feature_importances
                for feature in feature_importances:
                    outf.write( "{}\t{}\n".format(feature[ 0], feature[ 1]))
        
        #here we return the classifier object which has been optimised for f1 micro over the hyperparameterspace we have defined previously
        filename = typeclf.replace(" ", "_") +"_optimised.p"
        pickle.dump(optimised_clf, open(filename, 'wb'))
        
        print "FINISHED OPTIMISING {}".format(typeclf)
    
    return (optimised_clf, typeclf)

def classifier_comparison( classifier_list):
    #compare the classifiers in classifier list on test dataset
    labels = list(set( y_test))
    labels_names = map( lambda x: labeldict[ x], labels)
    means_list = []
    std_list = []
    row_names = []
    column_names = [ "       Accuracy", "    Precision / Positive Predictive Value", "Recall / Sensitivity", "F1", "Specificity      ", "Negative Predictive Value   ", "Speed"]
    
    #loop through classifiers
    for classifier in classifier_list:
        clf = classifier[ 0]
        clf_name = classifier[ 1]
        
        print "START TESTING " + clf_name
        
        #try classifier on test dataset
        y_pred = clf.predict( X_test)
        
        #get a (number classes)**2 confusion matrix
        confusion = confusion_matrix(y_test, y_pred, labels = labels)
        
        #get a report featuring the most important scores
        report = classification_report( y_test, y_pred, labels = labels, digits = 2)
        
        
        #plot the confusion matrix as a heatmap and chord diagram
        plot_confusion_matrix( confusion, labels_names, clf_name, report)
        
        n_iterations = 100
        n_size = int(len(X_test) * 1.0)

        print "STARTING PARALLEL BOOTSTRAPPING NOW"
        print "0 ITERARION OUT OF {}".format( n_iterations)
        if clf_name == "Voting Classifier":
            pool = Pool( processes = 1)
        else:
            pool = Pool( processes = args.t)
        iterable = range(n_iterations)
        func =  partial( bootstrapping_results, clf, n_size, labels, n_iterations)
        scores  = pool.map( func, iterable)
        pool.close()
        pool.join()
        scores = np.array(scores)
        
        means = np.mean(scores, axis=0)
        print scores
        print means
        std = np.std(scores, axis=0)
        
        means_list.append(list( means))
        std_list.append( list(std))
        
        row_names.append( clf_name)
        print "FINISHED TESTING " + clf_name
    
    #get a dataframe summarising all scores for all different classifiers and save it
    #scores_list = [value for xy in zip(means_list, std_list) for value in xy]
    
    #get a dataframe summarising all scores for all different classifiers and save it
    #scores_list = [value for xy in zip(means_list, std_list) for value in xy]
    comparison_df = pd.DataFrame( means_list, index = row_names, columns = column_names)
    comparison_df.Speed = [ 1 - (float(i)/max(comparison_df.Speed)) for i in comparison_df.Speed]
    
    comparison_df_2 = pd.DataFrame( std_list, index = row_names, columns = column_names)
    comparison_df_2.Speed = [ 0 for i in comparison_df.Speed]

    comparison_df = comparison_df.round( decimals = 3)
    comparison_df_2 = comparison_df_2.round( decimals = 3)
    comparison_df.to_csv( "{}_classifier_comparison_scores.tsv".format(args.n), sep = "\t")
    
    #plot a star (or radar or spider) plot as a comparison between different classifiers
    plot_star_plot_scores( comparison_df, comparison_df_2,pdf)

def bootstrapping_results(clf, n_size, labels, iterable, n_iterations):
    print "{} ITERARION OUT OF {}\r".format(n_iterations, iterable)
    sys.stdout.write("\r")
    X_boot, y_boot = resample(X_test, y_test, n_samples=n_size, replace = True)
    
    t0 = t.time()

    y_pred = clf.predict( X_boot)

    t1 = t.time()
    
    TP, FP, TN, FN = perf_measure(y_boot, y_pred)
    
    sens_specifity = sensitivity_specificity_support( y_boot, y_pred, average = "micro")
    report = classification_report( y_boot, y_pred, labels = labels, digits = 2)
    
    report = report.split("\n")[ -2]
    averages = report.split()
    
    speed = t1-t0
    if ( FN + TN) != 0:
        NPV = TN / ( FN + TN)
    else:
        NPV = 0
    specificity = float( sens_specifity[ 1])
    
    accuracy = round(accuracy_score( y_boot, y_pred, normalize = True), 2)
    precision = float(averages[ 3])
    recall = float(averages[ 4])
    f1 = float(averages[ 5])
    scores_row = np.array([accuracy,  precision, recall, f1, specificity, NPV, speed])
    return scores_row

def voting_clf( classifier_list):
    classifier_list = [ (x[ 1], x[ 0]) for x in classifier_list]
        
    typeclf = "Voting Classifier"
    print "STARTING TO OPTIMISE {}".format(typeclf)

    clf = VotingClassifier(estimators = classifier_list, n_jobs = args.t)
    clf.fit(X_train, y_train) 
    print "FINISHED OPTIMISING {}".format(typeclf)
    filename = typeclf.replace(" ", "_") +"_optimised.p"
    pickle.dump(clf, open(filename, "wb"))
    return (clf, typeclf)


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)    
    
def get_score(model, data, scorer=explained_variance_score):
    """ Estimate performance of the model on the data """
    prediction = model.inverse_transform(model.transform(data))
    return scorer(data, prediction)



"""____________________________________________________________________________HERE COME THE PLOTTING FUNCTIONS_________________________________________________________________________________________"""





def plot_parameter_tuning( scores_df, std_error_df, parameter ,value, typeclf, pdf):
    #plot the parameter tuning in a simple line plot 
    parameter_space = list(scores_df.index)
    x_dim = len(parameter_space)
    
    
    #for index, row in std_error_df.iterrows():
        #for col, item in enumerate( row):
            #ax.plot( , item )
    fig = plt.figure()
    ax = scores_df.plot( kind = "line",  use_index = True, legend = True, linestyle = '-' ,linewidth = 1.5, subplots = True, figsize = [10, 10])
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for a, column in enumerate(scores_df):
        ax[a].scatter( parameter_space, scores_df[column], marker = "o", alpha = 0.5, edgecolors = "none", zorder=9, facecolor = cycle[a])
        ax[a].fill_between( parameter_space, scores_df[column] - std_error_df[column], scores_df[column] + std_error_df[column], linestyle = "--", linewidth = 1, alpha = 0.15, zorder=1, facecolor = cycle[a])
        
        if column == "mean_test_precision_micro":
        
            for i, txt in enumerate( scores_df[column]):
                if parameter_space[i] == value:
                    y_vals = list(scores_df[column])
                    ax[a].annotate(str(round(txt, 3)), (parameter_space[i], y_vals[ i]))
                    print "and the optimised optimised precision is:".format( round(txt, 3))
                    linemax = y_vals[ i]
                    ax[a].plot( [parameter_space[i], ] ,  y_vals[ i]  , linestyle='-.', color="k", marker='x', markeredgewidth=2, ms=8, zorder=10)
    
    #plt.xlim( list(df.index)[0], list(df.index)[-1])
    
    #highglight the best parameter
    #ax[2].text( value, 0.25,'Optimal f1',rotation=270, color = "k")
    plt.suptitle( "Gridsearch for {} \n over parameter space of {}".format( typeclf, parameter, parameter, value ))
    
    plt.xticks(scores_df.index, scores_df.index, rotation='vertical')
    plt.ylabel('Scorers')
    
    
    plt.xlabel("Value " + parameter)
    for i in range(len(ax)):
        ax[i].set_ylim( [0 ,1])
        ax[i].axvline(x=value, color='k', linestyle=':', alpha = 0.6)
        ax[i].yaxis.grid()
        #ax[i].grid(True)
        #ax[i].set_xlim( [list(df.index)[0], list(df.index)[-1]])
    plt.tight_layout()
    pdf.savefig()
    
            
def plot_confusion_matrix( confusion, labels_names, clf_name, report):
    #plot the confusion matrix in a heatmap and a chorddiagram (or circular plot) and a table
    confusion_df = pd.DataFrame( confusion, index = labels_names, columns = labels_names )
    
    #the heatmap
    plt.figure()
    ax = sns.heatmap(confusion_df, annot=True, cmap = "PuBu",  fmt="d")
    plt.title( "Confusion Matrix for {}".format(clf_name))
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.tight_layout()
    pdf.savefig()
    
    gs = gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[10,1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    plt.suptitle( "Chord diagram of confusion matrix for \n{}".format(clf_name))
    
    
    matrix = confusion_df.as_matrix()
    
    colors = [ (255.0/255, 192.0/255, 203.0/255),  (176.0/255, 196.0/255, 222.0/255),  (60.0/255, 179.0/255, 113.0/255),  (221.0/255, 160.0/255, 221.0/255),  (32.0/255, 178.0/255, 170.0/255),  (205.0/255, 92.0/255,92.0/255),  (255.0/255, 222.0/255, 173.0/255)]
    
    #make the chord diagram
    nodePos = chordDiagram( matrix, ax1, pad=5 ,chordwidth=0.7, width=0.1,  colors=colors)
    prop = dict(fontsize=16*0.8, ha='center', va='center')

    nodes = list(confusion_df)
    for i in range(len(nodes)):
        ax.text(nodePos[i][0], nodePos[i][1], nodes[i], rotation=nodePos[i][2], **prop)
    


    
    report = report.split("\n")
    scores_per_class = {}
    rows = []
    data = []
    


    #make the table
    for i, class_name in enumerate( report[ 2: -3]):
        TP = matrix[i, i] 
        FP = np.sum(matrix, axis=0)[i] - matrix[i, i]  #The corresponding column for class_i - TP
        FN = np.sum(matrix, axis=1)[i] - matrix[i, i] # The corresponding row for class_i - TP
        TN = np.sum(matrix) - TP -FP - FN
        Total = TP + FN
        accuracy = round( (TP + TN) / (TN + FN + TP + FP), 2)
        class_name = class_name.split()
        host = labeldict[int(class_name[ 0])]
        precision = float(class_name[ 1])
        recall = float(class_name[ 2])
        f1 = float(class_name[ 3])
        data.append(  [accuracy, precision, recall, f1 , TP, FP, FN, TN, Total])
        rows.append(host)

    columns = ["Accuracy", "Precision", "Recall","F1", "TP", "FP", "FN", "TN", "Total"]
    
    scores_per_class_df = pd.DataFrame( data = data, columns = columns, index = rows)
    csv_name =  "{}_individual_scores.txt".format(clf_name.replace(" ", "_"))
    scores_per_class_df.to_csv( csv_name)
    
    cell_text = []
    for row in range(len(rows)):
        y_offset = data[row]
        cell_text.append(['{}'.format(x) for x in y_offset])

    table = ax2.table(cellText=cell_text, rowLabels=rows, rowColours=colors, colLabels=columns, loc = "center")
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.2)
    table.set_fontsize(15)
    table.scale(1, 0.8)
    
    ax1.axis('off')
    ax2.axis('off')
    pdf.savefig()
    plt.close()
    
    
def plot_star_plot_scores( df, std_err_df, pdf):
    
    #plot a star plot ( or spider plot or radar plot) to show the tradeoffs between precision, recall, f1 and accuracy between out classifiers
    my_palette = plt.cm.get_cmap("Set2", len(df.index))
    
    fig = plt.figure(figsize = [40, 30])
    gs = gridspec.GridSpec(4, 4)
    #gs.update(wspace= -0.05)
    
    
    for row in range(0, len(df.index)):
        individual_star_plot( row, df, std_err_df, my_palette(row), gs)
    
    #lgd = plt.legend( loc='upper center', bbox_to_anchor=(0.5,-0.1), ncol=2, fancybox=True, shadow = True, fontsize = 6)
    fig.suptitle( "Star plot of Classifier comparison")
    pdf.savefig()
    
def individual_star_plot( row, df, std_err_df, color, gs):
    
    ax = plt.subplot(gs[ row], polar=True)
   
    categories=list(df)
   
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.set_rlabel_position(0)
    
    values = list(df.iloc[row])
    errors = list(std_err_df.iloc[row])
    
    values += values[:1]
    errors += errors[:1]
    ax.plot(angles, values, linewidth=2, linestyle='-', label = df.index[ row], alpha = 1, color = color)
    
    upper_bound = [ df.iloc[ row][x]  + std_err_df.iloc[ row][x] for x in range(len(df.iloc[ row]))]
    upper_bound += upper_bound[:1]
    lower_bound = [ df.iloc[ row][x]  - std_err_df.iloc[ row][x] for x in range(len(df.iloc[ row]))]
    lower_bound += lower_bound[:1]
    
    ax.fill_between(angles, lower_bound,  upper_bound, alpha=0.3, color = color)
    
    plt.title(df.index[ row], size=11, color=color, y=1.1)

    plt.yticks([0, 0.2, 0.4 ,0.6, 0.8], ["0", "0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    plt.tight_layout()

    
    


"""____________________________________________________________________________HERE COMES THE MAIN FUNCTION_________________________________________________________________________________________"""






if __name__ == "__main__":
        warnings.filterwarnings("ignore")
        parser = argparse.ArgumentParser(description='read in pickled pandas dataframe of reduced dimension feature table. Then perform some machine learning on it')
        
        #get the dataframe as a pickle from the command line
        parser.add_argument('-p', type=str, help='pickle of pandas dataframe')
        parser.add_argument('-n', type=str, help='job name')
	parser.add_argument('-t', type=int, help='threads')
	parser.add_argument('-prefitted', action='store_true', help=" If this is activated dont optimise parameters but rather take optimised classifier from pickle")
	args = parser.parse_args()
    
        #read in the dataframe

        print "READING IN DATA"

        df = pd.read_hdf(args.p)
  #      df = pd.read_pickle(args.p)
        
        #split the dataset into training and testing
        
        print "SPLITTING DATA"
        
        train_test_splitting( df)
        
        optimised_classifiers = []
        
        print "STARTING TO OPTIMISE CLASSIFIERS"
        #Define which classifier you want to try and which grid values to use. all classifiers are stored in a tuple in the form (classifier, parameters to optimise on, name of the classifier to put on the plot ). Switch classifiers of by commenting out the line in which it was defined.
        classifier_grid_value_list = [
                       ( KNeighborsClassifier(), {'n_neighbors':  [1, 2, 3, 4, 5, 10, 15, 20, 30, 50,  ] }, "K-Nearest-Neighbour Classifier"),
            (RidgeClassifier(  solver = "auto", class_weight = 'balanced'),  {'alpha': [0, 0.001, 0.01, 0.05, 0.1, 1, 10, 100, 200, 400, 1000 ]}, "Ridge Regression Classifier"),
            ( LinearSVC(  dual = False, max_iter = 300, class_weight = 'balanced'), {'C': [0.0001,0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 3, 5,  8, 10, 20, 40, 100]}, "Linear Support Vector Classifier"),
            ( SVC(  kernel = "rbf", class_weight = 'balanced',  probability=False, gamma='auto'), {'C': [1000, 2000, 5000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000, 10000000, 100000000]}, "Radial Basis Function Support Vector Classifier"),
#            ( SVC(  kernel = "poly", class_weight = 'balanced',  probability=False, gamma='auto'), {'C':[ 100, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 200000, 500000, 1000000, 5000000, 10000000, 100000000]}, "Polynomial Function Support Vector Classifier"),
           ( SVC(  kernel = "sigmoid", class_weight = 'balanced',  probability=False, gamma='auto'),{'C': [1, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]}, "Sigmoid Function Support Vector Classifier"),
          ( GaussianNB( ), {}, "Naive Bayesian Classifier"),
#             ( GaussianProcessClassifier( multi_class = "one_vs_rest", n_jobs= args.t),  {}, "Gaussian Process Classifier"),
            ( QuadraticDiscriminantAnalysis( ), {'reg_param':   [-1.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5]}, "Quadratic Discriminant Classifier"),
     ( DecisionTreeClassifier( class_weight = 'balanced'),  {'max_depth':  [10, 15, 20, 25, 30, 50], 'min_samples_split': [3, 5, 10, 15, 20, 25]}, "Decision Tree Classifier"),
            ( RandomForestClassifier( n_estimators = 1000, class_weight = 'balanced'), {'max_depth':  [5, 10, 15, 20, 30, 50], 'max_features': [ 0.1, 0.2, 0.5, 0.7, 0.9]}, "Random Forest Classifier"),
                ( ExtraTreesClassifier( n_estimators = 1000, class_weight = 'balanced'), {'max_depth':  [5, 10, 15, 20, 30, 40, 50], 'max_features': [5, 10] }, "Extra-Trees Classifier"),
            ( GradientBoostingClassifier( n_estimators = 1000), {'max_depth':  [5, 10, 15, 20 ], 'max_features': [ 5, 10]}, "Gradient Boosting Classifier"),
            ( MLPClassifier(max_iter=1000 , hidden_layer_sizes = (100, 100, 100, 100, 100, 100, 100 )), {'alpha':   [0.001, 0.01, 0.05, 0.1, 1, 10, 20, 30, 50,100]  }, "Relu activated Multilayered Perceptron"),
                (MLPClassifier(activation = "logistic",max_iter=1000 ,  hidden_layer_sizes = (200, 200, 200, 200 )), {'alpha':   [0.001, 0.01, 0.05, 0.1, 1, 10, 20, 30, 50,100]  }, "Logistic activated Multi-layered Perceptron")
       
   #     ( SVC(  kernel = "precomputed", class_weight = 'balanced',  probability=False, gamma='auto'), {'C': [1, 2, 3, 4, 5, 7, 10]}, "Hamming Distance Kernel Support Vector Classifier")
               ]
        
        
        #Go through all classifiers and optimise them on hyperparameters using training dataset
        global pdf
        with PdfPages(args.n + '_classifier_comparison.pdf') as pdf:
            #pool = Pool(int(ceil(args.t / 10)))
            optimised_classifiers =map(gridsearch_best_parameters, classifier_grid_value_list)
            #pool.close()
            #pool.join()
            #voting_clf = voting_clf( optimised_classifiers[: -2])#
            
            print "FINISHED OPTIMISING CLASSIFIERS"

            #compare performance of classifiers on test dataset
            print  "START TESTING CLASSIFIERS"
            
            
            #optimised_classifiers.append( voting_clf)
            
            classifier_comparison( optimised_classifiers)
        
