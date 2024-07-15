"""
Polymer IR -- ML

util_functions.py 
"""
#%% PACKAGES / SETUP
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import mod.custom_plotter as cp
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, r_regression, f_regression

import sys
sys.path.append("..")

def init_plot(default = False, size = 'half', page_width_cm = 13.858):
    if size == 'full':
        cp.init_plot(scl_a4=1, page_lnewdth_cm=page_width_cm, figsze=[12.8,4.8*(4/3)],lnewdth=0.8,
                     mrksze=6, fontsize=9, labelfontsize=9, tickfontsize=8)

    if size == 'half':
        cp.init_plot(scl_a4=2, page_lnewdth_cm=page_width_cm, figsze=[6.4,4.8*(4/3)],lnewdth=0.8,
                     mrksze=6, fontsize=9, labelfontsize=9, tickfontsize=8)

    if default:
            cp.init_plot(scl_a4=2, page_lnewdth_cm=page_width_cm, 
                         figsze=[6.4,4.8*(4/3)],lnewdth=0.8, 
                         mrksze=6, fontsize=9, labelfontsize=9, tickfontsize=8)
        
def import_IR_DS(DS_filename, IR_folder, path=None, wn_min=0, wn_max=1e6,
                 process='mean', ignore_MCC=False):
    
    # If no path is supplied, use default
    if path is None: 
        path = 'data/'
    
    DS_importstr = path + DS_filename
    IR_folderpath = path + IR_folder + '/'
    
    # DS dictionary
    df_tmp = pd.read_csv(DS_importstr, delimiter='\t', header=None)
    DS_dict = {'name':df_tmp.iloc[:,0].values.astype(str),'DS':df_tmp.iloc[:,1].values}

    # IR data
    files = os.listdir(IR_folderpath)
    # print(files)
    
    # Check for MCC
    if ignore_MCC:
        tmp = [files[i].split('-')[1] for i in range(len(files))] 
        for i in range(len(files)-1,-1,-1):
            # print(i)
            if tmp[i] == 'MCC':
                files.pop(i)
  
    # Wavenumber array is constant --> Load first to determine final array shape
    df_tmp = pd.read_csv(IR_folderpath + files[0], delimiter='\t', header = None)
    wn_full = df_tmp.iloc[:,0].values
    wn_mask = (wn_full > wn_min) & (wn_full < wn_max)
    wn = wn_full[wn_mask]

    # Initialize IR_data, DS_data array and EXP_name array + fill in loop
    IR_data = np.zeros((len(files),len(wn)))
    DS_data = np.zeros(len(files))
    EXP_name = []

    # For plot only
    first_individ_idx = []
    first_individ_names = []

    for i in range(len(files)):
        df_tmp = pd.read_csv(IR_folderpath + files[i], delimiter='\t', header = None)
        transmission = df_tmp.iloc[:,1].values
        transmission_min = np.min(transmission)
        transmission_temp = transmission-transmission_min
        transmisssion_temp_max = np.max(transmission_temp)
        transmission_norm = transmission_temp/transmisssion_temp_max
        
        IR_data[i,:] = transmission_norm[wn_mask]
        
        # Generate target (DS) array based on filenames and append to EXP_name    
        tmp = files[i].split('-') 
        filestr = tmp[0] + '-' + tmp[1]
        
        DS_data[i] = DS_dict['DS'][DS_dict['name']==filestr]
        EXP_name.append(filestr)
            
        # For plot reasons: Save first appearing index for each DS
        if not filestr in first_individ_names:
            first_individ_names.append(filestr)
            first_individ_idx.append(i)
    
    # Convert list to numpy array
    first_individ_idx = np.array(first_individ_idx)
    EXP_name = np.array(EXP_name)
    
    # Mean data
    if process == 'mean':
        tmp_ir = np.zeros((len(first_individ_names),len(wn)))
        tmp_ds = np.zeros(len(first_individ_names))
        for i in range(len(first_individ_names)):
            tmp_ir[i,:] = np.mean(IR_data[EXP_name == first_individ_names[i],:],axis=0)
            tmp_ds[i] = DS_dict['DS'][DS_dict['name']==first_individ_names[i]]
            
        EXP_name = np.array(first_individ_names)
        first_individ_idx = np.arange(len(first_individ_names))
        IR_data = tmp_ir
        DS_data = tmp_ds
        
    # Only use one (first)
    if process == 'first':
        tmp_ir = np.zeros((len(first_individ_names),len(wn)))
        tmp_ds = np.zeros(len(first_individ_names))
        for i in range(len(first_individ_names)):
            tmp_ir[i,:] = IR_data[first_individ_idx[i],:]
            tmp_ds[i] = DS_dict['DS'][DS_dict['name']==first_individ_names[i]]
            
        EXP_name = np.array(first_individ_names)
        first_individ_idx = np.arange(len(first_individ_names))
        IR_data = tmp_ir
        DS_data = tmp_ds

    return IR_data, DS_data, wn, EXP_name, first_individ_idx, first_individ_names

def single_regression(IR_data, DS_data, test_index, train_index, cf=None):
    # Generate new instance if cf is not provided
    if cf is None:
        cf = config()
        
    # Linear Regression
    if cf.MODEL == 'lin':    
        model = LinearRegression(fit_intercept=cf.INTERCEPT, positive=cf.POSITIVE)
        model.fit(IR_data[train_index], DS_data[train_index])
    
    # Neural Net (MLP)
    elif cf.MODEL == 'nn':    
        model = MLPRegressor(hidden_layer_sizes=cf.HLS, activation=cf.ACTIVATION, solver='adam',       
                             alpha=cf.ALPHA, random_state=1, max_iter=cf.MAX_ITER, tol=cf.TOL)
        model.fit(IR_data[train_index], DS_data[train_index])
    
    # Fallback
    else: 
        print('No valid model is provided!')
    
    # Quality metrics 
    if len(test_index)>1:
        R2 = model.score(IR_data[test_index], DS_data[test_index])
    else:
        R2 = None
    DS_prediction = model.predict(IR_data[test_index])
    # Only allow non-negative predictions
    if cf.MODEL == 'lin': 
        DS_prediction[DS_prediction<0] = 0
    
    if len(test_index) > 1:
        MAE = np.mean(np.abs(DS_prediction-DS_data[test_index]))
        MRE = np.mean(np.abs(DS_prediction[DS_data[test_index]!=0]
                             -DS_data[test_index][DS_data[test_index]!=0])/DS_data[test_index][DS_data[test_index]!=0])
    else:
        MAE = np.abs(DS_prediction-DS_data[test_index])
        MRE = -1
        
    return model, R2, MAE, MRE

def perform_CV(IR_data, DS_data, EXP_name, individ_names, verbose=True, 
               IR_data_test_folder=None, cf=None, N_folds=None):
    
    # Generate new instance if cf is not provided
    if cf is None:
        cf = config()
        
    # Default index array
    N_tot = len(individ_names)
    # Define N_folds from test size if not provided
    if N_folds is None:
        N_folds = min(N_tot,max(2,int(1/cf.TEST_SIZE))) # minimum 2 fold, maximum N_tot folds
    index_array = np.arange(N_tot)
    
    # Initialize result arrays
    test_size_folds = np.zeros(N_folds*cf.N_REPS)
    MAE_folds = np.zeros(N_folds*cf.N_REPS)
    MRE_folds = np.zeros(N_folds*cf.N_REPS)
    R2_folds = np.zeros(N_folds*cf.N_REPS)
    coeff_folds = np.zeros((N_folds*cf.N_REPS,IR_data.shape[1]))
    DS_true_folds = []
    DS_pred_folds = []
    
    if IR_data_test_folder is None:
        DS_test_folder_folds = None
    else:
        DS_test_folder_folds = np.zeros((N_folds*cf.N_REPS,len(IR_data_test_folder)))
    
    # k-fold CV instance
    kf = RepeatedKFold(n_splits=N_folds, n_repeats=cf.N_REPS, random_state=None)
    test_index_kf = []
    train_index_kf = []
    
    # Actual repeated k-fold CV
    # NOTE: We perform a repeated k-fold on SPECIFIC experiments (defined by individ_names)
    for i, (exp_train_index, exp_test_index) in enumerate(kf.split(index_array)):
        # Extract the actual indices corresponding to this specific experiment
        test_index = []
        for e in range(len(exp_test_index)):
            # Test prints for debugging
            # print(exp_test_index[e],individ_names[exp_test_index[e]])
            for j in range(len(DS_data)):
                if EXP_name[j] == individ_names[exp_test_index[e]]:
                    test_index.append(j)

        all_index = list(np.arange(len(DS_data)))
        train_index = [j for j in all_index if j not in test_index]
        test_index_kf.append(test_index)
        train_index_kf.append(train_index)
        
        test_size_folds[i] = len(test_index)
                
        if len(DS_data) != len(test_index)+len(train_index):
            print('SOMETHING IS WRONG WITH TEST/TRAIN SPLIT')
            
        # Perform regression
        model, R2_folds[i], MAE_folds[i], MRE_folds[i] = \
            single_regression(IR_data, DS_data, test_index, train_index, cf=cf)
        
        DS_true_folds.append(DS_data[test_index])
        
        tmp_pred = model.predict(IR_data[test_index])
        if cf.MODEL == 'lin': 
            tmp_pred[tmp_pred<0] = 0
        DS_pred_folds.append(tmp_pred)
        
        # Extract coefficients in linear regression case
        if cf.MODEL == 'lin':
            coeff_folds[i,:] = model.coef_
            
        if cf.TEST_FOLDER and IR_data_test_folder is not None:
           DS_test_folder_folds[i,:] = model.predict(IR_data_test_folder)

    # Calculate overall metrics (weigh each fold with the actual fold size)
    weight = test_size_folds/(len(DS_data)*cf.N_REPS)
    MAE_mean = np.sum(MAE_folds*weight) 
    MRE_mean = np.sum(MRE_folds*weight) 
    if len(test_index)>1:  
        R2_mean = np.sum(R2_folds*weight)
    else:
        R2_mean = None    
    
    if verbose:
        print("-- CROSS VALIDATION --")
        if cf.MODEL == 'lin':
            print("Using a Linear regression model.")
        elif cf.MODEL == 'nn':
            print("Using a Neural Net (MLP) model.")
        print(f'Performed {cf.N_REPS} times a {N_folds}-fold cross validation.')
        print(f"The mean absolute error (across all folds) is {MAE_mean:.3f}")
        print(f"The mean relative error (across all folds) is {100*MRE_mean:.3f}%")
        print("##---------")
        
    return MAE_mean, R2_mean, MRE_mean, (N_folds, MAE_folds, R2_folds, coeff_folds, DS_test_folder_folds, test_index_kf, train_index_kf, DS_true_folds, DS_pred_folds)
        
def feature_selection(DATA, WN, TARGET=None, verbose=True, cf=None):
    
    # Generate new instance if cf is not provided
    if cf is None:
        cf = config()
        
    if cf.FS_TYPE == 'PCA':
        pca=PCA(n_components=cf.PCA_COMPONENTS)
        pca.fit(DATA)
        
        if verbose:
            print("-- PCA Feature Selection --")
            print(f'Cumulative explained variance ({cf.PCA_COMPONENTS}): {np.sum(pca.explained_variance_ratio_):.2f}.')
            print("##---------")
        
        return pca.transform(DATA), pca.transform(WN.reshape(1, -1)).reshape(-1), True, pca
    
    if cf.FS_TYPE == 'k_best':
        if len(WN) > cf.K_BEST:
            if cf.K_MODEL == 'r_regression':
                kb = SelectKBest(r_regression, k=cf.K_BEST)
            elif cf.K_MODEL == 'f_regression':                
                kb = SelectKBest(f_regression, k=cf.K_BEST)
            kb.fit(DATA,TARGET)
            
            if verbose:
                print("-- k-Best Feature Selection --")
                print(f'Using {cf.K_BEST} features and {cf.K_MODEL} model.')
                print("##---------")
            
            return kb.transform(DATA), kb.transform(WN.reshape(1, -1)).reshape(-1), True, kb
        else:
            if verbose:
                print("-- k-Best Feature Selection --")
                print(f'Number of provided wavenumbers is smaller than K_BEST')
                print(f'CAUTION! No Feature Selection performed!')
                print("##---------")
            
            return DATA, WN, False, None
    
    else:
        print('Provide valid FS_TYPE in config.py')
    
# Set-up config class
class config():
    def __init__(self):
        # Defome wavenumber ranges
        self.WN_DICT = {'full': (0, 1e6), 'non-fingerprint': (1500, 1e6), 
                       'fingerprint': (0, 1500), 'C=0, C-H, C-O': (1150, 1800), 
                       'carbonyl C=0': (1600, 1800), 'C-H': (1325, 1425), 
                       'ester C-O': (1150, 1300), 'random area': (2000, 2500)}
        # Ignore baseline (pure MCC)?
        self.IGNORE_MCC = False 
        # Fraction of samples to be used during testing (not exact, due to roundoff)
        self.TEST_SIZE = 2/16 
        # Number of repetition of full CV (randomized sample order)
        self.N_REPS = 1000
        # Linear Regression: 'lin', Neural Net: 'nn'
        self.MODEL = 'lin'
        # Perform feature selection?
        self.FS = False
        # Test files in data/test_IR_data?
        self.TEST_FOLDER = False         
        
        #%%% LINEAR MODEL PARAMETERS
        # Model parameter
        self.INTERCEPT = False # Fit y-intercept for linear model?
        self.POSITIVE = False # Force positive coefficients?
        
        #%%% MLP MODEL PARAMETERS
        self.HLS = (100,100)
        self.ACTIVATION = 'relu'     # 'identity', 'logistic', 'tanh', 'relu'
        self.MAX_ITER = 5000
        self.TOL = 1e-6
        self.ALPHA = 0.0001
        
        #%%% FEATURE SELECTION
        self.FS_TYPE = 'k_best'         # 'PCA', 'k_best'
        self.PCA_COMPONENTS = 3
        self.K_BEST = 500
        self.K_MODEL = 'f_regression'    # 'r_regression', 'f_regression'  
    
    
    
    
    
    
    
    
    
    
    
