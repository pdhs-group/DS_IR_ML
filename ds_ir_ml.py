"""
Perform the entire study published in 
"Efficient and accurate determination of the degree of substitution of cellulose acetate using ATR-FTIR spectrosopy and machine learning"
Frank Rhein, Timo Sehn, Michael Meier

Raw data is available at: URL_DATA
    
ds_ir_ml.py 
"""

#%% PACKAGES / SETUP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import mod.custom_plotter as cp
from sklearn.linear_model import LinearRegression

# Custom utility functions. See config class for default settings.
from mod.util_functions import import_IR_DS, perform_CV, feature_selection, init_plot, config

# Which parts of the paper do you want to evaluate?
COMPARE_P = True
IR_ML_BASELINE = True
K_FOLD_STUDY = True
WN_RANGE_STUDY = True
FS_STUDY = True
EXTRAPOLATION = True

# Close plots and setup intial style
LNE_WDTH = 16.58763
EXPORT = True
plt.close('all')
init_plot(size = 'half', page_width_cm = LNE_WDTH)

#%% P1: 31P vs. 1H vs. IR_int data on Wolfs.A [1]
if COMPARE_P:
    # Generate (default) config instance        
    cf = config()
    
    #%%% Import
    WN_MIN, WN_MAX = cf.WN_DICT['full']
    
    # Note: Exclude MCC (baseline) from evaluation
    IR_data, H_DS_data, wn, EXP_name, first_individ_idx, first_individ_names = \
        import_IR_DS('H_DS_Wolfs.txt', 'IR_data_Wolfs_A', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
        
    _, P_DS_data, _, _, _, _ = \
        import_IR_DS('P_DS_Wolfs.txt', 'IR_data_Wolfs_A', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
    
    _, IR_DS_data, _, _, _, _ = \
        import_IR_DS('IR_DS_Wolfs.txt', 'IR_data_Wolfs_A', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
    
    #%%% Calculations
    # 31P vs. 1H -> Ignore P_DS == -1 (not existent, insolubility)
    MAE_1H = np.mean(np.abs(H_DS_data[P_DS_data>0]-P_DS_data[P_DS_data>0]))
    MRE_1H = np.mean(np.abs(H_DS_data[P_DS_data>0]-P_DS_data[P_DS_data>0])/P_DS_data[P_DS_data>0])
    
    # 31P vs. IR_int 
    MAE_IR_int = np.mean(np.abs(IR_DS_data[P_DS_data>0]-P_DS_data[P_DS_data>0]))
    MRE_IR_int = np.mean(np.abs(IR_DS_data[P_DS_data>0]-P_DS_data[P_DS_data>0])/P_DS_data[P_DS_data>0])
        
    #%%% Report and Plot
    print('-- 31P ("Truth") vs. 1H --')
    print(f"Mean absolute error (MAE): {MAE_1H:.3f}")
    print(f"Mean relative error (MRE): {100*MRE_1H:.3f}\%")
    print("##---------")  
    print('-- 31P ("Truth") vs. IR_int --')
    print(f"Mean absolute error (MAE): {MAE_IR_int:.3f}")
    print(f"Mean relative error (MRE): {100*MRE_IR_int:.3f}\%")
    print("##---------") 
    
    init_plot(size = 'half', page_width_cm = LNE_WDTH)    
    
    fig1 = plt.figure()    
    ax1 = fig1.add_subplot(1,1,1) 
    
    ax1.plot([0,10],[0,10],color='k')
    ax1.scatter(P_DS_data, H_DS_data, edgecolors='k', color=cp.blue, 
                alpha=0.7, zorder=3, label='$\mathrm{DS_{1H}}$')
    ax1.scatter(P_DS_data, IR_DS_data, edgecolors='k', color=cp.red, 
                alpha=0.7, zorder=2, label='$\mathrm{DS_{IR,int}}$')    
    ax1.text(0.98,0.12,'$\mathrm{MAE_{1H}}='+f'{MAE_1H:.3f}$', transform=ax1.transAxes,
             horizontalalignment='right', verticalalignment='bottom',
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    ax1.text(0.98,0.02,'$\mathrm{MAE_{IR,int}}='+f'{MAE_IR_int:.3f}$', transform=ax1.transAxes,
             horizontalalignment='right', verticalalignment='bottom',
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    
    ax1.set_xlabel('$\mathrm{DS_{31P}}$ ("Truth") / $-$')
    ax1.set_ylabel('$\mathrm{DS_{1H}}$ / $\mathrm{DS_{IR,int}}$ / $-$')
    ax1.grid(True)
    ax1.set_xlim([1,3.5])
    ax1.set_ylim([1,3.5])
    plt.tight_layout() 

    ax1.legend()
    
    if EXPORT:
        plt.savefig("exp/31P_vs_1H_vs_IRint.png", dpi=300)        
        plt.savefig("exp/31P_vs_1H_vs_IRint.pdf")

#%% P2: IR ML baseline evaluation on Wolfs.A [1]
if IR_ML_BASELINE:
    # Generate (default) config instance        
    cf = config()
    
    #%%% Import
    WN_MIN, WN_MAX = cf.WN_DICT['full']
    
    # Note: Exclude MCC (baseline) from evaluation
    IR_data, H_DS_data, wn, EXP_name, first_individ_idx, first_individ_names = \
        import_IR_DS('H_DS_Wolfs.txt', 'IR_data_Wolfs_A', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
    
    _, IR_DS_data, _, _, _, _ = \
        import_IR_DS('IR_DS_Wolfs.txt', 'IR_data_Wolfs_A', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
        
    #%%% Calculations
    # Perform the CV
    MAE_IR_bl, R2_IR_bl, MRE_IR_bl, KF_info = \
        perform_CV(IR_data, H_DS_data, EXP_name, first_individ_names, 
                   verbose=True, cf=cf, N_folds=8)
        
    # Extract the test results
    P2_true = np.concatenate(KF_info[7])
    P2_pred = np.concatenate(KF_info[8])
    # Generate mean and std for the predicitons
    # Each value of H_DS_data occurs cf.N_REPS-times in P2_true
    P2_pred_sort = P2_pred[np.argsort(P2_true)] 
    P2_true_sort = P2_true[np.argsort(P2_true)] 
    
    P2_pred = np.reshape(P2_pred_sort,(len(H_DS_data),cf.N_REPS))
    P2_true = np.reshape(P2_true_sort,(len(H_DS_data),cf.N_REPS))
    
    ML_DS_data = np.zeros(len(H_DS_data))
    ML_DS_std = np.zeros(len(H_DS_data))
    for i in range(len(H_DS_data)):
        ML_DS_data[i] = np.mean(P2_pred[i,:])
        ML_DS_std[i] = np.std(P2_pred[i,:])       
    
    # 31P vs. IR_int 
    MAE_IR_int = np.mean(np.abs(IR_DS_data-H_DS_data))
    MRE_IR_int = np.mean(np.abs(IR_DS_data[H_DS_data>0]-H_DS_data[H_DS_data>0])/H_DS_data[H_DS_data>0])
        
    #%%% Report and Plot
    print('-- 1H ("Truth") vs. IR_int --')
    print(f"Mean absolute error (MAE): {MAE_IR_int:.3f}")
    print(f"Mean relative error (MRE): {100*MRE_IR_int:.3f}\%")
    print("##---------") 
    
    init_plot(size = 'half', page_width_cm = LNE_WDTH)    
    
    fig2 = plt.figure()    
    ax2 = fig2.add_subplot(1,1,1) 
    
    ax2.plot([-10,10],[-10,10],color='k')
    ax2.errorbar(np.sort(H_DS_data), ML_DS_data, yerr=ML_DS_std, fmt='o', mfc=cp.green,
                  mec='k', label='$\mathrm{DS_{IR,ML}}$', capsize=5, ecolor='k', 
                  elinewidth=1,alpha=0.7)
    
    ax2.scatter(H_DS_data, IR_DS_data, edgecolors='k', color=cp.red, 
                alpha=0.7, zorder=2, label='$\mathrm{DS_{IR,int}}$') 
    ax2.text(0.98,0.12,'$\mathrm{MAE_{IR,ML}}='+f'{MAE_IR_bl:.3f}$', transform=ax2.transAxes,
             horizontalalignment='right', verticalalignment='bottom',
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    ax2.text(0.98,0.02,'$\mathrm{MAE_{IR,int}}='+f'{MAE_IR_int:.3f}$', transform=ax2.transAxes,
             horizontalalignment='right', verticalalignment='bottom',
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    
    ax2.set_xlabel('$\mathrm{DS_{1H}}$ ("Truth") / $-$')
    ax2.set_ylabel('$\mathrm{DS_{IR,ML}}$ or $\mathrm{DS_{IR,int}}$ / $-$')
    ax2.grid(True)
    ax2.set_xlim([np.min([P2_true,P2_pred])-0.2,np.max([P2_true,P2_pred])+0.2])
    ax2.set_ylim([np.min([P2_true,P2_pred])-0.2,np.max([P2_true,P2_pred])+0.2])
    plt.tight_layout() 
    
    ax2.legend()
    
    if EXPORT:
        plt.savefig("exp/IR_ML_baseline.png", dpi=300)        
        plt.savefig("exp/IR_ML_baseline.pdf")
        
#%% P3: k-fold Parameterstudy on Wolfs.A [1]
if K_FOLD_STUDY:
    # Generate (default) config instance        
    cf = config()
    
    #%%% Import
    WN_MIN, WN_MAX = cf.WN_DICT['full']
    
    # Note: Exclude MCC (baseline) from evaluation
    IR_data, H_DS_data, wn, EXP_name, first_individ_idx, first_individ_names = \
        import_IR_DS('H_DS_Wolfs.txt', 'IR_data_Wolfs_A', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
            
    #%%% Calculations
    k_array = np.arange(2,17,2)    
    # k_array = np.array([2,4,8,16])
    MAE_array = np.zeros(k_array.shape)
    MAE_std_array = np.zeros(k_array.shape)
    MRE_array = np.zeros(k_array.shape)
    KF_info_array = []
    
    # Loop through all k-values
    for i in range(len(k_array)): 
        # Perform the CV
        MAE_IR_tmp, R2_IR_tmp, MRE_IR_tmp, KF_info_tmp = \
            perform_CV(IR_data, H_DS_data, EXP_name, first_individ_names, 
                       verbose=True, cf=cf, N_folds=k_array[i])
        
        MAE_array[i] = MAE_IR_tmp
        MRE_array[i] = MAE_IR_tmp
        
        KF_info_array.append(KF_info_tmp)
        
        P_true = np.concatenate(KF_info_tmp[7])
        P_pred = np.concatenate(KF_info_tmp[8])
        
        MAE_std_array[i] = np.std(np.abs(P_true-P_pred))
    
    # Extract data for box plot
    box_data = []
    for i in range(len(KF_info_array)):
        box_data.append(KF_info_array[i][1])
        
    #%%% Report and Plot    
    init_plot(size = 'half', page_width_cm = LNE_WDTH)    
    
    fig3 = plt.figure()    
    ax3 = fig3.add_subplot(1,1,1) 

    boxprops = dict()
    medianprops = dict(color='k')
    meanprops = dict(markerfacecolor='k', marker='.', markeredgecolor='k')
      
    bp = ax3.boxplot(box_data, patch_artist=True, boxprops=boxprops, medianprops=medianprops,
                     showfliers=False, positions = k_array, widths=1, showmeans = True,
                     meanprops=meanprops)
    
    for i in range(len(bp['boxes'])):
            bp['boxes'][i].set_facecolor(cp.green)
            bp['boxes'][i].set_alpha(0.8)
            bp['boxes'][i].set_edgecolor('k')
            
    ax3.set_xlabel('$k$ during k-fold / $-$')
    ax3.set_ylabel('$\mathrm{MAE_{IR,ML}}$ / $-$')
    ax3.set_xticks(k_array)
    ax3.grid(axis='y')
    ax3.set_xlim([1,17])
    plt.tight_layout() 
     
    
    if EXPORT:
        plt.savefig("exp/IR_ML_k_study.png", dpi=300)        
        plt.savefig("exp/IR_ML_k_study.pdf")
 
#%% P4: Wavenumber Parameterstudy on Wolfs.A [1]
if WN_RANGE_STUDY:
    # Generate (default) config instance        
    cf = config()
    
    #%%% Import
    col_array = ['gray',cp.green, cp.red, cp.blue, cp.yellow, cp.black, 
                 cp.purple, cp.orange, cp.cyan]
    
    WN_MIN, WN_MAX = cf.WN_DICT['full']
    
    IR_data_full, IR_DS_data, wn_full, _, first_individ_idx, first_individ_names = \
        import_IR_DS('IR_DS_Wolfs.txt', 'IR_data_Wolfs_A', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)

    #%%% Calculations
    # Get all wavenumber areas
    wn_areas = [*cf.WN_DICT]
    
    # Initialize result arrays
    MAE_array = np.zeros(len(wn_areas))
    MAE_test_array = np.zeros(len(wn_areas))
    KF_info_array = []
    
    # Loop over all of the cases and perform CV
    for i in range(len(wn_areas)):        
        # Extract wavenumbers
        WN_MIN, WN_MAX = cf.WN_DICT[wn_areas[i]]
    
        # Note: Exclude MCC (baseline) from evaluation
        IR_data, H_DS_data, wn, EXP_name, first_individ_idx, first_individ_names = \
            import_IR_DS('H_DS_Wolfs.txt', 'IR_data_Wolfs_A', path='data/', 
                         wn_min=WN_MIN, wn_max=WN_MAX, 
                         process='mean', ignore_MCC=cf.IGNORE_MCC)
 
        # Perform the CV
        MAE_IR_tmp, R2_IR_tmp, MRE_IR_tmp, KF_info_tmp = \
            perform_CV(IR_data, H_DS_data, EXP_name, first_individ_names, 
                       verbose=False, cf=cf, N_folds=8)
        
        KF_info_array.append(KF_info_tmp)
    
    # Extract data for box plot
    box_data_wn = np.zeros((len(KF_info_array),len(KF_info_array[0][1])))
    for i in range(len(KF_info_array)):
        box_data_wn[i,:] = KF_info_array[i][1]
    
    # Reference MAE
    MAE_ref = np.mean(np.abs(IR_DS_data[IR_DS_data!=0]-H_DS_data[IR_DS_data!=0])) 
    
    #%%% Report and Plot    
    ## --- SETUP  
    init_plot(size = 'full', page_width_cm = LNE_WDTH)    
    
    plt.rcParams['figure.figsize']=[plt.rcParams['figure.figsize'][0],
                                    plt.rcParams['figure.figsize'][1]*1.5]
    fig5 = plt.figure()    
    gs = GridSpec(3, 1, figure=fig5)
    ax4 = fig5.add_subplot(gs[0:2, :])
    ax5 = fig5.add_subplot(gs[2, :])
    
    ## --- DATA RANGES  
    greys = cm.Greys(np.linspace(0.4, 0.7, len(first_individ_idx)))
    
    for i in range(len(first_individ_idx)):
        ax5.plot(wn_full, 100*IR_data_full[first_individ_idx[i],:], linewidth=1, color=greys[i])
        
    rect_list = []
    patch_list = []
    tmp_ht = 0
    for a in range(len(wn_areas)):
        wn_min_tmp, wn_max_tmp = cf.WN_DICT[wn_areas[a]]    
        rect_list.append(matplotlib.patches.Rectangle((wn_min_tmp,tmp_ht), wn_max_tmp-wn_min_tmp, 
                                                      100/len(wn_areas), facecolor=col_array[a],
                                                      alpha=0.8, zorder=3, edgecolor='k'))
        ax5.add_patch(rect_list[a])
        patch_list.append(matplotlib.patches.Patch(facecolor=col_array[a], alpha=0.8, 
                                                   label=wn_areas[a], edgecolor='k'))
        tmp_ht += 100/len(wn_areas)    
    
    ax5.text(0.005,0.97,'(b)',transform=ax5.transAxes, horizontalalignment='left',
             verticalalignment='top', 
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    ax5.set_xlabel(r'Wavenumber $\nu$ / cm$^{-1}$')
    ax5.set_ylabel('Transmission / $\%$')
    ax5.set_xlim(min(wn_full),max(wn_full))
    ax5.set_ylim(-9,109)
    ax5.invert_xaxis()
    ax5.grid(True)
    
    plt.tight_layout()
    
    box = ax5.get_position()
    ax5.set_position([box.x0, box.y0, box.width*0.8, box.height])
        
    ax5.legend(handles=patch_list, bbox_to_anchor=(1.02, 1.02),loc='upper left', fontsize='small', 
                   framealpha=1, ncol=1, labelspacing = 0.6) 
       
    ## --- RESULTS    
    init_plot(size = 'full', page_width_cm = LNE_WDTH)    
            
    boxprops = dict()
    medianprops = dict(color='k')
        
    bp = ax4.boxplot(box_data_wn.T, patch_artist=True, boxprops=boxprops, medianprops=medianprops,
                     showfliers=False)
    
    for i in range(len(bp['boxes'])):
            bp['boxes'][i].set_facecolor(col_array[i])
            bp['boxes'][i].set_alpha(0.8)
            bp['boxes'][i].set_edgecolor('k')
            
    ax4.text(0.005,0.98,'(a)',transform=ax4.transAxes, horizontalalignment='left',
             verticalalignment='top', 
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))        
    ax4.axhline(MAE_ref,color='k',linewidth=1.5, zorder=0, label='$\mathrm{MAE_{IR,int}}$')
    ax4.set_xlabel('Wavenumber Area')
    ax4.set_ylabel('$\mathrm{MAE_{IR,ML}}$ / $-$')
    ax4.set_ylim(0,0.5)
    ax4.set_xticklabels([])
    ax4.set_xticks([])
    ax4.grid(True)
    
    plt.tight_layout()
    
    box = ax4.get_position()
    ax4.set_position([box.x0, box.y0, box.width*0.8, box.height])
    
    ax4.legend()
    ref_handle, _ = ax4.get_legend_handles_labels()    
    ax4.legend(handles=patch_list+ref_handle, bbox_to_anchor=(1.02, 1.02),
               loc='upper left',
               fontsize='small', framealpha=1, ncol=1, labelspacing = 0.6) 
    
    if EXPORT:
        plt.savefig("exp/IR_ML_wn_ranges.png", dpi=300)        
        plt.savefig("exp/IR_ML_wn_ranges.pdf")


#%% P5: FS parameterstudy on Wolfs.A [1]
if FS_STUDY:
    # Generate (default) config instance        
    cf = config()
    cf.K_MODEL = 'f_regression'
    cf.FS_TYPE = 'k_best'
    
    #%%% Import
    wn_areas = [*cf.WN_DICT]
    col_array = ['gray',cp.green, cp.red, cp.blue, cp.yellow, cp.black, 
                 cp.purple, cp.orange, cp.cyan]
    
    WN_MIN, WN_MAX = cf.WN_DICT['full']
    
    # Note: Exclude MCC (baseline) from evaluation
    IR_data_full, H_DS_data, wn_full, EXP_name, first_individ_idx, first_individ_names = \
        import_IR_DS('H_DS_Wolfs.txt', 'IR_data_Wolfs_A', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
        
    #%%% Calculations    
    # Initialize result arrays
    n_array = np.arange(50,1751,100)
    MAE_array = np.zeros(len(n_array))
    MAE_std_array = np.zeros(len(n_array))
    KF_info_array = []
    
    # Loop through all n-values
    for i in range(len(n_array)): 
        # Adjust K_BEST
        cf.K_BEST = n_array[i]
             
        # Do the feature selection
        if n_array[i] < len(wn_full):
            IR_data_fs, wn_fs, FS_success, fs = feature_selection(IR_data_full, 
                                                                  wn_full, H_DS_data,
                                                                  cf=cf)
        
        else:
            IR_data_fs, wn_fs = np.copy(IR_data_full), np.copy(wn_full)
            
        # Perform the CV
        MAE_FS_tmp, _, _, KF_info_tmp = \
            perform_CV(IR_data_fs, H_DS_data, EXP_name, first_individ_names, 
                       verbose=False, cf=cf, N_folds=8)
            
        MAE_array[i] = MAE_FS_tmp
        
        KF_info_array.append(KF_info_tmp)
        
        P_true = np.concatenate(KF_info_tmp[7])
        P_pred = np.concatenate(KF_info_tmp[8])
        
        MAE_std_array[i] = np.std(np.abs(P_true-P_pred))
    
    # Find minimum index for plotting reasons
    idx_min = np.argmin(MAE_array)
    
    # Extract data for box plot
    box_data_fs = []
    for i in range(len(KF_info_array)):
        box_data_fs.append(KF_info_array[i][1])
        
    # WN data for "jump in performance"
    cf.K_BEST = 950
    _, wn_fs_950, _, _ = feature_selection(IR_data_full, wn_full, H_DS_data, cf=cf)
    cf.K_BEST = 850
    _, wn_fs_850, _, _ = feature_selection(IR_data_full, wn_full, H_DS_data, cf=cf)
    diff_wn = np.setdiff1d(wn_fs_950,wn_fs_850)
        
    #%%% Report and Plot    
    ## --- SETUP  
    init_plot(size = 'full', page_width_cm = LNE_WDTH)    
    
    plt.rcParams['figure.figsize']=[plt.rcParams['figure.figsize'][0],
                                    plt.rcParams['figure.figsize'][1]*1.5]
    fig6 = plt.figure()    
    gs = GridSpec(3, 1, figure=fig6)
    ax7 = fig6.add_subplot(gs[0:2, :])
    ax6 = fig6.add_subplot(gs[2, :])
    
    ## --- DATA RANGES  
    greys = cm.Greys(np.linspace(0.4, 0.7, len(first_individ_idx)))
            
    # Reload the best FS 
    cf.K_BEST = n_array[idx_min]
    IR_data_fs, wn_fs, FS_success, fs = feature_selection(IR_data_full, wn_full, H_DS_data,
                                                          cf=cf)
    
    ax6.scatter(wn_fs, np.zeros(wn_fs.shape), marker='o', color='k', 
                facecolor=cm.Greys(0.2), alpha=0.4, 
                label='selected', zorder=5)
    ax6.plot(wn_full, 100*IR_data_full[-1,:], linewidth=2, color=cm.Greys(0.8), zorder=1)
    
    # Plot ranges
    rect_list = []
    patch_list = []
    tmp_ht = 100/(len(wn_areas)+1)
    for a in range(len(wn_areas)):
        wn_min_tmp, wn_max_tmp = cf.WN_DICT[wn_areas[a]]    
        rect_list.append(matplotlib.patches.Rectangle((wn_min_tmp,tmp_ht), wn_max_tmp-wn_min_tmp, 
                                                      100/(len(wn_areas)+1),
                                                      facecolor=col_array[a],
                                                      alpha=0.8, zorder=3, edgecolor='k'))
        ax6.add_patch(rect_list[a])
        patch_list.append(matplotlib.patches.Patch(facecolor=col_array[a], alpha=0.8, 
                                                   label=wn_areas[a], edgecolor='k'))
        tmp_ht += 100/(len(wn_areas)+1) 
        
    ax6.text(0.005,0.97,'(b)',transform=ax6.transAxes, horizontalalignment='left',
             verticalalignment='top', 
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    ax6.text(0.995,0.97,'$n_{\mathrm{opt}}'+f'={n_array[idx_min]}$',transform=ax6.transAxes,
             horizontalalignment='right',
             verticalalignment='top', 
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    ax6.set_xlabel(r'Wavenumber $\nu$ / cm$^{-1}$')
    ax6.set_ylabel('Transmission / $\%$')
    ax6.set_xlim(min(wn_full),max(wn_full))
    ax6.set_ylim(-9,109)
    ax6.invert_xaxis()
    ax6.grid(True)
    
    plt.tight_layout()
    
    ## --- RESULTS    
    ax7.errorbar(n_array, MAE_array, yerr=MAE_std_array, 
                 #[np.zeros(MAE_std_array.shape),MAE_std_array],
                 fmt='o', mec='k', 
                 mfc=cp.green, capsize=5, ecolor='k', elinewidth=1)

            
    ax7.text(0.005,0.98,'(a)',transform=ax7.transAxes, horizontalalignment='left',
             verticalalignment='top', 
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))        
    ax7.set_xlabel('$n$ of n-best feature selection / $-$')
    ax7.set_ylabel('$\mathrm{MAE_{IR,ML}}$ / $-$')
    ax7.grid(axis='y')
    plt.tight_layout()
    
    box = ax7.get_position()
    ax7.set_position([box.x0, box.y0, box.width*0.8, box.height])
    
    # ax7.legend()
    ref_handle, _ = ax6.get_legend_handles_labels()    
    ax7.legend(handles=patch_list+ref_handle, bbox_to_anchor=(1.02, 1.02),
                loc='upper left',
                fontsize='small', framealpha=1, ncol=1, labelspacing = 0.6) 
    
    if EXPORT:
        plt.savefig("exp/IR_ML_FS_study.png", dpi=300)        
        plt.savefig("exp/IR_ML_FS_study.pdf")

#%% P4/P5 Combined plot
if WN_RANGE_STUDY and FS_STUDY:
    ## --- SETUP  
    init_plot(size = 'full', page_width_cm = LNE_WDTH)    
    
    plt.rcParams['figure.figsize']=[plt.rcParams['figure.figsize'][0],
                                    plt.rcParams['figure.figsize'][1]*(5/3)]
    fig20 = plt.figure()    
    gs = GridSpec(5, 2, figure=fig20)
    ax20 = fig20.add_subplot(gs[0:3, 0])
    ax21 = fig20.add_subplot(gs[0:3, 1])
    ax22 = fig20.add_subplot(gs[3:5, :])
    
    ## --- DATA RANGES  
    greys = cm.Greys(np.linspace(0.4, 0.7, len(first_individ_idx)))
            
    # Reload the best FS 
    cf.K_BEST = n_array[idx_min]
    IR_data_fs, wn_fs, FS_success, fs = feature_selection(IR_data_full, wn_full, H_DS_data,
                                                          cf=cf)
    
    ax22.scatter(wn_fs, np.zeros(wn_fs.shape), marker='o', color='k', 
                facecolor=cm.Greys(0.2), alpha=0.4, 
                label=r'selected $\nu$', zorder=5)
    ax22.plot(wn_full, 100*IR_data_full[-1,:], linewidth=2, color=cm.Greys(0.8), zorder=1)
    
    # Plot ranges
    rect_list = []
    patch_list = []
    tmp_ht = 100/(len(wn_areas)+1)
    for a in range(len(wn_areas)):
        wn_min_tmp, wn_max_tmp = cf.WN_DICT[wn_areas[a]]    
        rect_list.append(matplotlib.patches.Rectangle((wn_min_tmp,tmp_ht), wn_max_tmp-wn_min_tmp, 
                                                      100/(len(wn_areas)+1),
                                                      facecolor=col_array[a],
                                                      alpha=0.8, zorder=3, edgecolor='k'))
        ax22.add_patch(rect_list[a])
        patch_list.append(matplotlib.patches.Patch(facecolor=col_array[a], alpha=0.8, 
                                                   label=wn_areas[a], edgecolor='k'))
        tmp_ht += 100/(len(wn_areas)+1) 
        
    ax22.text(0.015,0.97,'(c)',transform=ax22.transAxes, horizontalalignment='left',
             verticalalignment='top', 
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    ax22.text(0.995,0.97,'$n_{\mathrm{opt}}'+f'={n_array[idx_min]}$',transform=ax22.transAxes,
             horizontalalignment='right',
             verticalalignment='top', 
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    ax22.set_xlabel(r'Wavenumber $\nu$ / cm$^{-1}$')
    ax22.set_ylabel('Transmission / $\%$')
    ax22.set_xlim(min(wn_full),max(wn_full))
    ax22.set_ylim(-9,109)
    ax22.invert_xaxis()
    ax22.legend()
    ax22.grid(True)
    
    plt.tight_layout()
    
    ## --- FS   
    boxprops = dict()
    medianprops = dict(color='k')
    meanprops = dict(markerfacecolor='k', marker='.', markeredgecolor='k')
      
    bp = ax21.boxplot(box_data_fs, patch_artist=True, boxprops=boxprops, medianprops=medianprops,
                      showfliers=False, widths=50, showmeans = True, positions=n_array,
                      meanprops=meanprops)
    
    for i in range(len(bp['boxes'])):
            bp['boxes'][i].set_facecolor(cp.green)
            if i == idx_min:
                bp['boxes'][i].set_facecolor(cp.red)
            bp['boxes'][i].set_alpha(0.8)
            bp['boxes'][i].set_edgecolor('k')

            
    ax21.text(0.01,0.98,'(b)',transform=ax21.transAxes, horizontalalignment='left',
             verticalalignment='top', 
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))        
    ax21.set_xlabel('$n$ of n-best feature selection / $-$')
    ax21.set_ylabel('$\mathrm{MAE_{IR,ML}}$ / $-$')
    ax21.grid(axis='y')
    skip = 4
    ax21.set_xticks(ax21.get_xticks()[::skip])
    ax21.set_xticklabels(n_array[::skip].astype(str))
    plt.tight_layout()    
       
    ## --- WN RANGES                
    boxprops = dict()
    medianprops = dict(color='k')
    meanprops = dict(markerfacecolor='k', marker='.', markeredgecolor='k')
        
    bp = ax20.boxplot(box_data_wn.T, patch_artist=True, boxprops=boxprops, medianprops=medianprops,
                     showfliers=False, widths=0.5, showmeans = True,
                     meanprops=meanprops)
    
    for i in range(len(bp['boxes'])):
            bp['boxes'][i].set_facecolor(col_array[i])
            bp['boxes'][i].set_alpha(0.8)
            bp['boxes'][i].set_edgecolor('k')
            
    ax20.text(0.01,0.98,'(a)',transform=ax20.transAxes, horizontalalignment='left',
             verticalalignment='top', 
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))        
    ax20.axhline(MAE_ref,color='k',linewidth=1.5, zorder=0, label='$\mathrm{MAE_{IR,int}}$')
    ax20.set_xlabel('Wavenumber Area')
    ax20.set_ylabel('$\mathrm{MAE_{IR,ML}}$ / $-$')
    ax20.set_ylim(0,0.5)
    ax20.set_xticklabels([])
    ax20.set_xticks([])
    ax20.grid(True)
    
    plt.tight_layout()

    box = ax22.get_position()
    ax22.set_position([box.x0, box.y0, box.width*0.8, box.height])

    ref_handle, _ = ax22.get_legend_handles_labels()    
    ax22.legend(handles=patch_list+ref_handle, bbox_to_anchor=(1.02, 1.02),
                loc='upper left',
                fontsize='small', framealpha=1, ncol=1, labelspacing = 0.6) 
    
    if EXPORT:
        plt.savefig("exp/IR_ML_WN_FS.png", dpi=300)        
        plt.savefig("exp/IR_ML_WN_FS.pdf")
        
        
#%% P6: Extrapolation on Wolfs.B, Sehn.A, Sehn.B and Sehn.C [2,3]
if EXTRAPOLATION:
    # Generate (default) config instance        
    cf = config()
    cf.FS = True
    cf.K_BEST = 250
    cf.K_MODEL = 'f_regression'
    cf.FS_TYPE = 'k_best'
    
    #%%% Import
    wn_areas = [*cf.WN_DICT]
    col_array = ['gray',cp.green, cp.red, cp.blue, cp.yellow, cp.black, 
                 cp.purple, cp.orange, cp.cyan]
    
    WN_RANGE = 'full'
    WN_MIN, WN_MAX = cf.WN_DICT[WN_RANGE]
    
    # Wolfs.A for training
    IR_data_full, H_DS_data, wn_full, EXP_name, first_individ_idx, first_individ_names = \
        import_IR_DS('H_DS_Wolfs.txt', 'IR_data_Wolfs_A', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
    
    # Wolfs.B for evaluation
    IR_data_Wb, H_DS_data_Wb, _, _, _, fin_Wb = \
        import_IR_DS('H_DS_Wolfs.txt', 'IR_data_Wolfs_B', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
    
    # Sehn.A for evaluation
    IR_data_Sa, H_DS_data_Sa, _, _, _, fin_Sa = \
        import_IR_DS('H_DS_TS.txt', 'IR_data_TS_A', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
        
    # Sehn.B for evaluation
    IR_data_Sb, H_DS_data_Sb, _, _, _, fin_Sb = \
        import_IR_DS('H_DS_TS.txt', 'IR_data_TS_B', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
    
    # Sehn.C for evaluation
    IR_data_Sc, H_DS_data_Sc, _, _, _, fin_Sc = \
        import_IR_DS('H_DS_TS.txt', 'IR_data_TS_C', path='data/', 
                     wn_min=WN_MIN, wn_max=WN_MAX, 
                     process='mean', ignore_MCC=cf.IGNORE_MCC)
        
    #%%% Calculations    
             
    # Do the feature selection
    if cf.FS and (cf.K_BEST < len(wn_full)):
        # Fit and transform on Wolfs.A
        IR_data_fs, wn_fs, _, fs = feature_selection(IR_data_full, wn_full,  
                                                     H_DS_data, cf=cf)
        # Apply to Wolfs.B, Sehn.A and Sehn.B
        IR_data_fs_Wb = fs.transform(IR_data_Wb)        
        IR_data_fs_Sa = fs.transform(IR_data_Sa)      
        IR_data_fs_Sb = fs.transform(IR_data_Sb)    
        IR_data_fs_Sc = fs.transform(IR_data_Sc)
        
    else:
        IR_data_fs, wn_fs = np.copy(IR_data_full), np.copy(wn_full)
        IR_data_fs_Wb = np.copy(IR_data_Wb)
        IR_data_fs_Sa = np.copy(IR_data_Sa)
        IR_data_fs_Sb = np.copy(IR_data_Sb)
        IR_data_fs_Sc = np.copy(IR_data_Sc)
            
    # Train the model on all Wolfs.A data
    model = LinearRegression(fit_intercept=cf.INTERCEPT, positive=cf.POSITIVE)
    model.fit(IR_data_fs, H_DS_data)
    
    model_full = LinearRegression(fit_intercept=cf.INTERCEPT, positive=cf.POSITIVE)
    model_full.fit(IR_data_full, H_DS_data)
    
    # Predict Wolfs.B
    H_DS_pred_Wb = model.predict(IR_data_fs_Wb)
    H_DS_pred_Wb_full = model_full.predict(IR_data_Wb)
    
    # Predict Sehn.A
    H_DS_pred_Sa = model.predict(IR_data_fs_Sa)
    H_DS_pred_Sa_full = model_full.predict(IR_data_Sa)

    # Predict Sehn.B
    H_DS_pred_Sb = model.predict(IR_data_fs_Sb)
    H_DS_pred_Sb_full = model_full.predict(IR_data_Sb)
    
    # Predict Sehn.C
    H_DS_pred_Sc = model.predict(IR_data_fs_Sc)
    H_DS_pred_Sc_full = model_full.predict(IR_data_Sc)
    
    # Calculate MAE and MRE 
    MAE_Wb = np.mean(np.abs(H_DS_pred_Wb[H_DS_data_Wb>0]-H_DS_data_Wb[H_DS_data_Wb>0]))
    MRE_Wb = np.mean(np.abs(H_DS_pred_Wb[H_DS_data_Wb>0]-H_DS_data_Wb[H_DS_data_Wb>0])/H_DS_data_Wb[H_DS_data_Wb>0])
    MAE_Wb_full = np.mean(np.abs(H_DS_pred_Wb_full[H_DS_data_Wb>0]-H_DS_data_Wb[H_DS_data_Wb>0]))
    MRE_Wb_full = np.mean(np.abs(H_DS_pred_Wb_full[H_DS_data_Wb>0]-H_DS_data_Wb[H_DS_data_Wb>0])/H_DS_data_Wb[H_DS_data_Wb>0])
    
    MAE_Sa = np.mean(np.abs(H_DS_pred_Sa[H_DS_data_Sa>0]-H_DS_data_Sa[H_DS_data_Sa>0]))
    MRE_Sa = np.mean(np.abs(H_DS_pred_Sa[H_DS_data_Sa>0]-H_DS_data_Sa[H_DS_data_Sa>0])/H_DS_data_Sa[H_DS_data_Sa>0])
    MAE_Sa_full = np.mean(np.abs(H_DS_pred_Sa_full[H_DS_data_Sa>0]-H_DS_data_Sa[H_DS_data_Sa>0]))
    MRE_Sa_full = np.mean(np.abs(H_DS_pred_Sa_full[H_DS_data_Sa>0]-H_DS_data_Sa[H_DS_data_Sa>0])/H_DS_data_Sa[H_DS_data_Sa>0])
    
    MAE_Sb = np.mean(np.abs(H_DS_pred_Sb[H_DS_data_Sb>0]-H_DS_data_Sb[H_DS_data_Sb>0]))
    MRE_Sb = np.mean(np.abs(H_DS_pred_Sb[H_DS_data_Sb>0]-H_DS_data_Sb[H_DS_data_Sb>0])/H_DS_data_Sb[H_DS_data_Sb>0])
    MAE_Sb_full = np.mean(np.abs(H_DS_pred_Sb_full[H_DS_data_Sb>0]-H_DS_data_Sb[H_DS_data_Sb>0]))
    MRE_Sb_full = np.mean(np.abs(H_DS_pred_Sb_full[H_DS_data_Sb>0]-H_DS_data_Sb[H_DS_data_Sb>0])/H_DS_data_Sb[H_DS_data_Sb>0])
    
    MAE_Sc = np.mean(np.abs(H_DS_pred_Sc[H_DS_data_Sc>0]-H_DS_data_Sc[H_DS_data_Sc>0]))
    MRE_Sc = np.mean(np.abs(H_DS_pred_Sc[H_DS_data_Sc>0]-H_DS_data_Sc[H_DS_data_Sc>0])/H_DS_data_Sc[H_DS_data_Sc>0])
    MAE_Sc_full = np.mean(np.abs(H_DS_pred_Sc_full[H_DS_data_Sc>0]-H_DS_data_Sc[H_DS_data_Sc>0]))
    MRE_Sc_full = np.mean(np.abs(H_DS_pred_Sc_full[H_DS_data_Sc>0]-H_DS_data_Sc[H_DS_data_Sc>0])/H_DS_data_Sc[H_DS_data_Sc>0])
    
    #%%% Report and Plot
    print(f'-- WITH FEATURE SELECT (n={cf.K_BEST}) --')
    print('-- Wolfs.B --')
    print(f"Mean absolute error (MAE): {MAE_Wb:.3f}")
    print(f"Mean relative error (MRE): {100*MRE_Wb:.3f}\%")
    print("##---------")  
    print('-- Sehn.A --')
    print(f"Mean absolute error (MAE): {MAE_Sa:.3f}")
    print(f"Mean relative error (MRE): {100*MRE_Sa:.3f}\%")
    print("##---------") 
    print('-- Sehn.B --')
    print(f"Mean absolute error (MAE): {MAE_Sb:.3f}")
    print(f"Mean relative error (MRE): {100*MRE_Sb:.3f}\%")
    print("##---------") 
    print('-- Sehn.C --')
    print(f"Mean absolute error (MAE): {MAE_Sc:.3f}")
    print(f"Mean relative error (MRE): {100*MRE_Sc:.3f}\%")
    print("##---------") 
    print(f'-- WITHOUT FEATURE SELECT (full wavenumber range) --')
    print('-- Wolfs.B --')
    print(f"Mean absolute error (MAE): {MAE_Wb_full:.3f}")
    print(f"Mean relative error (MRE): {100*MRE_Wb_full:.3f}\%")
    print("##---------")  
    print('-- Sehn.A --')
    print(f"Mean absolute error (MAE): {MAE_Sa_full:.3f}")
    print(f"Mean relative error (MRE): {100*MRE_Sa_full:.3f}\%")
    print("##---------") 
    print('-- Sehn.B --')
    print(f"Mean absolute error (MAE): {MAE_Sb_full:.3f}")
    print(f"Mean relative error (MRE): {100*MRE_Sb_full:.3f}\%")
    print("##---------") 
    print('-- Sehn.C --')
    print(f"Mean absolute error (MAE): {MAE_Sc_full:.3f}")
    print(f"Mean relative error (MRE): {100*MRE_Sc_full:.3f}\%")
    print("##---------") 
    
    # --- SETUP
    init_plot(size = 'full', page_width_cm = LNE_WDTH)    
    
    plt.rcParams['figure.figsize']=[plt.rcParams['figure.figsize'][0],
                                    plt.rcParams['figure.figsize'][1]*(5/3)]
    fig8 = plt.figure()    
    gs = GridSpec(5, 2, figure=fig8)
    ax8 = fig8.add_subplot(gs[0:3, 0])
    ax10 = fig8.add_subplot(gs[0:3, 1])
    ax9 = fig8.add_subplot(gs[3:5, :])
    
    import matplotlib.font_manager as font_manager
    monofont = font_manager.FontProperties(family='Consolas')
    
    # --- REGRESSION RESULTS FEATURE SELECT
    ax8.plot([0,10],[0,10],color='k')
    ax8.scatter(H_DS_data_Wb, H_DS_pred_Wb, edgecolors='k', color=cp.green, 
                alpha=0.7, zorder=3, label='Wolfs.B')  
    ax8.scatter(H_DS_data_Sa, H_DS_pred_Sa, edgecolors='k', color=cp.red, 
                alpha=0.7, zorder=3, label='Sehn.A') 
    ax8.scatter(H_DS_data_Sb, H_DS_pred_Sb, edgecolors='k', color=cp.blue, 
                alpha=0.7, zorder=3, label='Sehn.B')
    ax8.scatter(H_DS_data_Sc, H_DS_pred_Sc, edgecolors='k', color=cp.purple, 
                alpha=0.7, zorder=3, label='Sehn.C')
    ax8.text(0.02,0.02,'(a)', transform=ax8.transAxes,
              horizontalalignment='left', verticalalignment='bottom',
              bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    ax8.text(0.98,0.02,f'n-best ($n={cf.K_BEST}$)', transform=ax8.transAxes,
              horizontalalignment='right', verticalalignment='bottom',
              bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    
    ax8.set_xlabel('$\mathrm{DS_{1H}}$ ("Truth") / $-$')
    ax8.set_ylabel('$\mathrm{DS_{IR,ML}}$ / $-$')
    ax8.grid(True)
    ax8.set_xlim([0.5,3])
    ax8.set_ylim([0.5,3])
    plt.tight_layout() 
    
    ax8.legend(prop=monofont)
    
    # --- REGRESSION RESULTS FULL RANGE
    ax10.plot([0,10],[0,10],color='k')
    ax10.scatter(H_DS_data_Wb, H_DS_pred_Wb_full, edgecolors='k', color=cp.green, 
                alpha=0.7, zorder=3, label='Wolfs.B')  
    ax10.scatter(H_DS_data_Sa, H_DS_pred_Sa_full, edgecolors='k', color=cp.red, 
                alpha=0.7, zorder=3, label='Sehn.A') 
    ax10.scatter(H_DS_data_Sb, H_DS_pred_Sb_full, edgecolors='k', color=cp.blue, 
                alpha=0.7, zorder=3, label='Sehn.B')
    ax10.scatter(H_DS_data_Sc, H_DS_pred_Sc_full, edgecolors='k', color=cp.purple, 
                alpha=0.7, zorder=3, label='Sehn.C')
    ax10.text(0.02,0.02,'(b)', transform=ax10.transAxes,
              horizontalalignment='left', verticalalignment='bottom',
              bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    ax10.text(0.98,0.02,'no feature select', transform=ax10.transAxes,
              horizontalalignment='right', verticalalignment='bottom',
              bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    
    ax10.set_xlabel('$\mathrm{DS_{1H}}$ ("Truth") / $-$')
    ax10.set_ylabel('$\mathrm{DS_{IR,ML}}$ / $-$')
    ax10.grid(True)
    ax10.set_xlim([0.5,3])
    ax10.set_ylim([0.5,3])
    plt.tight_layout() 
    
    ax10.legend(prop=monofont)
    
    # --- IR DATA
    # Re-Load FS   
    scat = ax9.scatter(wn_fs, np.zeros(wn_fs.shape), marker='o', color='k', 
                       label=r'selected $\nu$', 
                       facecolor=cm.Greys(0.2), alpha=0.4, zorder=5)
    ax9.plot(wn_full, 100*IR_data_full[10,:], linewidth=1, color='k', zorder=5, label='Wolfs.A')
    ax9.plot(wn_full, 100*IR_data_Wb[0,:], linewidth=1, color=cp.green, zorder=4, label='Wolfs.B')
    ax9.plot(wn_full, 100*IR_data_Sa[5,:], linewidth=1, color=cp.red, zorder=4, label='Sehn.A')
    ax9.plot(wn_full, 100*IR_data_Sb[4,:], linewidth=1, color=cp.blue, zorder=4, label='Sehn.B')
    ax9.plot(wn_full, 100*IR_data_Sc[2,:], linewidth=1, color=cp.purple, zorder=4, label='Sehn.C')
        
    ax9.text(0.005,0.97,'(c)',transform=ax9.transAxes, horizontalalignment='left',
             verticalalignment='top', zorder=99,
             bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    ax9.set_xlabel(r'Wavenumber $\nu$ / cm$^{-1}$')
    ax9.set_ylabel('Transmission / $\%$')
    ax9.set_xlim(min(wn_full),max(wn_full))
    ax9.set_ylim(-9,109)
    ax9.invert_xaxis()
    ax9.grid(True)
    
    plt.tight_layout()
    
    box = ax9.get_position()
    ax9.set_position([box.x0, box.y0, box.width*0.83, box.height])
    
    ax9.legend()
    ref_handle, _ = ax9.get_legend_handles_labels()  
    l1 = plt.legend([scat], [r'selected $\nu$'+'\n'+'in (a)'], bbox_to_anchor=(1.02, 0.43),
                loc='upper left', 
                fontsize='small', framealpha=1, ncol=1)
    ax9.add_artist(l1)
    ax9.legend(handles=ref_handle[1:], bbox_to_anchor=(1.02, 1.02),
                loc='upper left', prop=monofont,
                fontsize='small', framealpha=1, ncol=1) 
        
    if EXPORT:
        plt.savefig("exp/IR_ML_Extrapolation.png", dpi=300)        
        plt.savefig("exp/IR_ML_Extrapolation.pdf")