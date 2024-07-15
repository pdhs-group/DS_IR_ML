import matplotlib.pyplot as plt

#%% Define KIT colors
transparent = (1.0, 1.0, 1.0, 0.0)
white = (1.0, 1.0, 1.0)
black='#000000'
gray='#404040'
green='#009682'
blue='#4664aa'
red='#a22223'
yellow='#fce500'
orange='#df9b1b'
lightgreen='#8cb63c'
purple='#a3107c'
brown='#a7822e'
cyan='#23a1e0'

def init_plot(scl_a4=1, page_lnewdth_cm=16.5, scl=1, fnt='arial',figsze=[6.4,4.8], 
              mrksze=6, lnewdth=1.5, fontsize = 10, labelfontsize=9, tickfontsize=8):
    
        # --- Initialize defaults ---
        plt.rcdefaults()
        
        # --- Calculate figure size in inches ---
        # scl_a4=2: Half page figure
        if scl_a4==2:     
                fac=scl*page_lnewdth_cm/(2.54*figsze[0]*2) #2.54: cm --> inch
                figsze=[figsze[0]*fac,figsze[1]*fac]
        
        # scl_a4=1: Full page figure
        elif scl_a4==1:
                fac=scl*page_lnewdth_cm/(2.54*figsze[0]) #2.54: cm --> inch
                figsze=[figsze[0]*fac,figsze[1]*fac]
        
        # --- Adjust legend ---
        plt.rc('legend',fontsize=fontsize*scl,fancybox=True, shadow=False, edgecolor='k', 
               handletextpad=0.2, handlelength=1, borderpad=0.2, labelspacing=0.2, columnspacing=0.2)
        
        # --- General plot setup ---
        plt.rc('mathtext', fontset='cm')
        plt.rc('font', family=fnt)
        plt.rc('xtick', labelsize=tickfontsize*scl)
        plt.rc('ytick', labelsize=tickfontsize*scl)
        plt.rc('axes', labelsize=labelfontsize*scl, linewidth=0.5*scl, titlesize=labelfontsize*scl)
        plt.rc('legend', fontsize=fontsize*scl)
        plt.rc('axes', axisbelow=True) # Grid lines in Background
        plt.rcParams['lines.markersize']=mrksze
        plt.rcParams['hatch.linewidth']=lnewdth/2
        plt.rcParams['lines.linewidth']=lnewdth     
        plt.rcParams['figure.figsize']=figsze