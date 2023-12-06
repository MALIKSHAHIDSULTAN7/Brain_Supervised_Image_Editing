import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
########################################## Utility functions ############################

#########################################################################################
####################################### Reading Data ####################################
#########################################################################################

def read_data_csv(fpath,labels_path):
    """
    Takes input the file path csv
    Ouputs the data frame for the data
    """
    data = pd.read_csv(fpath)
    #########################################
    ############ counting total images shown#
    trig = data['Trigger']
    count_stims = 0
    for i in np.arange(0,len(trig)):
        if i+1 < len(trig):
            if trig[i] != trig[i+1]:
                count_stims += 1
    
    #########################################
    count_stims  = count_stims//2
    labels_data  = pd.read_csv(labels_path)
    labels_data  = labels_data.loc[:,['response','is_target']]
    count_images = labels_data.shape[0]
    assert count_stims == count_images, "The files are not same !!! Search for relevant files" 
    # Write function to match the images and the trigger. 
    flag = False
    i = 0
    j = 0
    event = data.columns.get_loc("Event")
    while i<len(trig):
        if trig[i] == 16:
            n = 300
            flag = True
            while flag:
                if trig[i:i+n].sum() != n*16:
                    n -= 1
                else:
                    flag = False
                    print(n)
            
            if labels_data.iloc[j,0] == 'space':
                if labels_data.iloc[j,1] == 1:
                    data.iloc[i:i+n,event] = 'correctly_identified'
                elif labels_data.iloc[j,1] == 0:
                    data.iloc[i:i+n,event] == 'incorrectly_identified'
            elif labels_data.iloc[j,0] == None:
                if labels_data.iloc[j,1] == 1:
                    data.iloc[i:i+n,event] = 'not_identified'
                elif labels_data.iloc[j,1] == 0:
                    data.iloc[i:i+n,event] = 'correclty_not_identified'
            j += 1
            i  = i+n
        else:
            data.iloc[i,event] = 'Rest'
            i+=1
        
      
    return data, count_stims, count_images

fpath = "/home/sultm0a/Documents/eeg_data_image_editing/12_oct/K10_12_100i2b_0001_raw.csv"
labels_path =  "/home/sultm0a/Documents/eeg_data_image_editing/12_oct/kilich_oddball_test_2023-10-12_19h36.41.191_1.csv"



data, stims,actual_shown  = read_data_csv(fpath, labels_path)
data.to_csv("/home/sultm0a/Documents/eeg_data_image_editing/12_oct/kilich_oddball_test_2023-10-12_19h36.41.organised.csv")

print(data.head())
print("Total Images shown in this experiment = {}".format(stims))
print("Total Images shown in this experiment = {}".format(actual_shown))
sampling_rate = 300;

"""
df = data.iloc[:,1:25]

raw_array = mne.io.RawArray(df.values.T, 
                            info=mne.create_info(ch_names=df.columns.tolist(), sfreq=sampling_rate))
raw_array.set_channel_types = 'eeg'

fig  = raw_array.compute_psd(tmax = np.inf, fmax  = 150,picks = 'Fp2').plot(picks = 'Fp2')
for ax in fig.axes[0:]:
    freqs = ax.lines[-1].get_xdata()
    psds  = ax.lines[-1].get_ydata()
    for freq in (60,120):
        idx = np.searchsorted(freqs,freq)
        ax.arrow(
            x = freqs[idx],
            y = psds[idx] + 18,
            dx = 0,
            dy = -12,
            color = "red",
            width = 0.1,
            head_width = 3,
            length_includes_head = True
        )

plt.show()

"""