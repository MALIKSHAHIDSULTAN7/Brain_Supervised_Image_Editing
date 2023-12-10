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
    labels_data['response'].fillna(value = 'not_cat',inplace=True)
    
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
                    print('Gone 1')
                elif labels_data.iloc[j,1] == 0:
                    data.iloc[i:i+n,event] = 'incorrectly_identified'
                    print('Gone 2')

            elif labels_data.iloc[j,0] == "not_cat":
                
                if labels_data.iloc[j,1]   == 1:
                    data.iloc[i:i+n,event] = 'not_identified'
                    print('Gone 3')
                elif labels_data.iloc[j,1] == 0:
                    data.iloc[i:i+n,event] = 'correclty_not_identified'
                    print('Gone 4')
            print('Index {}'.format(i))
            print('Case  {}'.format(j))
            j += 1
            i  = i+n
        elif trig[i] == 0:
            data.iloc[i,event] = 'Rest'
            i+=1
        
      
    return data, count_stims, count_images

fpath = "/home/sultm0a/Documents/eeg_data_image_editing/12_oct/K10_12_100i2b_0001_raw.csv"
labels_path =  "/home/sultm0a/Documents/eeg_data_image_editing/12_oct/kilich_oddball_test_2023-10-12_19h36.41.191_1.csv"




data, stims,actual_shown  = read_data_csv(fpath, labels_path)
data.to_csv("/home/sultm0a/Documents/eeg_data_image_editing/12_oct/kilich_oddball_test_2023-10-12_19h36.41.organised.csv")

req_data  = ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Fp1',
       'Fp2', 'T3', 'T5', 'O1', 'O2', 'F7', 'F8', 'T6',
       'T4', 'Pz','Trigger']
ch_types = ['eeg' if i != 'Trigger' else 'stim' for i in req_data]

def re_reference_to_LE(data):
    ref_df = data.copy()
    ref_df['LE'] = (ref_df['A1'] + ref_df['A2'])/2
    channels = ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Fp1',
       'Fp2', 'T3', 'T5', 'O1', 'O2', 'F7', 'F8', 'T6',
       'T4', 'Pz']
    ref_df[channels] = ref_df[channels].values - ref_df['LE'].values.reshape(-1, 1)
    return ref_df



print(data.head())
print("Total Images shown in this experiment = {}".format(stims))
print("Total Images shown in this experiment = {}".format(actual_shown))
sampling_rate = 300;


df = re_reference_to_LE(data)
df = df[req_data]
print(df.head())

montage = mne.channels.make_standard_montage('standard_1005')
ch_names = df.columns.tolist()
raw_array = mne.io.RawArray(df.values.T, 
                            info=mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types))
raw_array.set_montage(montage)



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

event_labels = data['Event'].values

events = []

# Define event_id mapping
event_id_mapping = {'Rest': 0, 'correctly_identified': 1, 
                    "correclty_not_identified":2,
                    'incorrectly_identified': 3,
                    'not_identified':4
                    }  # Add more events as needed


# Identify the onset and offset of each event
events_onset = []
events_offset = []
current_event = event_labels[0]
onset = 0

for i, label in enumerate(event_labels[1:], start=1):
    if label != current_event:
        events_onset.append(onset)
        events_offset.append(i - 1)
        onset = i
        current_event = label

# Add the last event
events_onset.append(onset)
events_offset.append(len(event_labels) - 1)

print(event_labels[0])

print(len(events_onset))

print(len(events_offset))

print(event_labels[events_onset])

# Create events as (onset, duration, event_id)
events = [[onset, offset - onset + 1, event_id_mapping[current_event]] for onset, offset, current_event in zip(events_onset, events_offset, event_labels[events_onset])]

print(events)

events_array = np.array(events)
events_array[:,1] = 0
events = mne.find_events(raw_array,'Trigger', initial_event='False',consecutive=False)
print(50*'#')
print(events)
print(events.shape)
epochs = mne.Epochs(raw_array, events, tmin=-0.2, tmax= 0.4, baseline= (None, 0), preload=True)


fig = epochs.plot(picks = 'Fp2', events = events_array)
plt.show()
raw_ica  = epochs.copy()

random_state = 32   # ensures ICA is reproducable each time it's run
ica_n_components = .99     # Specify n_components as a decimal to set % explained variance
ica = mne.preprocessing.ICA(
n_components=0.99, method="fastica", max_iter="auto", random_state=97)
ica.fit(raw_ica)
ica.plot_components()
plt.show()