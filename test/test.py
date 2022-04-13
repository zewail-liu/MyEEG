import mne
from sklearn import preprocessing
import numpy as np
from keras.utils.np_utils import to_categorical

rename_map = {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2',
              'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz',
              'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1',
              'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1',
              'EEG-Pz': 'Pz', 'EEG-15': 'P2', 'EEG-16': 'POz'}

raw = mne.io.read_raw_eeglab('1.set')
raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
raw.crop(367)
raw.filter(0.1, None)

"""ica"""
ica = mne.preprocessing.ICA(n_components=16)
ica.fit(raw)
# ica.plot_components()
ica.exclude = [1, 3, 4]
ica.apply(raw)
ch_pick = ['FC1',
           'FC2',
           'FC3',
           'FC4',
           'C1',
           'C2',
           'C3',
           'C4',
           'CP1',
           'CP2',
           'CP3',
           'CP4']
raw.pick(ch_pick)

"""epochs"""
event_id = {'class1, Left hand\t- cue onset (BCI experiment)': 0,
            'class2, Right hand\t- cue onset (BCI experiment)': 1,
            'class3, Foot, towards Right - cue onset (BCI experiment)': 2,
            'class4, Tongue\t\t- cue onset (BCI experiment)': 3}
events, _ = mne.events_from_annotations(raw, event_id)
epochs = mne.Epochs(raw, events, tmin=-0.5 + 1 / 250, tmax=4, baseline=None, preload=True)

"""scaler"""
scaler = preprocessing.StandardScaler()
data = epochs.get_data()
labels = epochs.events
labels = labels[:, 2]
# scaler进行列标准化; among channels > inside channel
for i in range(len(data)):
    scaler.fit(data[i])
    data[i] = scaler.transform(data[i])

"""save"""
data = data.swapaxes(1, 2)
reshaped_data = np.empty((0, 1125, 2))
for i in range(len(ch_pick) // 2):
    reshaped_data = np.concatenate((reshaped_data, data[:, :, i * 2:i * 2 + 2]))

labels_onehot = to_categorical(labels)  # one-hot
labels_onehot = np.concatenate(
    (labels_onehot, labels_onehot, labels_onehot, labels_onehot, labels_onehot, labels_onehot), 0)
np.save('bci_iv_data_1D.npy', reshaped_data)
np.save('bci_iv_labels_onehot_1D.npy', labels_onehot)
