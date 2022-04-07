import mne
from mne.io import read_raw_eeglab
from mne.preprocessing import ICA

raw = read_raw_eeglab('../test/iv_2a_withloc.set', preload=True)
raw.drop_channels(['Fp1', 'Fp2', 'Fpz'])
event_dict = {'class1, Left hand\t- cue onset (BCI experiment)': 0,
              'class2, Right hand\t- cue onset (BCI experiment)': 1}
events, _ = mne.events_from_annotations(raw, event_id=event_dict)
epochs = mne.Epochs(raw, events, tmin=1 / 250, tmax=4, baseline=None, preload=True)
epochs.save('test_ori.fif', overwrite=True)

# 滤波
raw = raw.filter(l_freq=1, h_freq=50)

# ICA
ica = ICA(n_components=19)
ica.fit(raw)
ica.exclude = [0]  # EOG
ica.apply(raw)

epochs = mne.Epochs(raw, events, tmin=1 / 250, tmax=4, baseline=None, preload=True)
epochs.save('test_after.fif', overwrite=True)
