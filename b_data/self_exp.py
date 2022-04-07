import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def raw_part():
    raw = mne.io.read_raw_curry(r'D:\00-data\220304\Acquisition 18.cdt')

    raw = raw.pick(['Pz', 'P3', 'CP4', 'CPz', 'CP3', 'C4', 'Cz', 'C3', 'FC3', 'FCz', 'FC4'])
    raw.crop(126)
    raw.load_data()
    raw.notch_filter(np.arange(50, 250, 50))
    # raw.plot_psd()
    raw.filter(0.1, 200)
    # raw.plot_psd()

    ica = mne.preprocessing.ICA(n_components=11)
    ica.fit(raw)
    ica.plot_components()
    ica.exclude = [0, 1, 2, 3]
    ica.apply(raw)
    raw.save('raw.fif')


def epoch_part():
    raw = mne.io.read_raw_fif('raw.fif')
    event_id = {'1': 0, '2': 1}
    events, _ = mne.events_from_annotations(raw, event_id)
    epochs = mne.Epochs(raw, events, tmin=1 + 1 / 1024, tmax=5, baseline=None, preload=True)

    epochs.save('_-epo.fif')

    scaler = preprocessing.StandardScaler()  # (x,y,z)
    data = epochs.get_data()
    labels = epochs.events
    labels = labels[:, 2]
    data = data.swapaxes(1, 2)  # scaler进行列标准化
    for i in range(len(data)):
        scaler.fit(data[i])
        data[i] = scaler.transform(data[i])
    data = data.swapaxes(1, 2)
    data = data[:, np.newaxis, :, :]
    np.save('data', data)
    np.save('labels', labels)


if __name__ == '__main__':
    epoch_part()
