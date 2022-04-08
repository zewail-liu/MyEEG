import mne
import numpy as np
import os
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical


def process_curry_0408():
    fn = r'D:\00-data\220408\ori\Acquisition 21.cdt'
    """raw part"""
    raw = mne.io.read_raw_curry(fn)
    raw.drop_channels(['Trigger'])
    raw.load_data()
    raw.notch_filter(np.arange(50, 401, 50))
    raw.filter(0.1, None)
    # raw.plot_psd()

    """ica"""
    ica = mne.preprocessing.ICA(n_components=11)
    ica.fit(raw)
    # ica.plot_components()
    ica.exclude = [5, 6, 8]
    ica.apply(raw)

    ch_pick = ['F3',
               'F4',
               'FC3',
               'FC4',
               'C3',
               'C4',
               'CP3',
               'CP4']
    raw.pick(ch_pick)

    """epochs"""
    event_id = {'20': 0, '21': 1}
    events, _ = mne.events_from_annotations(raw, event_id)
    epochs = mne.Epochs(raw, events, tmin=1 + 1 / 1024, tmax=5, baseline=None, preload=True)

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
    data = data[:, np.newaxis, :, :]
    # labels = to_categorical(labels)  # one-hot
    save_path = os.path.dirname(os.path.split(fn)[0]) + '\\'  # 上级
    np.save(save_path + 'data.npy', data)
    np.save(save_path + 'labels.npy', labels)


if __name__ == '__main__':
    process_curry_0408()
