import mne
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def get_curry0408_withbaseSMOTE():
    """
    left, right, base, 3 classes with SOMTE DA
    :return:
    train_data, train_label, test_data, test_label
    data shapes (-1, 4096, 2)
    label onehot (0, 0, 1)
    """
    fn = r'D:\00-data\220408\ori\Acquisition 21.cdt'
    """raw part"""
    raw = mne.io.read_raw_curry(fn, verbose='ERROR')
    raw.drop_channels(['Trigger'])
    raw.load_data()
    raw.notch_filter(np.arange(50, 401, 50), verbose='ERROR')
    raw.filter(0.1, None, verbose='ERROR')
    # raw.plot_psd()

    """ica"""
    ica = mne.preprocessing.ICA(n_components=11)
    ica.fit(raw, verbose='ERROR')
    # ica.plot_components()
    ica.exclude = [5, 6, 8]
    ica.apply(raw, verbose='ERROR')

    ch_pick = ['F3', 'F4',
               'FC3', 'FC4',
               'C3', 'C4',
               'CP3', 'CP4']

    """epochs"""
    event_id = {'20': 1, '21': 2}  # base : 0
    events, _ = mne.events_from_annotations(raw, event_id, verbose='ERROR')
    events_with_base = np.empty((0, 3)).astype('int32')
    for e in events:
        events_with_base = np.concatenate((events_with_base,
                                           e.reshape((1, 3)),
                                           np.array([e[0] + 4500, 0, 0]).reshape((1, 3))), 0)
    epochs = mne.Epochs(raw, events_with_base, tmin=1 + 1 / 1024, tmax=5, baseline=None, picks=ch_pick, preload=True,
                        verbose='ERROR')

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
    # save_path = os.path.dirname(os.path.split(fn)[0]) + '\\'  # 上级

    data = data.swapaxes(1, 2)  # shape like (time_points, 2)

    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)

    def transform_data(x, y, chs=8):
        reshaped_x = np.empty((0, x.shape[1], 2))
        reshaped_y = np.empty(0)
        for i in range(0, chs, 2):
            reshaped_x = np.concatenate((reshaped_x, x[:, :, i:i + 2]))
            reshaped_y = np.concatenate((reshaped_y, y))

        reshaped_x = reshaped_x.reshape(reshaped_x.shape[0], reshaped_x.shape[1] * reshaped_x.shape[2])
        reshaped_x, reshaped_y = SMOTE(random_state=42).fit_resample(reshaped_x, reshaped_y)
        reshaped_x = reshaped_x.reshape(reshaped_x.shape[0], -1, 2)

        reshaped_y = to_categorical(reshaped_y)
        return reshaped_x, reshaped_y

    train_data, train_label = transform_data(train_data, train_label)
    test_data, test_label = transform_data(test_data, test_label)
    return train_data, train_label, test_data, test_label


def get_curry0408_41DCNN():
    """
    left, right, 2 classes
    :return:
    train_data, train_label, test_data, test_label
    data shapes (-1, 4096, 2)
    label onehot (0, 1)
    """
    fn = r'D:\00-data\220408\ori\Acquisition 21.cdt'
    """raw part"""
    raw = mne.io.read_raw_curry(fn, verbose='ERROR')
    raw.drop_channels(['Trigger'])
    raw.load_data()
    raw.notch_filter(np.arange(50, 401, 50), verbose='ERROR')
    raw.filter(0.1, None, verbose='ERROR')
    # raw.plot_psd()

    """ica"""
    ica = mne.preprocessing.ICA(n_components=11)
    ica.fit(raw, verbose='ERROR')
    # ica.plot_components()
    ica.exclude = [5, 6, 8]
    ica.apply(raw, verbose='ERROR')

    ch_pick = ['F3', 'F4',
               'FC3', 'FC4',
               'C3', 'C4',
               'CP3', 'CP4']

    """epochs"""
    event_id = {'20': 0, '21': 1}  # base : 0
    events, _ = mne.events_from_annotations(raw, event_id, verbose='ERROR')
    epochs = mne.Epochs(raw, events, tmin=1 + 1 / 1024, tmax=5, baseline=None, picks=ch_pick, preload=True,
                        verbose='ERROR')

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
    data = data.swapaxes(1, 2)  # shape like (time_points, 2)

    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)

    def transform_data(x, y, chs=8):
        reshaped_x = np.empty((0, x.shape[1], 2))
        reshaped_y = np.empty(0)
        for i in range(0, chs, 2):
            reshaped_x = np.concatenate((reshaped_x, x[:, :, i:i + 2]))
            reshaped_y = np.concatenate((reshaped_y, y))

        reshaped_y = to_categorical(reshaped_y)
        return reshaped_x, reshaped_y

    train_data, train_label = transform_data(train_data, train_label)
    test_data, test_label = transform_data(test_data, test_label)
    print('data loaded...')
    return train_data, train_label, test_data, test_label


def get_curry0408():
    """
    left, right, 2 classes
    :return:
    train_data, train_label, test_data, test_label
    data shapes (-1, 1, ch, timepoints)
    label onehot (0, 1)
    """
    fn = r'D:\00-data\220408\ori\Acquisition 21.cdt'
    """raw part"""
    raw = mne.io.read_raw_curry(fn, verbose='ERROR')
    raw.drop_channels(['Trigger'])
    raw.load_data()
    raw.notch_filter(np.arange(50, 401, 50), verbose='ERROR')
    raw.filter(0.1, None, verbose='ERROR')
    # raw.plot_psd()

    """ica"""
    ica = mne.preprocessing.ICA(n_components=11)
    ica.fit(raw, verbose='ERROR')
    # ica.plot_components()
    ica.exclude = [5, 6, 8]
    ica.apply(raw, verbose='ERROR')

    ch_pick = ['F3', 'F4',
               'FC3', 'FC4',
               'C3', 'C4',
               'CP3', 'CP4']

    """epochs"""
    event_id = {'20': 0, '21': 1}  # base : 0
    events, _ = mne.events_from_annotations(raw, event_id, verbose='ERROR')
    epochs = mne.Epochs(raw, events, tmin=1 + 1 / 1024, tmax=5, picks=ch_pick, baseline=None, preload=True,
                        verbose='ERROR')

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
    labels = to_categorical(labels)
    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)

    print('data loaded...')
    return train_data, train_label, test_data, test_label


def get_curry0413():
    """
    left, right, tongue, feet  4 classes
    :return:
    train_data, train_label, test_data, test_label
    data shapes (-1, 1, chs, 4096)
    label onehot (0, 0, 0, 1)
    """
    fn = r'D:\00-data\220413\Acquisition 23.cdt'
    ch_pick = [
        [
            'Fp1', 'Fp2',
            'F11', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F12',
            'FC3', 'FCz', 'FC4',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'CP3', 'CPz', 'CP4',
            'P3', 'Pz', 'P4',
            'O1', 'Oz', 'O2',
        ],
        [
            'F3', 'F4',
            'FC3', 'FC4',
            'C3', 'C4',
            'CP3', 'CP4',
        ]
    ]
    # ch_pick = ['C3', 'C4', ]
    """raw part"""
    raw = mne.io.read_raw_curry(fn, verbose='ERROR')
    raw.pick(ch_pick[1], verbose='ERROR')
    raw.load_data()
    raw.notch_filter(np.arange(50, 151, 50), verbose='ERROR')
    raw.filter(0.1, 140, verbose='ERROR')

    # raw.plot_psd()

    """ica"""
    # ica = mne.preprocessing.ICA(n_components=22)
    # ica.fit(raw, verbose='ERROR')
    # # ica.plot_components()
    # ica.exclude = [17, 19]
    # ica.apply(raw, verbose='ERROR')

    """epochs"""
    event_id = {'40': 0, '41': 1, '42': 2, '43': 3}
    events, _ = mne.events_from_annotations(raw, event_id, verbose='ERROR')
    epochs = mne.Epochs(raw, events, tmin=2 + 1 / 1024, tmax=6, baseline=None, preload=True,
                        verbose='ERROR')

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
    labels = to_categorical(labels)
    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)

    print('data loaded...')
    return train_data, test_data, train_label, test_label


if __name__ == '__main__':
    res = get_curry0413()
    for r in res:
        print(r.shape)
