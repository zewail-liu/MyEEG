import mne
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

"""
    bci_iv_2a数据集，22通道EEG，3通道EOG，250hz采样频率，提示到运动想象结束持续4s，288次
    T为训练数据，E为评估数据
    有效标签：
        769 0x0301 Cue onset left (class 1)
        770 0x0302 Cue onset right (class 2)
        771 0x0303 Cue onset foot (class 3)
        772 0x0304 Cue onset tongue (class 4)
        1023 0x03FF Rejected trial

"""
rename_mapping = {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2',
                  'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz',
                  'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1',
                  'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1',
                  'EEG-Pz': 'Pz', 'EEG-15': 'P2', 'EEG-16': 'POz'}
data_path = r'D:\00-data\BCI_IV_2a\ori\\'

ch_pos = np.array([[0.00000000e+00, 6.07384809e+01, 5.94629038e+01],
                   [-5.92749782e+01, 3.09552850e+01, 5.24713950e+01],
                   [-3.23513771e+01, 3.24361839e+01, 7.15980612e+01],
                   [0.00000000e+00, 3.29278836e+01, 7.83629662e+01],
                   [3.23513771e+01, 3.24361839e+01, 7.15980612e+01],
                   [5.92749782e+01, 3.09552850e+01, 5.24713950e+01],
                   [-8.08315480e+01, 4.94950483e-15, 2.62918398e+01],
                   [-6.31712807e+01, 3.86812534e-15, 5.68716915e+01],
                   [-3.45373740e+01, 2.11480423e-15, 7.76670445e+01],
                   [0.00000000e+00, 5.20474890e-15, 8.50000000e+01],
                   [3.46092032e+01, 2.11920249e-15, 7.76350633e+01],
                   [6.31673102e+01, 3.86788221e-15, 5.68761015e+01],
                   [8.08315480e+01, 4.94950483e-15, 2.62918398e+01],
                   [-5.92749782e+01, -3.09552850e+01, 5.24713950e+01],
                   [-3.23513771e+01, -3.24361839e+01, 7.15980612e+01],
                   [4.03250273e-15, -3.29278836e+01, 7.83629662e+01],
                   [3.23513771e+01, -3.24361839e+01, 7.15980612e+01],
                   [5.92749782e+01, -3.09552850e+01, 5.24713950e+01],
                   [-2.60420934e+01, -5.99127302e+01, 5.43808250e+01],
                   [7.43831863e-15, -6.07384809e+01, 5.94629038e+01],
                   [2.60254380e+01, -5.98744128e+01, 5.44309771e+01],
                   [9.67783732e-15, -7.90255389e+01, 3.13043800e+01]])


def get_bci_iv_2a(subject: int):
    """
    :param subject: SN of subject : [1,9]
    :return:
    """
    raw = mne.io.read_raw_gdf(data_path + f'A0{subject}T.gdf', verbose='ERROR')
    raw.rename_channels(rename_mapping)
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    raw._set_channel_positions(ch_pos, raw.ch_names)
    raw.crop(367)
    raw.load_data()
    raw.filter(0.1, None, verbose='ERROR')
    """ica"""
    # ica = mne.preprocessing.ICA(n_components=13)
    # ica.fit(raw)
    # ica.plot_components()
    # ica.exclude = [1, 3, 4]
    # ica.apply(raw)
    ch_pick = ['FC1', 'FC2', 'FC3', 'FC4',
               'C1', 'C2', 'C3', 'C4',
               'CP1', 'CP2', 'CP3', 'CP4']
    # raw.pick(ch_pick)

    """epochs"""
    event_id = {'769': 0, '770': 1, '771': 2, '772': 3}
    events, _ = mne.events_from_annotations(raw, event_id, verbose='ERROR')
    epochs = mne.Epochs(raw, events, tmin=-0.5 + 1 / 250, tmax=4, baseline=None, preload=True, verbose='ERROR')

    """scaler"""
    scaler = preprocessing.StandardScaler()
    data = epochs.get_data()
    labels = epochs.events[:, 2]
    # scaler进行列标准化; among channels > inside channel
    for i in range(len(data)):
        scaler.fit(data[i])
        data[i] = scaler.transform(data[i])

    """save"""
    data = data[:, np.newaxis, :, :]
    labels = to_categorical(labels)  # one-hot
    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(train_data.shape)
    print(train_label.shape)
    return train_data, test_data, train_label, test_label


if __name__ == '__main__':
    get_bci_iv_2a(3)
