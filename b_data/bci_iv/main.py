from mne.io import concatenate_raws, read_raw_edf
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import mne
import numpy as np

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


class ProcessData:
    def __init__(self, subjects_start=1, subjects_end=1, round_start=3, round_end=14):
        self.subjects_start = subjects_start
        self.subjects_end = subjects_end
        self.round_start = round_start
        self.round_end = round_end
        self.filename = f'S{self.subjects_start}-{self.subjects_end}'

        self.raw = self.collect_data()
        self.epochs = self.filter_ica()
        self.make_labels(self.epochs)

    def collect_data(self):
        for subject in range(self.subjects_start, self.subjects_end + 1):
            for R in range(self.round_start, self.round_end + 1):
                fn = '../b_data/bci_iv/' + 'A' + '%02d' % subject + '.gdf'
                if subject == 1 and R == 3:
                    raw = read_raw_edf(fn)
                else:
                    tem = read_raw_edf(fn)
                    raw.append(tem)
        # events, _ = mne.events_from_annotations(raw)
        # epochs = mne.Epochs(raw, events, tmin=1 / 160, tmax=4, baseline=None)
        # epochs.save(self.filename + '_original_epo.fif')
        return raw

    def filter_ica(self):
        raw = self.raw
        raw.load_data()

        # 滤波
        raw = raw.filter(l_freq=1, h_freq=None)

        # ICA
        ica = ICA(n_components=20)
        ica.fit(raw)
        ica.exclude = [0]  # EOG
        ica.apply(raw)

        # epoch
        events, _ = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, tmin=1 / 160, tmax=4, baseline=None, preload=True)
        epochs.save(self.filename + '_ICA_epo.fif')
        return epochs

    def make_labels(self, epochs):
        e = epochs
        l = e.events[:, 2]
        labels = np.empty(shape=(0, 3))
        for i in range(len(l)):
            # label b'T0' -> (数据处理.md, 0, 0, 0)
            label = np.zeros((1, 3))
            label[0, l[i] - 1] = 1
            labels = np.concatenate((labels, label), 0)
        np.save(self.filename + '_labels', labels)


if __name__ == '__main__':
    # p = ProcessData(subjects_start=1, subjects_end=10)
    filename = 'bci_iv_2a'

    raw = mne.io.read_raw_gdf('A01T.gdf')
    raw.load_data()
    event_dict = {'769': 0, '770': 1, '771': 2, '772': 3}
    events, _ = mne.events_from_annotations(raw, event_id=event_dict)
    epochs = mne.Epochs(raw, events, tmin=1 / 250, tmax=4, baseline=None, preload=True)
    epochs = epochs.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    epochs.save(filename + '_epo.fif', overwrite=True)

    labels = np.empty(shape=(0, 4))
    for i in range(len(events)):
        # label b'T0' -> (1, 0, 0, 0)
        label = np.zeros((1, 4))
        label[0, events[i][2]] = 1
        labels = np.concatenate((labels, label), 0)
    np.save(filename + '_labels', labels)
