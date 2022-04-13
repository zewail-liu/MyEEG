import scipy.io as scio
from torch.utils.data import Dataset
import numpy as np
import mne


class Dataset_BCI_ii(Dataset):
    """
        bci_ii_Data set III数据集，3通道EEG，128hz采样频率，提示到运动想象结束持续9s，140次, 0.5-30Hz filter
        左右手两分类
    """

    def __init__(self):
        data_file = r'/bci_ii/dataset_BCIcomp1.mat'
        self.data = scio.loadmat(data_file)
        self.x = self.data['x_train'].swapaxes(0, 2).astype('float32')
        self.x = self.x[:, np.newaxis, :, :]
        self.data_shape = self.x.shape
        self.y = self.data['y_train'] - 1

    def __getitem__(self, item):
        return self.x[item], self.y[item, 0]

    def __len__(self):
        return len(self.x)


class Dataset_BCI_iv_2a(Dataset):
    def __init__(self):
        data_file = r'1_epo.fif'
        self.data = mne.read_epochs(data_file)
        self.x = self.data.get_data().astype('float32')
        self.x = self.x[:, np.newaxis, :, :]
        self.data_shape = self.x.shape
        self.y = self.data.events.astype('int64')[:, 2]

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


class DatasetTest(Dataset):
    def __init__(self):
        self.x = np.load(r'C:\Users\aaze\PycharmProjects\EEGnet\2.npy').astype('float32')
        self.x = self.x[:, np.newaxis, :, :]
        self.y = np.load(r'C:\Users\aaze\PycharmProjects\EEGnet\2_label.npy').astype('int64')

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.y)


class Dataset_experiment_0304(Dataset):
    def __init__(self):
        self.x = np.load(r'C:\Users\aaze\PycharmProjects\EEGnet\b_data\data.npy').astype('float32')
        self.y = np.load(r'C:\Users\aaze\PycharmProjects\EEGnet\b_data\labels.npy').astype('int64')

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


class Dataset_experiment_0408(Dataset):
    def __init__(self):
        self.x = np.load(r'D:\00-data\220408\data.npy').astype('float32')
        self.y = np.load(r'D:\00-data\220408\labels.npy').astype('int64')

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


def dataset_test(module: str):
    if module == 'bci_ii':
        m = Dataset_BCI_ii()
    elif module == 'bci_iv_2a':
        m = Dataset_BCI_iv_2a()
    else:
        return
    x, y = m[:]
    print('**********  ' + module + '  **********')
    print(x.dtype)
    print(y.dtype)
    print(m.data_shape)
    print(x.shape, y.shape)
    print(y)


if __name__ == '__main__':
    # dataset_test('bci_iv_2a')
    # dataset_test('bci_ii')
    a = DatasetTest()
