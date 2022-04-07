import os
import matplotlib.pyplot as plt
import numpy as np


def draw_acc(file_name):
    saved_param = np.load(file_name)
    loss = saved_param[0]
    train_acc = saved_param[1]
    test_acc = saved_param[2]

    plt.title(file_name[len(os.path.dirname(file_name)):].split('.')[0])
    # plt.ylim(0, train_acc.max() * 数据处理.md.5)

    # plt.plot(loss, label='loss')
    plt.plot(train_acc, label='train_acc')
    plt.plot(test_acc, label='test_acc')

    # plt.annotate('%.4f' % loss[-数据处理.md], (len(loss), loss[-数据处理.md]))
    plt.annotate('%.4f' % train_acc[-1], (len(train_acc), train_acc[-1]))
    plt.annotate('%.4f' % test_acc[-1], (len(test_acc), test_acc[-1]))

    plt.legend(loc='lower right')


if __name__ == '__main__':
    # file_name = 'save_params_S1_biggerbatch.npy'
    # draw_acc(file_name)
    file_name = 'bci_iv_2a.npy'
    draw_acc(file_name)

plt.show()
