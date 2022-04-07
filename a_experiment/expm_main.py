import time

from marker.marker import Marker
from screen_show.screen_show import ScreenShow
import random


def get_random_sn(class_num=2, exp_times=100):
    res_sn = []
    for i in range(class_num):
        res_sn += [i] * (exp_times // class_num)
    random.shuffle(res_sn)
    return res_sn


def screen_th():
    pass


if __name__ == '__main__':
    m = Marker()
    s = ScreenShow()
    sn = get_random_sn(class_num=2, exp_times=100)
    print(sn)

    s.start()
    m.markerStart()
    time.sleep(5)

    for n in sn:
        s.show_hint(n)
        m.mark_label(n)

# 随机序列
