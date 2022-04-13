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


if __name__ == '__main__':
    m = Marker()
    s = ScreenShow()
    sn = get_random_sn(class_num=4, exp_times=60)

    s.start()
    m.markerStart()
    time.sleep(5)

    # m.mark_label(110)
    # s.baseline()
    # m.mark_label(111)
    # time.sleep(5)

    for n in sn:
        m.mark_label(n + 40)
        s.show_hint(n)
