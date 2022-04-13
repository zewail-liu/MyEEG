import json
import os
import time
from enum import Enum

import serial


class MarkEnum(Enum):
    """
    messages from Server to Client (use with CTRL_FromServer)
    """
    MARK_START = 100
    MARK_END = 200

    MARK_ONE = 10
    MARK_TWO = 20
    MARK_THREE = 30
    MARK_FOUR = 40
    MARK_FIVE = 50
    MARK_SIX = 60
    MARK_SEVEN = 70
    MARK_EIGHT = 80
    MARK_NINE = 90


class Config:
    @staticmethod
    def load() -> dict:
        conf = None
        path = os.path.realpath(__file__)
        path = path[:path.rindex(os.path.sep)]
        with open(path + "/config.json", 'r', encoding='utf-8') as f:
            conf = json.load(f)
        print(conf)
        return conf


class Marker:
    def __init__(self):
        self.wait_time = 0.001
        self.__ser = None
        self.__eeg_config = Config().load()
        if self.open():
            print("mark port opened success")
        else:
            print("mark port opened failed")

    def markerStart(self):
        try:
            marker = MarkEnum.MARK_START.value
            self.__ser.write([marker])  # 打mark
            self.__ser.flush()
            time.sleep(self.wait_time)
            self.__ser.write([0])
            self.__ser.flush()
        except:
            print('error')
            pass

    def markerEnd(self):
        try:
            marker = MarkEnum.MARK_END.value
            self.__ser.write([marker])  # 打mark
            self.__ser.flush()
            time.sleep(self.wait_time)
            self.__ser.write([0])
            self.__ser.flush()
        except:
            print('error')
            pass

    def mark_label(self, label):
        try:
            self.__ser.write([label])  # 打mark
            self.__ser.flush()
            time.sleep(self.wait_time)
            self.__ser.write([0])
            self.__ser.flush()
        except:
            print('error')
            pass

    def open(self):
        try:
            if self.__ser is None or not self.__ser.isOpen():
                self.__ser = serial.Serial(self.__eeg_config['eeg_mark_port'], self.__eeg_config['eeg_mark_baud'],
                                           timeout=3)
        except:
            return False
        return self.__ser.isOpen()

    def close(self):
        if self.__ser is not None and self.__ser.isOpen():
            self.__ser.close()


def put_marker():
    mark = Marker()
    while True:
        mark.markerStart()
        time.sleep(1)
        mark.markerEnd()
        time.sleep(1)


if __name__ == '__main__':
    # controller = NSController()
    # controller.start()  # add Eeg thread
    import threading

    print('Marker Test:')
    # marker_thread = threading.Thread(target=put_marker)  # add mark thead
    # marker_thread.start()
    m = Marker()
    while True:
        m.mark_label(1)
        time.sleep(1)
        m.mark_label(2)
        time.sleep(1)
