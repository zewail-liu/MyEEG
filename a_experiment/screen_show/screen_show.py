import cv2
import numpy as np
import screeninfo
import winsound
import os


class ScreenShow:
    """
        beep-----------hint-----------break-----------end
                1s              4s              5s
        total: 10s per round

        after started, only call show_hint(hint_type) to implement
    """

    def __init__(self):
        self.image_path = os.path.dirname(__file__)
        self.black = np.zeros((1080, 1920, 3), dtype=np.float32)
        self.screen_id = 0

    def start(self):
        """
            when called, press any key to continue the program
        """
        self.show_full_screen(self.black)
        cv2.waitKey()

    def show_hint(self, hint_type: int):
        hints = ['fist_left', 'fist_right']
        fn = self.image_path + fr'/{hints[hint_type]}.jpg'
        self.show_full_screen(self.black)
        winsound.MessageBeep(-1)
        cv2.waitKey(1000)
        self.show_full_screen(cv2.imread(fn))
        cv2.waitKey(4000)
        self.show_full_screen(self.black)
        cv2.waitKey(5000)

    def show_full_screen(self, image):
        screen = screeninfo.get_monitors()[self.screen_id]
        window_name = 'MI-EEG-Experiment'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, image)

    def end(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    t = ScreenShow()
    t.start()
    t.show_hint(1)

    # import keyboard as k
    # while True:
    #     e = k.read_event()
    #     if e.name == 'right':
    #         t.show_hint(1)
    #     elif e.name == 'left':
    #         t.show_hint(0)
    #     elif e.name == 'q':
    #         exit()
