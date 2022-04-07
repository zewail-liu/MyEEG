# 使用xbox手柄控制tello

import djitellopy
import pygame

tello = djitellopy.Tello()
tello.connect()

pygame.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()
done = False

while not done:
    res = [0, 0, 0, 0]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        if event.type == pygame.JOYBUTTONDOWN:
            if joystick.get_button(0):  # A键起飞
                tello.takeoff()
            if joystick.get_button(1):  # B键降落
                tello.land()
                done = True
                break

        if event.type == pygame.JOYAXISMOTION:
            # axes = joystick.get_numaxes()
            for i in range(4):  # 左右摇杆
                axis = joystick.get_axis(i)
                if abs(axis) > 0.5:
                    res[i] = axis * 50  # 方向及速度
            break
    tello.send_rc_control(res[0], -res[1], -res[3], res[2])
    pygame.time.wait(100)

pygame.quit()
