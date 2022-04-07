import pygame

# 定义一些寒色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# 这是一个简单的类，将帮助我们打印到屏幕上。它与操纵杆无关，只是输出信息。
class TextPrint:
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def print(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, [self.x, self.y])
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10


pygame.init()

# 设置屏幕得到宽度和长度 [width,height]
size = [500, 700]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("My Game")

# 保持循环直到用户点击关闭按钮
done = False

# 被用来管理屏幕更新的速度
clock = pygame.time.Clock()

# 初始化joystick
pygame.joystick.init()

# 准备好打印
textPrint = TextPrint()

# -------- 程序主循环 -----------
while done == False:
    # 事件处理的步骤
    for event in pygame.event.get():  # 用户要做的事情（键盘事件...）
        if event.type == pygame.QUIT:  # 如果用户触发了关闭事件
            done = True  # 设置我们做了这件事的标志，所以我们就可以退出循环了

        #	可能的joystick行为: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
        if event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")

    # 绘制的步骤
    # 首先，用白色清除屏幕。不要放其它的绘图指令
    # 在这条上面的指令，将会被擦除
    screen.fill(WHITE)
    textPrint.reset()

    # 得到joystick的数量
    joystick_count = pygame.joystick.get_count()

    textPrint.print(screen, "Number of joysticks: {}".format(joystick_count))
    textPrint.indent()

    # 在每个joystick中：
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()

        textPrint.print(screen, "Joystick {}".format(i))
        textPrint.indent()

        # 从操作系统中获取控制器/joystick的名称
        name = joystick.get_name()
        textPrint.print(screen, "Joystick name: {}".format(name))

        # 通常轴成对运行，一个轴向上/向下，另一个轴向左/右。
        axes = joystick.get_numaxes()
        textPrint.print(screen, "Number of axes: {}".format(axes))
        textPrint.indent()

        for i in range(axes):
            axis = joystick.get_axis(i)
            textPrint.print(screen, "Axis {} value: {:>6.3f}".format(i, axis))
        textPrint.unindent()

        buttons = joystick.get_numbuttons()
        textPrint.print(screen, "Number of buttons: {}".format(buttons))
        textPrint.indent()

        for i in range(buttons):
            button = joystick.get_button(i)
            textPrint.print(screen, "Button {:>2} value: {}".format(i, button))
        textPrint.unindent()

        # 帽子开关。完全或完全没有方向，不像操纵杆。
        # 值在数组中返回
        hats = joystick.get_numhats()
        textPrint.print(screen, "Number of hats: {}".format(hats))
        textPrint.indent()

        for i in range(hats):
            hat = joystick.get_hat(i)
            textPrint.print(screen, "Hat {} value: {}".format(i, str(hat)))
        textPrint.unindent()

        textPrint.unindent()

    # 所有绘图的指令必须在这一条前面

    # 向前运行，并更新屏幕
    pygame.display.flip()

    # 限制每秒20帧
    clock.tick(20)

# 关闭窗口并退出.
# 如果你忘记这行，程序会被挂起，如果它从IDLE中运行的话
# （通常在IDLE中运行，需要两条退出语句）
# pygame.quit()
# sys.exit()
pygame.quit()
