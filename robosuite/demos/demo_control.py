"""
此脚本展示了 robosuite 中各类控制器的各项功能。
! 本 demo 目的：确认机械臂的 X, Y, Z 轴方向和期望的是否一致，以及旋转方向是否符合右手定则。
测试逻辑：
针对指定的控制器，脚本会依次遍历其动作空间的每一个维度。
在每个维度上，它会从 "中性值" 开始执行一个微小的 "扰动测试值(test_value = 0.1)"，该动作会持续 steps_per_action = 75 步；
随后，在进入下一个维度的测试前，所有维度会恢复到中性值并保持 steps_per_rest = 75 步。

    示例：
    假设 OSC_POSE 控制器（不含夹爪）的预期动作空间为 (dx, dy, dz, droll, dpitch, dyaw)，其随时间变化的测试序列如下：
        
        ***START OF DEMO***
        ( dx,  0,  0,  0,  0,  0, grip)     <-- Translation in x-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0, dy,  0,  0,  0,  0, grip)     <-- Translation in y-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0, dz,  0,  0,  0, grip)     <-- Translation in z-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0, dr,  0,  0, grip)     <-- Rotation in roll (x) axis       for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0,  0, dp,  0, grip)     <-- Rotation in pitch (y) axis      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0,  0,  0, dy, grip)     <-- Rotation in yaw (z) axis        for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        ***END OF DEMO***

    因此, OSC_POSE 控制器的预期表现应该是：先依次沿 X、Y、Z 方向进行直线运动，随后依次绕其 X、Y、Z 轴进行旋转。

各控制器预期表现概览
请参考模块部分中控制器的文档，了解每个控制器的概述。每个控制器都应在其控制空间内以受控的方式运行。以下是测试期间各控制器的定性行为描述：

* OSC_POSE: 夹爪相对于全局坐标系，依次沿 x, y, z 方向直线移动，随后依次绕 x, y, z 轴旋转。
* OSC_POSITION: 夹爪相对于全局坐标系，依次沿 x, y, z 方向直线移动。
* IK_POSE: 夹爪相对于机器人末端执行器局部坐标系，依次沿 x, y, z 方向直线移动，随后依次绕 x, y, z 轴旋转。
* JOINT_POSITION (关节位置): 机器人各关节依次以受控方式转动。
* JOINT_VELOCITY (关节速度)：机器人各关节依次以受控方式转动。
* JOINT_TORQUE (关节力矩)：与其他控制器不同，力矩控制器表现得相当“迟钝/无力”。因为它本质上只是对 MuJoCo 执行器直接力矩控制的简单封装。因此，当机器人具有非零速度时，即便给入“中性”的 0 力矩 也不保证机器人能保持稳定状态。

"""

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.robots import Bimanual
from robosuite.utils.input_utils import *

import pprint

pp = pprint.PrettyPrinter(indent=2)

if __name__ == "__main__":

    # step1 创建空字典，用于存储传递给环境创建函数的各种选项参数
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # step2 选择仿真环境, 并将其添加到 options 字典中
    options["env_name"] = choose_environment()  # "Lift" (键盘输入为 1)

    # case1 If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            # 如果选择的是相同的单臂, 则直接设置机器人为 Baxter（因为双手操作需要特定机器人）
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            # 循环两次，选择两个不同的单臂
            for i in range(2):
                print("Please choose Robot {}...\n".format(i))  # 提示用户选择了多单臂配置
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # case2 Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        # 否则，选择单臂机器人, 可选包括 robots = {"Sawyer", "Panda", "Jaco", "Kinova3", "IIWA", "UR5e"}
        options["robots"] = choose_robots(exclude_bimanual=True)  # "UR5e"（键盘输入为 5）

    # 获取关节维度的临时方法
    joint_dim = 6 if options["robots"] == "UR5e" else 7

    # step3 选择控制器类型
    controller_name = choose_controller()  # ps [4] OSC_POSE（键盘输入为 4）

    # Load the desired controller，加载所选控制器的配置参数
    options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

    print("# Using options:")
    pp.pprint(options)  # 打印总的配置选项
    # {
    #     "controller_configs":
    #     {
    #         "control_delta": False,
    #         "damping_ratio": 1,
    #         "damping_ratio_limits": [0, 10],
    #         "impedance_mode": "fixed",
    #         "input_max": 1,
    #         "input_min": -1,
    #         "interpolation": None,
    #         "kp": 150,
    #         "kp_limits": [0, 300],
    #         "orientation_limits": None,
    #         "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    #         "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    #         "position_limits": None,
    #         "ramp_ratio": 0.2,
    #         "type": "OSC_POSE",
    #         "uncouple_pos_ori": True,
    #     },
    #     "env_name": "Lift",
    #     "robots": "UR5e",
    # }

    # step4 定义要使用的预定义控制器动作（action_dim, num_test_steps, test_value）
    controller_settings = {
        "OSC_POSE": [6, 6, 0.1],
        "OSC_POSITION": [3, 3, 0.1],
        "IK_POSE": [6, 6, 0.01],
        "JOINT_POSITION": [joint_dim, joint_dim, 0.2],
        "JOINT_VELOCITY": [joint_dim, joint_dim, -0.1],
        "JOINT_TORQUE": [joint_dim, joint_dim, 0.25],
    }

    print(f"# controller_settings.keys():\n", controller_settings.keys())
    # dict_keys(["OSC_POSE", "OSC_POSITION", "IK_POSE", "JOINT_POSITION", "JOINT_VELOCITY", "JOINT_TORQUE"])

    # step5 为每个控制器测试定义变量
    # ps1 动作空间的维度。对于 OSC_POSE，是 6（3个平移 + 3个旋转）。
    action_dim = controller_settings[controller_name][0]
    # ps2 需要测试的总回合数。因为 OSC_POSE 有 6 个自由度，所以我们要分 6 个回合测试，每个测试回合针对一个特定的自由度进行验证。
    num_test_steps = controller_settings[controller_name][1]
    # ps3 测试时给定的动作强度。因为 control_delta=True，代表每一帧向目标方向移动归一化后的值（即 test_value x output_max），而不是真实距离 0.1 米！！！
    test_value = controller_settings[controller_name][2]

    # step6 定义每个控制器动作使用的步数以及动作之间的步数
    steps_per_action = 75  # 动作持续时间。表示每个动作执行的持续时间，大约持续3.75秒，用于让机器人向某个方向连续运动。
    steps_per_rest = 75  # 动作后的冷却时间。表示在每个动作执行完成后，系统会进入休息状态75步，这样可以观察机器人在停止运动后是否能够保持稳定，不会出现晃动现象。

    # step6 初始化环境
    # initialize the task
    env = suite.make(
        **options,  # 解包选项字典作为参数
        has_renderer=True,  # 关键：开启 mjviewer 窗口。你可以看到机器人动，并用鼠标调戏它。
        has_offscreen_renderer=False,  # 不显示窗口，但在内存里偷偷“画图”。这通常用于深度学习（训练神经网络时，AI 只需要像素数据，不需要人去看）。
        ignore_done=True,  # 忽略环境完成标志，即使任务完成也继续运行
        use_camera_obs=False,  # 不使用相机观测数据
        # 在仿真中，horizon 是一个 Episode（回合）的硬上限。每个回合是指运动步数 + 冷却步数，一共 6 个回合(一个自由度对应一个回合)当, env.step() 运行的次数达到这个值【（150+150）*6=900】时，done 信号会变成 True，就完成了一个 episode。
        horizon=(steps_per_action + steps_per_rest) * num_test_steps,  # 设置环境的总步数。
        control_freq=20,  # 控制频率设为20Hz，1/20=0.05秒，即：每0.05秒执行一次动作。
    )
    print("env init success: ")
    pp.pprint(env)  # <robosuite.environments.manipulation.lift.Lift object at 0x7ff880557730>

    # 打印环境的所有属性和方法
    print("# env 所有属性和方法:")
    for attr in dir(env):
        if not attr.startswith("_"):  # 过滤掉私有属性
            try:
                value = getattr(env, attr)
                if callable(value):
                    print(f"  {attr}: <method>")
                else:
                    print(f"  {attr}: {type(value).__name__} = {value}")
            except:
                print(f"  {attr}: <无法访问>")

    # step7 重置环境
    # ps 在 robosuite 的设计中，一个 Episode（回合） 指的是从 env.reset() 开始，到满足结束条件（或达到步数上限）为止的完整过程。
    env.reset()

    if env.has_renderer:
        env.viewer.set_camera(camera_id=0)  # 设置渲染器的摄像头ID为0
        print("camera_id set success!")  # debug

    # step8 为了适应多臂设置（例如：Baxter），我们需要确保填充任何额外的动作空间
    n = 0
    gripper_dim = 0
    for robot in env.robots:
        # 1. 获取夹爪自由度
        # 如果是双臂机器人(Bimanual)，取其右手的自由度；否则直接取夹爪自由度。
        # 例子：UR5e 配合单指夹爪，gripper_dim = 1
        gripper_dim = robot.gripper["right"].dof if isinstance(robot, Bimanual) else robot.gripper.dof

        # 2. 计算控制组数 n
        # robot.action_dim 是机器人接受的总输入向量长度（包含控制器维度 + 夹爪维度）。
        # 例子：UR5e 在 OSC_POSE 模式下，它的总 action_dim 是 7 (6位控制 + 1位夹爪)。
        # 计算：n += int(7 / (6 + 1))  => n = 1
        n += int(robot.action_dim / (action_dim + gripper_dim))

    # step9 定义中性值（零动作向量）
    # 在 control_delta=True 的情况下，输入 0 代表“保持当前位置，不要位移”。
    neutral = np.zeros(action_dim + gripper_dim)

    # step10 跟踪 done 变量以知道何时 break loop
    count = 0

    # 遍历控制器空间
    while count < num_test_steps:  # 循环 6 次（针对 OSC_POSE）
        action = neutral.copy()  # 每一轮开始，先将 action 重置为全是 0

        # 第一阶段：执行扰动（Action Loop）
        for i in range(steps_per_action):  # 持续 75 步

            # case1 如果 count 是 3, 4, 5，代表测试旋转 (Rotation)
            if controller_name in {"IK_POSE", "OSC_POSE"} and count > 2:
                vec = np.zeros(3)  # 设置此值为缩放的轴角向量
                vec[count - 3] = test_value  # 针对旋转维度赋值
                action[3:6] = vec  # 放入 action 的后三位

            # case2 如果 count 是 0, 1, 2，代表测试平移 (Translation)
            else:
                action[count] = test_value  # 针对平移维度赋值

            # ps 执行动作！将动作重复 n 次（对于多臂情况，代码会通过 np.tile 让两只手同时做一模一样的动作（比如两只手同时向左移，同时向上）
            total_action = np.tile(action, n)
            env.step(total_action)

            if env.has_renderer:
                env.render()  # 更新并显示当前仿真环境的画面
                # print("render success!")

        # 第二阶段：保持静止（Rest Loop）
        for i in range(steps_per_rest):  # 持续 75 步
            total_action = np.tile(neutral, n)  # 使用中性动作（暂停）
            env.step(total_action)  # 执行中性动作，告诉机器人原地待命

            if env.has_renderer:
                env.render()  # 更新并显示当前仿真环境的画面
                # print("render success!")

        count += 1

    # 在开始下一个测试之前关闭此环境
    env.close()
