from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

import pprint

pp = pprint.PrettyPrinter(indent=2)

# 在机器人仿真（Simulation）中，代码负责计算复杂的物理公式（比如：机械臂撞到桌子产生多大的反作用力），但这些数字对人类来说不可见。
# ps 渲染器的作用：将内存中的物理数据（坐标、碰撞模型、光照）转换成我们能看见的图像（像素）。
# 如果没有渲染器，仿真依然在运行，但你只能看到一堆不断变化的坐标数值；有了渲染器，你才能在屏幕上看到真实的 3D 机器人。

# 在 robosuite 中，默认使用的渲染器是基于 MuJoCo 物理引擎自带的图形接口，称为 mjviewer。
#       1、实时交互： 它不仅能让你“看”，还允许你用鼠标拖拽机器人、改变视角、暂停仿真或通过图形界面查看传感器的实时数据。
#       2、低开销： 它是专门为 MuJoCo 设计的，运行速度极快，不会拖慢你的训练速度。
#       3、功能性强： 它可以显示“物理属性”，比如碰撞体（Collisions）的接触点、力的矢量方向等。
# 小知识： robosuite 还支持其他更高级的渲染器（如 NVIDIA 的 Omniverse / Isaac Sim），那些渲染器画出来的图像像电影大片一样真实，但对显卡要求极高；而 mjviewer 就像是一个高效的“工程图”，重点在于准确和快速。

#       1、实时交互：你可以直接用鼠标去拖拽仿真环境里的机器人，或者通过双击某个零件来聚焦。
#       2、低延迟：它是基于桌面端图形卡（OpenGL）直接绘制的，非常流畅。
#       3、调试功能：按 H 键可以显示帮助菜单，你可以切换显示“接触力”、“关节轴”或“质心位置”，这对于调试你的算法非常有帮助。

if __name__ == "__main__":
    """
    主函数：初始化 robosuite 环境并执行随机动作演示

    该函数负责：
    1. 创建环境选项字典
    2. 显示欢迎信息和版本号
    3. 根据用户选择配置环境、机器人和控制器
    4. 初始化环境并执行随机动作循环
    """

    # step1 创建空字典，用于存储传递给环境创建函数的各种选项参数
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    # Welcome to robosuite v1.4.0!

    print(suite.__logo__)
    #   ;     /        ,--.
    #  ["]   ["]  ,<  |__**|
    # /[_]\  [~]\/    |//  |
    #  ] [   OOO      /o|__|

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

        # 如果选择的不是相同的单臂, 就创建一个空列表存储多个单臂
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
        # 选择单臂机器人, 包括 robots = {"Sawyer", "Panda", "Jaco", "Kinova3", "IIWA", "UR5e"}
        options["robots"] = choose_robots(exclude_bimanual=True)  # "UR5e"

    # step3 选择控制器类型
    controller_name = choose_controller()  # OSC_POSE

    # Load the desired controller，加载所选控制器的配置参数
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    print("# Using options:")
    pp.pprint(options)  # 打印总的配置选项
    # {
    #     "controller_configs": {
    #         "control_delta": True,
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

    # step4 初始化环境
    # initialize the task
    env = suite.make(
        **options,  # 解包选项字典作为参数
        has_renderer=True,  # 关键：开启 mjviewer 窗口。你可以看到机器人动，并用鼠标调戏它。
        has_offscreen_renderer=False,  # 不显示窗口，但在内存里偷偷“画图”。这通常用于深度学习（训练神经网络时，AI 只需要像素数据，不需要人去看）。
        ignore_done=True,  # 忽略环境完成标志，即使任务完成也继续运行
        use_camera_obs=False,  # 不使用相机观测数据
        control_freq=20,  # 控制频率设为20Hz，1/20=0.05秒，即：每0.05秒执行一次动作。
        # render_camera="frontview",  # 设置初始视角
    )
    print("env init success: ")
    pp.pprint(env)  # <robosuite.environments.manipulation.lift.Lift object at 0x7ff880557730>

    # 打印环境的所有属性和方法
    # print("# env 所有属性和方法:")
    # for attr in dir(env):
    #     if not attr.startswith("_"):  # 过滤掉私有属性
    #         try:
    #             value = getattr(env, attr)
    #             if callable(value):
    #                 print(f"  {attr}: <method>")
    #             else:
    #                 print(f"  {attr}: {type(value).__name__} = {value}")
    #         except:
    #             print(f"  {attr}: <无法访问>")

    # step5 重置环境
    env.reset()  # 重置环境到初始状态

    if env.has_renderer:
        env.viewer.set_camera(camera_id=0)  # 设置渲染器的摄像头ID为0
        print("camera_id set success!")  # debug

    # Get action limits
    low, high = env.action_spec  # 获取动作空间的上下限值，用于后续归一化随机动作

    # step6 做可视化
    for i in range(30000):
        # 采取随机动作，并进行归一化
        action = np.random.uniform(low, high)  # 生成在动作空间限制范围内的随机动作向量

        # NOTE 这是最重要的！
        obs, reward, done, _ = env.step(action)  # 执行随机动作并获取环境反馈：观测值、奖励、完成标志和其他信息

        # print("# obs.keys: ")
        # pp.pprint(obs.keys())
        # odict_keys(
        #     [
        #         # ? 为什么不直接给角度？ 在深度学习中，角度在 2pi 和 0 之间会有跳变，使用三角函数表示可以保证数值的连续性，更有利于神经网络训练。
        #         "robot0_joint_pos_cos",  # 机器人关节位置（角度）的余弦值（用于角度编码）
        #         "robot0_joint_pos_sin",  # 机器人关节位置（角度）的正弦值（用于角度编码）
        #         "robot0_joint_vel",  # 机器人各关节当前的旋转速度
        #         "robot0_eef_pos",  # 机器人末端执行器（End Effector）中心在三维世界坐标系中的的位置坐标
        #         "robot0_eef_quat",  # 机器人末端执行器（End Effector）中心的姿态四元数表示
        #         "robot0_gripper_qpos",  # 机器人夹爪开合的位置（开合程度）
        #         "robot0_gripper_qvel",  # 机器人夹爪开合的速度（开合速度）
        #         "cube_pos",  # 被抓取的红色方块在世界坐标系中的位置坐标
        #         "cube_quat",  # 红色方块当前的旋转姿态（四元数）
        #         "gripper_to_cube_pos",  # Important！夹爪相对于立方体的相对位置向量。这是一个非常重要的特征，AI 靠这个数据来判断手离物体还有多远。
        #         # 这是一个混合向量。它把上面所有的本体感受数据（位置、速度、末端位置等）拼接成了一个长向量，方便直接输入给神经网络。
        #         "robot0_proprio-state",  # 机器人本体感觉状态（整合了机器人的自身感知信息）
        #         "object-state",  # 物体状态（类似于 robot0_proprio-state，整合了环境中物体的全部观测信息）
        #     ]
        # )

        if env.has_renderer:
            # 更新并显示当前仿真环境的画面
            env.render()
            print("render success!")
            # ps 当调用 env.render() 并且没有做特殊配置时，robosuite 通常会调用 MuJoCo 原生自带的可视化交互窗口，这就是 mjviewer。

        # pp.pprint(obs["robot0_joint_vel"])
        # array([0.42086176, -0.51354486, -0.06998176, 1.01123367, 0.31168851, 1.27760452])
        # pp.pprint(obs["robot0_joint_pos_cos"])
        # array([0.92058423, 0.03065515, -0.80687498, -0.85191482, 0.06005678, -0.4462671])
        # pp.pprint(obs["robot0_joint_pos_sin"])
        # array([-0.39054407, -0.99953002, 0.59072224, -0.52368039, -0.99819496, -0.89489981])

        # 如果想在控制台看到机器人是否接近了物体，可以打印 gripper_to_cube_pos。
        # 当这个向量的数值接近于 0 时，说明夹爪已经碰到了方块。
        # pp.pprint(obs["gripper_to_cube_pos"])

        dist = np.linalg.norm(obs["gripper_to_cube_pos"])  # 计算欧几里得距离
        print(f"距离方块还有: {dist:.4f} 米")

        if dist < 0.02:
            print("准备抓取！")
