"""Sensor Corruption Demo.

This script provides an example of using the Observables functionality to implement a corrupted sensor
(corruption + delay).
Images will be rendered in a delayed fashion, such that the user will have seemingly delayed actions

This is a modified version of the demo_device_control teleoperation script.

这个脚本的核心目的是展示：如果传感器“坏了”或者“被干扰了”，机器人该怎么办？

在现实世界中，传感器永远不会提供 100% 完美的干净数据。摄像头会有噪点，深度相机在遇到黑色物体时会失效，力矩传感器会因为温度产生偏移。

该 Demo 循环展示了三种常见的传感器失真情况：
    A. 图像噪点 (Image Noise/Corruption)
        效果：你会看到摄像头的画面突然变得像老式电视机一样全是雪花，或者颜色突然偏黄。
        物理意义: 模拟低光照环境下的感光度(ISO)噪点，或者传输链路中的信号干扰。
    B. 深度缺失 (Depth Map Artifacts)
        效果：深度图中会出现大片的“空洞”或黑点。
        物理意义: 模拟红外深度相机(如 Kinect 或 RealSense)在遇到高反射表面(镜子)或吸光表面(黑色吸光布)时无法测距的情况。
    C. 观测值偏移 (Observation Bias/Drift)
        效果：机器人明明停在原位，但返回的 robot0_eef_pos(末端位置)却在轻微跳动。
        物理意义: 模拟编码器(Encoder)的累计误差或热漂移。

实验整体功能与验证目的：
    整体功能: 它在机器人仿真环境(robosuite)与控制器之间加入了一个“干扰层”。它不仅让机器人动起来, 还让机器人“看”到的图像带噪点、延迟，“感觉”到的关节位置有偏差、不准。
    具体要做什么：
        手动控制机器人（键盘/手鼠）完成任务（如 Lift 抬升）
        实时对视觉 (Camera) 和本体感受 (Joint Position) 两种传感器施加高斯噪声 (Gaussian Noise) 和时间延迟 (Delay)。
    验证什么：
        验证控制算法的鲁棒性：在数据又延迟又不准的情况下，控制算法是否还能稳定。
        验证人的适应性：演示人类操作员在面对 40ms 以上延迟和画面雪花时，操作难度会如何剧增。

Example:
    $ python demo_sensor_corruption.py --environment Stack --robots Panda --delay 0.05 --corruption 5.0 --toggle-corruption-on-grasp
"""

import argparse
import sys

import cv2
import numpy as np

import robosuite as suite  # 导入机器人仿真核心库
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action  # 导入将设备输入转为机器人动作的工具函数
from robosuite.utils.observables import Observable, create_gaussian_noise_corrupter, create_uniform_sampled_delayer
from robosuite.wrappers import VisualizationWrapper  # ps 导入可视化包装器，用于显示坐标轴或抓取指示

import pprint

pp = pprint.PrettyPrinter(indent=2)

if __name__ == "__main__":

    # step1 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")

    parser.add_argument(
        "--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env"
    )  # 参数: 机器人型号 (list of str)，可以选择多个机器人
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )  # 参数: 多臂环境配置 (str)，单臂时可忽略！在 TwoArmLift 或 TwoArmPegInHole 等环境里，这个参数决定了两个机器人的相对位置（例如 opposed 表示对面站立，side-by-side 表示并排站立）。
    parser.add_argument(
        "--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'"
    )  # 参数: 当前控制的手臂 (str)

    parser.add_argument(
        "--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action"
    )  # 参数: 布尔开关，抓取时是否切换手臂控制。仅多臂有用。按下抓取键（键盘 Space），控制权会在左手和右手之间自动切换。
    parser.add_argument(
        "--toggle-corruption-on-grasp", action="store_true", help="Toggle corruption ON / OFF on gripper action"
    )  # 参数: 布尔开关，抓取时是否开启/关闭传感器干扰。非常有用的调试工具。按下抓取键时，当前的“噪声/延迟”会瞬间开启或关闭。方便你实时对比“地狱模式”和“正常模式”的区别。

    parser.add_argument(
        "--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'"
    )  # 参数: 控制器算法 (str)
    parser.add_argument(
        "--device", type=str, default="keyboard"
    )  # 参数: 控制设备类型 (str)，有键盘或者空间鼠标（6DOF）

    parser.add_argument(
        "--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs"
    )  # 位置控制灵敏度 (float)。缩放你键盘输入的位移量，设为 2.0，按一下 W 机器人会跑得比平时远一倍。
    parser.add_argument(
        "--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs"
    )  # 旋转控制灵敏度 (float)。缩放姿态旋转的速度。
    parser.add_argument(
        "--delay", type=float, default=0.04, help="average delay to use (sec)"
    )  # 传感器平均延迟时间，单位秒 (float)
    parser.add_argument(
        "--corruption", type=float, default=20.0, help="Scale of corruption to use (std dev)"
    )  # 噪声强度标准差 (float)。数值越大，画面越花（雪花点越多），关节跳动越离谱。它直接对应高斯分布的标准差 \sigma。
    parser.add_argument(
        "--camera", type=str, default="agentview", help="Name of camera to render"
    )  # 指定渲染的相机名称 (str)

    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    args = parser.parse_args()

    # step2 控制器配置加载
    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == "ik":
        # 如果控制器是 IK，则使用 [逆运动学] 姿态控制器
        controller_name = "IK_POSE"
    elif args.controller == "osc":
        # 如果控制器是 OSC，则使用 [操作空间] 控制器
        controller_name = "OSC_POSE"
    else:
        # 错误处理，不支持的控制器类
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)  #

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    print("# Using config:")
    pp.pprint(config)
    # {
    #     "controller_configs": {
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
    #     "robots": ["UR5e"],
    # }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        # 如果是双臂机器人，则在配置字典中添加双臂排列方式
        config["env_configuration"] = args.config
    else:
        # 单臂机器人则不额外配置参数
        args.config = None

    # step3 环境创建
    # ! has_renderer=False 表示不使用原生 mujoco 窗口，改用 OpenCV 渲染
    env = suite.make(
        **config,
        has_renderer=False,  # 不开启原生窗口渲染（因为后面要用 OpenCV）
        has_offscreen_renderer=True,  # 开启离屏渲染，用于获取相机图像
        ignore_done=True,  # 忽略任务完成标志，使仿真持续运行
        camera_names=args.camera,  # 指定相机名称
        camera_heights=args.height,  # 设置渲染高度
        camera_widths=args.width,  # 设置渲染宽度
        use_camera_obs=True,  # 开启相机观测，obs 字典将包含图像数据
        use_object_obs=True,  # 开启物体观测，obs 字典将包含物体位置
        hard_reset=False,  # 设为 False 以避免在 Linux/macOS 上频繁重置导致的崩溃
    )

    # NOTE 又是包装器！
    # 将原始环境包装，增加可视化引导标记功能
    env = VisualizationWrapper(env, indicator_configs=None)

    # step4 传感器干扰（Observables）核心配置
    attributes = [
        "corrupter",
        "delayer",
        "sampling_rate",
    ]  # 定义观测系统的三个核心属性标签列表 (list of str)，分别为：扰动、延迟、采样率
    corruption_mode = 1  # int, 信号干扰模式开关，1 表示开启，0 表示关闭
    obs_settings = {}  # dict, 存储不同观测量的修改逻辑

    # 动态修改观测量的属性（噪声、延迟、采样率）
    def modify_obs(obs_name, attrs, mods):
        # 遍历属性名和对应的修改器
        for attr, mod in zip(attrs, mods):
            # 调用 env 接口修改来指定的可观测对象属性
            env.modify_observable(
                observable_name=obs_name,  # 观测量名称 (str)
                attribute=attr,  # 属性名 (str)
                modifier=mod,  # 修改器对象或数值 (callable or float)
            )

    # A. 图像干扰配置
    # Add image corruption and delay
    image_sampling_rate = 10.0  # 设置相机图像的采样频率为 10Hz (float)
    image_obs_name = f"{args.camera}_image"  # 拼接相机观测量的完整键名 (str)，如 "agentview_image"

    # ! image_corrupter: 高斯噪声生成器，负责给 RGB 图像（512x384x3）的每个像素加上随机的正负数值，造成“雪花”效果。
    image_corrupter = create_gaussian_noise_corrupter(mean=0.0, std=args.corruption, low=0, high=255)
    # ! image_delayer: 均匀分布延迟器，可以在 [delay-0.025, delay+0.025] 之间随机采样，负责把当前的画面“扣留”一会再发给用户，模拟网络传输延迟。
    image_delayer = create_uniform_sampled_delayer(min_delay=max(0, args.delay - 0.025), max_delay=args.delay + 0.025)

    # ps1 将以上图像的三个修改器打包成列表 (list)
    image_modifiers = [image_corrupter, image_delayer, image_sampling_rate]

    # 执行图像观测流的修改
    modify_obs(obs_name=image_obs_name, attrs=attributes, mods=image_modifiers)

    # 在设置记录的字典中存入图像修改逻辑
    # Add entry for the corruption / delay settings in dict
    obs_settings[image_obs_name] = {
        "attrs": attributes[:2],  # 仅记录 corrupter 和 delayer
        "mods": lambda: (image_modifiers[:2] if corruption_mode else [None, None]),  # 动态决定是否施加干扰
    }

    # B. 本体感受（关节位置）干扰配置
    # Add proprioception corruption and delay
    proprio_sampling_rate = 20.0  # 设置本体感受（关节数据）采样频率为 20Hz (float)
    # proprio_obs_name: 获取关节位置观测量的名称 (str)，例如 "robot0_joint_pos"
    proprio_obs_name = f"{env.robots[0].robot_model.naming_prefix}joint_pos"  #'robot0_joint_pos'

    # 获取关节物理限位范围 (np.ndarray, shape = (num_joints = 7, 2))，用于计算噪声强度
    joint_limits = env.sim.model.jnt_range[env.robots[0]._ref_joint_indexes]  # shpae = (6, 2)
    print(f"joint_limits: \n", joint_limits)
    # array(
    #     [
    #         [-6.28319, 6.28319],
    #         [-6.28319, 6.28319],
    #         [-3.14159, 3.14159],
    #         [-6.28319, 6.28319],
    #         [-6.28319, 6.28319],
    #         [-6.28319, 6.28319],
    #     ]
    # )

    # 计算每个关节的可动范围。它是“最大限位 - 最小限位”，例如某个关节可以从 -1.5rad 运动到 1.5rad，那么它的 joint_range 就是 3.0rad。
    joint_range = joint_limits[:, 1] - joint_limits[:, 0]  # shape = (6,)
    print(f"joint_range:\n", joint_range)
    # array([12.56638, 12.56638, 6.28318, 12.56638, 12.56638, 12.56638])

    # ! proprio_corrupter：本体干扰器，负责给关节角度加上微小的偏移量，模拟传感器精度不足。
    proprio_corrupter = create_gaussian_noise_corrupter(mean=0.0, std=joint_range / 50.0)

    # 初始化当前关节延迟的实测值 (float)
    curr_proprio_delay = 0.0
    # 创建一个临时的、针对本体感受的延迟采样器
    tmp_delayer = create_uniform_sampled_delayer(
        min_delay=max(0, (args.delay - 0.025) / 2), max_delay=(args.delay + 0.025) / 2
    )

    # 闭包函数：同步更新延迟时间
    def proprio_delayer():
        global curr_proprio_delay  # 声明使用全局变量
        curr_proprio_delay = tmp_delayer()  # 采样一次延迟时间并保存

        # 返回该延迟秒数
        return curr_proprio_delay

    # 计算以仿真步数为单位的延迟
    def calculate_proprio_delay():
        base = env.model_timestep  # 获取仿真步长，通常 0.002s (float)

        # 将连续秒数转为仿真步数对应的物理秒数 (float)
        return base * round(curr_proprio_delay / base) if corruption_mode else 0.0

    # ps2 将以上关节数据的三个修改器打包成列表 (list)
    proprio_modifiers = [proprio_corrupter, proprio_delayer, proprio_sampling_rate]

    # step5 创建“延迟真值”观测器用于对比
    # We will create a separate "ground truth" delayed proprio observable to track exactly how much corruption we're getting in real time
    proprio_sensor = env._observables[proprio_obs_name]._sensor  # 获取底层的传感器引用对象 (Sensor)
    proprio_ground_truth_obs_name = (
        f"{proprio_obs_name}_ground_truth"  # 定义新观测量的名称 (str)，'robot0_joint_pos_ground_truth'
    )

    # 创建一个全新的 Observable：只有 delay，但没有 corrupter
    observable = Observable(
        name=proprio_ground_truth_obs_name,  # 观测量名称
        sensor=proprio_sensor,  # 关联的物理传感器
        delayer=lambda: curr_proprio_delay,  # 使用与主观测同步的延迟量
        sampling_rate=proprio_sampling_rate,  # 采样率
    )

    env.add_observable(observable)  # 将这个“延迟真值”观测量加入环境系统

    # 强制激活关节位置观测
    env.modify_observable(observable_name=proprio_obs_name, attribute="active", modifier=True)

    # 执行关节数据观测流的修改
    modify_obs(obs_name=proprio_obs_name, attrs=attributes, mods=proprio_modifiers)

    # 记录所有设置，方便实时切换开关
    # Add entry for the corruption / delay settings in dict
    obs_settings[proprio_obs_name] = {
        "attrs": attributes[:2],  # 干扰器和延迟器
        "mods": lambda: proprio_modifiers[:2] if corruption_mode else [None, None],  # 动态开关
    }  # 记录关节位置的修改逻辑
    obs_settings[proprio_ground_truth_obs_name] = {
        "attrs": [attributes[1]],  # 仅延迟器
        "mods": lambda: [lambda: curr_proprio_delay] if corruption_mode else [None],  # 动态开关
    }  # 记录“延迟真值”的修改逻辑

    # step6 设备与交互循环
    # Setup printing options for numbers
    np.set_printoptions(precision=3, suppress=True, floatmode="fixed")  # 设置 NumPy 打印格式，保留3位小数

    # 控制设备初始化
    if args.device == "keyboard":  # 如果选择键盘控制
        from robosuite.devices import Keyboard  # 导入键盘设备类

        # 初始化键盘 (Keyboard)
        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":  # 如果选择 SpaceMouse
        from robosuite.devices import SpaceMouse  # 导入 SpaceMouse 类

        # 初始化 (SpaceMouse)
        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    print(f"device: ", device)  # 打印选择的设备
    # device:  <robosuite.devices.keyboard.Keyboard object at 0x7f2d432814c0>

    # step7 Episode 大循环
    while True:
        # 重置环境并获取初始观测字典 (dict)
        obs = env.reset()

        # 每一 episode 重置时，默认开启干扰
        corruption_mode = 1

        # 记录上一时刻的抓取键状态
        last_grasp = 0

        # 启动设备监听线程或初始化缓冲区
        device.start_control()

        # Action 小循环（仿真运行）
        while True:
            # 设置当前激活的机器人实例 (Robot)
            active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]
            # print(f"# Using gripper: ", active_robot.gripper.name)  # Robotiq85Gripper0

            # 将用户输入转化为动作向量 action (np.ndarray, shape=(7,)) 和抓取信号 grasp (float)
            action, grasp = input2action(
                device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config
            )
            # print(f"action: ", action) print(f"grasp: ", grasp)
            # 如果触发了重置键（通常是 ESC）
            if action is None:
                # 跳出当前小循环，重新开始 Episode
                break

            # 如果检测到抓取键按下（从 -1 变为 1 的瞬间）
            # 抓取动作触发干扰开关逻辑
            # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
            # toggle arm control and / or corruption if requested
            if last_grasp < 0 < grasp:
                # 如果开启了切换手臂功能
                if args.switch_on_grasp:
                    # 切换控制左右手
                    args.arm = "left" if args.arm == "right" else "right"
                # 如果开启了抓取切换干扰功能
                if args.toggle_corruption_on_grasp:  # 如果开启了抓取切换干扰功能
                    # 翻转干扰开关状态
                    corruption_mode = 1 - corruption_mode
                    # 遍历所有记录的干扰配置
                    for obs_name, settings in obs_settings.items():
                        # 立即更新干扰状态
                        modify_obs(obs_name=obs_name, attrs=settings["attrs"], mods=settings["mods"]())
            # 更新上一时刻抓取状态
            last_grasp = grasp

            # 计算当前 action 向量与环境要求 action 维度的差值 (int)
            rem_action_dim = env.action_dim - action.size  # rem_action_dim = 0
            # case1 如果 action 太短（例如多臂环境）
            if rem_action_dim > 0:
                # Initialize remaining action space
                rem_action = np.zeros(rem_action_dim)
                # ps This is a multi-arm setting, choose which arm to control and fill the rest with zeros
                # 如果控制右手
                if args.arm == "right":
                    action = np.concatenate(
                        [action, rem_action]
                    )  # 动作拼接到左侧，右侧补零 (np.ndarray, shape=(env.action_dim,))
                # 如果控制左手
                elif args.arm == "left":
                    action = np.concatenate(
                        [rem_action, action]
                    )  # 动作拼接到右侧，左侧补零 (np.ndarray, shape=(env.action_dim,))
                else:
                    # Only right and left arms supported
                    print(
                        "Error: Unsupported arm specified -- "
                        "must be either 'right' or 'left'! Got: {}".format(args.arm)
                    )

            # case2 如果 action 太长（环境没有夹爪动作）
            elif rem_action_dim < 0:
                # We're in an environment with no gripper action space, so trim the action space to be the action dim
                action = action[: env.action_dim]  # 截断 action 向量 (np.ndarray)

            # 执行仿真步进。注意：此时返回的 obs 已根据之前的 modify_observable 进行了“加工” (dict)
            obs, reward, done, info = env.step(action)

            # step7 数据打印与显示
            # observed_value: np.array, shape (num_joints,), 包含噪声
            observed_value = obs[proprio_obs_name]  # 获取受干扰后的关节位置 (np.ndarray, shape=(7,))
            # ground_truth_delayed_value: np.array, shape (num_joints,), 仅延迟，无噪声
            ground_truth_delayed_value = obs[proprio_ground_truth_obs_name]  # 获取仅延迟的真值 (np.ndarray, shape=(7,))
            print(
                f"Observed joint pos: {observed_value}, "
                f"Corruption: {observed_value - ground_truth_delayed_value}, "
                f"Delay: {calculate_proprio_delay():.3f} sec"
            )  # 打印当前物理延迟秒数

            # obs[args.camera + "_image"] 类型为 np.ndarray, shape=(384, 512, 3)
            # [..., ::-1] 将颜色从 RGB 转为 BGR（OpenCV 要求）
            # np.flip(..., 0) 将图像沿水平轴翻转（纠正渲染出的倒影）
            # im: np.array, shape (height, width, 3), dtype=uint8
            im = np.flip(obs[args.camera + "_image"][..., ::-1], 0).astype(np.uint8)
            # print(f"im.shape = ", im.shape)  # im.shape =  (384, 512, 3)

            cv2.imshow("offscreen render", im)
            cv2.waitKey(1)
