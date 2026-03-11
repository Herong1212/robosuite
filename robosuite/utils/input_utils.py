"""
Utility functions for grabbing user inputs
"""

import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.devices import *
from robosuite.models.robots import *
from robosuite.robots import *


def choose_environment():
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all environments, 获取并排序所有可用环境列表
    envs = sorted(suite.ALL_ENVIRONMENTS)

    # Select environment to run
    print("Here is a list of environments in the suite:\n")

    for k, env in enumerate(envs):
        print("[{}] {}".format(k, env))
    # 打印环境选项供用户查看
    print()
    #     [0] Door
    #     [1] Lift
    #     [2] NutAssembly
    #     [3] NutAssemblyRound
    #     [4] NutAssemblySingle
    #     [5] NutAssemblySquare
    #     [6] PickPlace
    #     [7] PickPlaceBread
    #     [8] PickPlaceCan
    #     [9] PickPlaceCereal
    #     [10] PickPlaceMilk
    #     [11] PickPlaceSingle
    #     [12] Stack
    #     [13] TwoArmHandover
    #     [14] TwoArmLift
    #     [15] TwoArmPegInHole
    #     [16] Wipe

    try:
        # 提示用户输入数字选择环境
        s = input("Choose an environment to run " + "(enter a number from 0 to {}): ".format(len(envs) - 1))
        # Choose an environment to run (enter a number from 0 to 16): 1

        # parse input into a number within range
        #   int(s) - 将用户输入的字符串转换为整数
        #   max(int(s), 0) - 确保索引不小于0（防止负数输入）
        #   min(..., len(envs)) - 确保索引不超过环境总数（防止超出范围）
        k = min(max(int(s), 0), len(envs))
    except:
        k = 0
        # 验证输入有效性，若无效则默认选择第一个环境
        print("Input is not valid. Use {} by default.\n".format(envs[k]))

    # Return the chosen environment name, 返回用户选择的环境名称字符串
    return envs[k]


def choose_controller():
    """
    Prints out controller options, and returns the requested controller name

    Returns:
        str: Chosen controller name
    """
    # get the list of all controllers
    controllers_info = suite.controllers.CONTROLLER_INFO
    controllers = list(suite.ALL_CONTROLLERS)

    # Select controller to use
    print("Here is a list of controllers in the suite:\n")

    for k, controller in enumerate(controllers):
        print("[{}] {} - {}".format(k, controller, controllers_info[controller]))
    # 打印控制器选项供用户查看
    print()
    #     [0] JOINT_VELOCITY - Joint Velocity
    #     [1] JOINT_TORQUE - Joint Torque
    #     [2] JOINT_POSITION - Joint Position
    #     [3] OSC_POSITION - Operational Space Control (Position Only)
    #     [4] OSC_POSE - Operational Space Control (Position + Orientation)
    #     [5] IK_POSE - Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)

    try:
        # 提示用户输入数字选择控制器
        s = input("Choose a controller for the robot " + "(enter a number from 0 to {}): ".format(len(controllers) - 1))

        # parse input into a number within range
        k = min(max(int(s), 0), len(controllers) - 1)
    except:
        k = 0
        # 验证输入有效性，若无效则默认选择第一个控制器
        print("Input is not valid. Use {} by default.".format(controllers)[k])

    # Return chosen controller
    return controllers[k]


def choose_multi_arm_config():
    """
    Prints out multi-arm environment configuration options, and returns the requested config name

    Returns:
        str: Requested multi-arm configuration name
    """
    # Get the list of all multi arm configs
    env_configs = {
        "Single Arms Opposed": "single-arm-opposed",
        "Single Arms Parallel": "single-arm-parallel",
        "Bimanual": "bimanual",
    }

    # Select environment configuration
    print("A multi-arm environment was chosen. Here is a list of multi-arm environment configurations:\n")

    for k, env_config in enumerate(list(env_configs)):
        print("[{}] {}".format(k, env_config))
    print()
    try:
        s = input(
            "Choose a configuration for this environment "
            + "(enter a number from 0 to {}): ".format(len(env_configs) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(env_configs))
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(list(env_configs)[k]))

    # Return requested configuration
    return list(env_configs.values())[k]


def choose_robots(exclude_bimanual=False):
    """
    Prints out robot options, and returns the requested robot. Restricts options to single-armed robots if
    @exclude_bimanual is set to True (False by default)
    打印机器人选项，并返回请求的机器人。如果 @exclude_bimanual 设置为 True，则限制选项为单臂机器人（默认为 False）

    Args:
        exclude_bimanual (bool): 如果设置，则从机器人选项中排除双臂机器人

    Returns:
        str: 请求的机器人名称
    """
    # 获取机器人列表
    robots = {
        "Sawyer",
        "Panda",
        "Jaco",
        "Kinova3",
        "IIWA",
        "UR5e",
    }

    # 如果不禁用双臂机器人，则添加Baxter
    if not exclude_bimanual:
        robots.add("Baxter")

    # 确保集合按确定顺序排序
    robots = sorted(robots)

    # 选择机器人
    print("Here is a list of available robots:\n")

    for k, robot in enumerate(robots):
        print("[{}] {}".format(k, robot))
    # 打印机器人选项供用户查看
    print()
    #     [0] IIWA
    #     [1] Jaco
    #     [2] Kinova3
    #     [3] Panda
    #     [4] Sawyer
    #     [5] UR5e

    try:
        s = input("Choose a robot " + "(enter a number from 0 to {}): ".format(len(robots) - 1))
        # parse input into a number within range
        k = min(max(int(s), 0), len(robots))
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(list(robots)[k]))

    # Return requested robot
    return list(robots)[k]


def input2action(device, robot, active_arm="right", env_configuration=None):
    """
    Converts an input from an active device into a valid action sequence that can be fed into an env.step() call

    If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action

    Args:
        device (Device): A device from which user inputs can be converted into actions. Can be either a Spacemouse or
            Keyboard device class. -- 可以从中将用户输入转换为动作的设备。可以是 Spacemouse 或 Keyboard 设备类

        robot (Robot): Which robot we're controlling. --  我们正在控制哪个机器人

        active_arm (str): Only applicable for multi-armed setups (e.g.: multi-arm environments or bimanual robots).
            Allows inputs to be converted correctly if the control type (e.g.: IK) is dependent on arm choice.
            Choices are {right, left}
            仅适用于[多臂]设置(例如：多臂环境或双手机器人)。如果控制类型 (例如: IK) 依赖于臂的选择，则允许正确转换输入。选项为{right, left}

        env_configuration (str or None): Only applicable for multi-armed environments. Allows inputs to be converted
            correctly if the control type (e.g.: IK) is dependent on the environment setup. Options are:
            {bimanual, single-arm-parallel, single-arm-opposed}
            仅适用于多臂环境。如果控制类型 (例如: IK) 依赖于环境设置，则允许正确转换输入。选项为：{bimanual, single-arm-parallel, single-arm-opposed}

    Returns:
        2-tuple:

            - (None or np.array): Action interpreted from @device including any gripper action(s). None if we get a
                reset signal from the device. -- 从 @device 解释的动作，包括任何夹爪动作。如果从设备获得重置信号则为 None。
            - (None or int): 1 if desired close, -1 if desired open gripper state. None if get a reset signal from the
                device. -- 1 表示期望关闭，-1 表示期望打开夹爪状态。如果从设备获得重置信号则为 None。

    """
    # 从设备获取控制器状态字典
    state = device.get_controller_state()

    # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
    #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
    #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)

    # Note: 设备输出旋转时 x 和 z 翻转，以考虑机器人起始时夹爪朝下的情况
    #       还要注意，输出的旋转是绝对旋转，而输出的dpos是增量位置
    #       中性用户输入的原始增量旋转保存在raw_drotation中(滚动、俯仰、偏航)

    # 从状态字典中提取各项值
    # dpos: 位移向量，shape (3,)，类型 np.array
    # rotation: 绝对旋转，类型 np.array
    # raw_drotation: 原始增量旋转，shape (3,)，类型 np.array
    # grasp: 夹爪状态，布尔值，类型 bool
    # reset: 重置信号，布尔值，类型 bool
    dpos, rotation, raw_drotation, grasp, reset = (
        state["dpos"],  # 位移变化，np.array (3,)
        state["rotation"],  # 旋转，np.array
        state["raw_drotation"],  # 原始增量旋转，np.array (3,)
        state["grasp"],  # 夹爪操作，bool
        state["reset"],  # 重置信号，bool
    )

    # 如果需要重置，立即返回 None
    if reset:
        return None, None

    # 获取控制器引用
    # 如果不是双臂机器人，则使用 robot.controller；如果是双臂机器人，则使用 robot.controller[active_arm]
    controller = robot.controller if not isinstance(robot, Bimanual) else robot.controller[active_arm]
    # 获取夹爪自由度
    # 如果不是双臂机器人，则使用 robot.gripper.dof；如果是双臂机器人，则使用 robot.gripper[active_arm].dof
    gripper_dof = robot.gripper.dof if not isinstance(robot, Bimanual) else robot.gripper[active_arm].dof

    # 首先处理原始 drotation
    # 重新排列轴的顺序，从 [0,1,2] 变为 [1,0,2]，即交换 x 和 y 轴
    drotation = raw_drotation[[1, 0, 2]]

    # case1 如果控制器是 IK_POSE 类型
    if controller.name == "IK_POSE":
        # If this is panda, want to swap x and y axis
        if isinstance(robot.robot_model, Panda):
            drotation = drotation[[1, 0, 2]]
        else:
            # Flip x
            drotation[0] = -drotation[0]
        # Scale rotation for teleoperation (tuned for IK)
        drotation *= 10
        dpos *= 5
        # relative rotation of desired from current eef orientation
        # map to quat
        drotation = T.mat2quat(T.euler2mat(drotation))

        # If we're using a non-forward facing configuration, need to adjust relative position / orientation
        if env_configuration == "single-arm-opposed":
            # Swap x and y for pos and flip x,y signs for ori
            dpos = dpos[[1, 0, 2]]
            drotation[0] = -drotation[0]
            drotation[1] = -drotation[1]
            if active_arm == "left":
                # x pos needs to be flipped
                dpos[0] = -dpos[0]
            else:
                # y pos needs to be flipped
                dpos[1] = -dpos[1]

        # Lastly, map to axis angle form
        drotation = T.quat2axisangle(drotation)

    # case2 如果控制器是 OSC_POSE 类型
    elif controller.name == "OSC_POSE":
        # Flip z
        drotation[2] = -drotation[2]
        # Scale rotation for teleoperation (tuned for OSC) -- gains tuned for each device
        drotation = drotation * 1.5 if isinstance(device, Keyboard) else drotation * 50
        dpos = dpos * 75 if isinstance(device, Keyboard) else dpos * 125

    # case3 如果控制器是 OSC_POSITION 类型
    elif controller.name == "OSC_POSITION":
        dpos = dpos * 75 if isinstance(device, Keyboard) else dpos * 125

    # case4 其他控制器类型不支持
    else:
        # No other controllers currently supported
        print("Error: Unsupported controller specified -- Robot must have either an IK or OSC-based controller!")

    # 将 0 映射到 -1(打开) 并将 1 映射到 1(关闭)
    # map 0 to -1 (open) and map 1 to 1 (closed)
    grasp = 1 if grasp else -1

    # 基于单个机器人的动作空间，创建动作
    if controller.name == "OSC_POSITION":
        # 对于只有位置控制的 OSC，动作只包含位置和夹爪
        # np.concatenate([dpos, [grasp] * gripper_dof])，其中dpos是(3,)，[grasp]*gripper_dof是(gripper_dof,)
        # 所以 action 形状为(3+gripper_dof,)
        action = np.concatenate([dpos, [grasp] * gripper_dof])
    else:
        # 对于其他控制器，动作包含位置、旋转和夹爪
        # np.concatenate([dpos, drotation, [grasp] * gripper_dof])，其中dpos是(3,)，drotation是(3,)，[grasp]*gripper_dof是(gripper_dof,)
        # 所以 action 形状为(3+3+gripper_dof,) = (6+gripper_dof,)
        action = np.concatenate([dpos, drotation, [grasp] * gripper_dof])

    # Return the action and grasp
    return action, grasp  # shape = (np.array, int)
