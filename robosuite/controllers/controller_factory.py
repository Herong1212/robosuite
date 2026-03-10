"""
Set of functions that streamline controller initialization process
"""

import json
import os
from copy import deepcopy

import numpy as np

from .interpolators.linear_interpolator import LinearInterpolator
from .joint_pos import JointPositionController
from .joint_tor import JointTorqueController
from .joint_vel import JointVelocityController
from .osc import OperationalSpaceController

# Global var for linking pybullet server to multiple ik controller instances if necessary
pybullet_server = None


def reset_controllers():
    """
    Global function for doing one-time clears and restarting of any global controller-related
    specifics before re-initializing each individual controller again
    """
    global pybullet_server
    # Disconnect and reconnect to pybullet server if it exists
    if pybullet_server is not None:
        pybullet_server.disconnect()
        pybullet_server.connect()


def get_pybullet_server():
    """
    Getter to return reference to pybullet server module variable

    Returns:
        PyBulletServer: Server instance running PyBullet
    """
    global pybullet_server
    return pybullet_server


# NOTE 加载控制器配置参数
def load_controller_config(custom_fpath=None, default_controller=None):
    """
    Utility function that loads the desired controller and returns the loaded configuration as a dict

    If @default_controller is specified, any value inputted to @custom_fpath is overridden and the default controller
    configuration is automatically loaded. See specific arg description below for available default controllers.

    Args:
        custom_fpath (str): 自定义控制器配置.json文件的绝对文件路径
        default_controller (str): 如果指定，则覆盖 @custom_fpath 并加载指定控制器的默认配置文件。
            选择包括: {"JOINT_POSITION", "JOINT_TORQUE", "JOINT_VELOCITY", "OSC_POSITION", "OSC_POSE", "IK_POSE"}

    Returns:
        dict: 控制器配置 字典

    Raises:
        AssertionError: [Unknown default controller name]
        AssertionError: [No controller specified]
    """
    # First check if default controller is not None; if it is not, load the appropriate controller
    if default_controller is not None:

        # 验证请求的默认控制器是否存在于可用控制器中
        from robosuite.controllers import ALL_CONTROLLERS

        assert (
            default_controller in ALL_CONTROLLERS
        ), "Error: Unknown default controller specified. Requested {}, " "available controllers: {}".format(
            default_controller, list(ALL_CONTROLLERS)
        )

        # 构造默认控制器配置文件的路径
        custom_fpath = os.path.join(
            os.path.dirname(__file__), "..", "controllers/config/{}.json".format(default_controller.lower())
        )
        print(f"Loading default controller config: {custom_fpath}")
        # /home/robot/robosuite/robosuite/controllers/../controllers/config/osc_pose.json

    # 确保提供了自定义文件路径或默认控制器
    assert custom_fpath is not None, "Error: Either custom_fpath or default_controller must be specified!"

    # 从 JSON 文件加载控制器配置
    try:
        with open(custom_fpath) as f:
            controller_config = json.load(f)
    except FileNotFoundError:
        print("Error opening controller filepath at: {}. " "Please check filepath and try again.".format(custom_fpath))

    # Return the loaded controller
    return controller_config


def controller_factory(name, params):
    """
    Generator for controllers

    Creates a Controller instance with the provided @name and relevant @params.

    Args:
        name (str): the name of the controller. Must be one of: {JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY,
            OSC_POSITION, OSC_POSE, IK_POSE}
        params (dict): dict containing the relevant params to pass to the controller
        sim (MjSim): Mujoco sim reference to pass to the controller

    Returns:
        Controller: Controller instance

    Raises:
        ValueError: [unknown controller]
    """

    interpolator = None
    if params["interpolation"] == "linear":
        interpolator = LinearInterpolator(
            ndim=params["ndim"],
            controller_freq=(1 / params["sim"].model.opt.timestep),
            policy_freq=params["policy_freq"],
            ramp_ratio=params["ramp_ratio"],
        )

    if name == "OSC_POSE":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.set_states(dim=3)  # EE control uses dim 3 for pos and ori each
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.set_states(ori="euler")
        params["control_ori"] = True
        return OperationalSpaceController(interpolator_pos=interpolator, interpolator_ori=ori_interpolator, **params)

    if name == "OSC_POSITION":
        if interpolator is not None:
            interpolator.set_states(dim=3)  # EE control uses dim 3 for pos
        params["control_ori"] = False
        return OperationalSpaceController(interpolator_pos=interpolator, **params)

    if name == "IK_POSE":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.set_states(dim=3)  # EE IK control uses dim 3 for pos and dim 4 for ori
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.set_states(dim=4, ori="quat")

        # Import pybullet server if necessary
        global pybullet_server
        from .ik import InverseKinematicsController

        if pybullet_server is None:
            from robosuite.controllers.ik import PyBulletServer

            pybullet_server = PyBulletServer()
        return InverseKinematicsController(
            interpolator_pos=interpolator,
            interpolator_ori=ori_interpolator,
            bullet_server_id=pybullet_server.server_id,
            **params,
        )

    if name == "JOINT_VELOCITY":
        return JointVelocityController(interpolator=interpolator, **params)

    if name == "JOINT_POSITION":
        return JointPositionController(interpolator=interpolator, **params)

    if name == "JOINT_TORQUE":
        return JointTorqueController(interpolator=interpolator, **params)

    raise ValueError("Unknown controller name: {}".format(name))
