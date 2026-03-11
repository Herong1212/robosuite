"""
Script to showcase domain randomization functionality.

Fixed domain randomization compatibility for mujoco!=3.1.1. ------ From v1.5.2

At this moment, the randomization functionality focuses on visual variations, including colors, textures, and camera viewpoints.
"""

import robosuite.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import DomainRandomizationWrapper

import pprint

pp = pprint.PrettyPrinter(indent=2)

# We'll use instance randomization so that entire geom groups are randomized together
macros.USING_INSTANCE_RANDOMIZATION = True  # 实例随机化，如果设为 False，它可能会把机械臂的每一个小螺丝钉都染成不同颜色；设为 True，它会把整个机械臂作为一个整体（Instance）来染同一种颜色。这看起来会更符合真实物理逻辑。

if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = choose_environment()  # "Lift"

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)  # "UR5e"

    # Choose controller
    controller_name = choose_controller()  # "OSE_POSE"

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    print("# Using Options:")
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

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        hard_reset=False,  # TODO: Not setting this flag to False brings up a segfault on macos or glfw error on linux
    )

    # NOTE 包装器！输入和输出在接口上是一致的。这是 robosuite 的灵魂设计之一。
    # 输入（Input）：一个原始的 robosuite 环境对象（例如你用 suite.make("Lift", ...) 创建出来的 env）。
    # 输出（Output）：一个被“增强”了的环境对象。
    # 当执行代码时，会遵循"就近原则"：
    #   case1：如果在包装器中定义了（如 step 和 reset）：
    #       处理方式：包装器会先执行自己的逻辑（比如：改颜色、改摩擦力），然后通常会通过 super().step() 或 self.env.step() 把请求传给内部的原始环境。
    #   case2：如果包装器没定义（如 action_spec 或 reward_range）：
    #       处理方式：包装器会通过 Python 的 __getattr__ 魔法方法，自动把请求“转发”给内部的原始环境。
    env = DomainRandomizationWrapper(env)
    # env.randomize_domain()  # 由此可知，此时调用的是 class DomainRandomizationWrapper(Wrapper) 里面的 method 了！

    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(100):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
