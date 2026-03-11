"""
This file implements a wrapper for facilitating domain randomization over robosuite environments.

此文件实现了一个用于在 robosuite 环境中促进领域随机化的包装器。
"""

import numpy as np

from robosuite.utils.mjmod import CameraModder, DynamicsModder, LightingModder, TextureModder
from robosuite.wrappers import Wrapper

# * 以下字典定义了随机化的“尺度”

DEFAULT_COLOR_ARGS = {
    "geom_names": None,  # all geoms are randomized -- 指定要随机化颜色的几何体名称列表。设置为 None 表示对场景中“所有”物体进行随机化。
    "randomize_local": True,  # sample nearby colors -- 采样附近的颜色，即是否在原始颜色附近进行微调。True 表示颜色只会在原色基础上轻微波动；False 则会变成完全随机的各种颜色。
    "randomize_material": True,  # randomize material reflectance / shininess / specular -- 随机化材料反射率/光泽度/镜面反射
    "local_rgb_interpolation": 0.2,  # RGB颜色插值的范围。数值越大，随机出来的颜色偏离原始颜色的程度就越高（0.2 属于轻微变色）。
    "local_material_interpolation": 0.3,  # 材质属性插值的范围。数值越大，物体的反光和质感变化就越剧烈。
    "texture_variations": [
        "rgb",
        "checker",
        "noise",
        "gradient",
    ],  # all texture variation types -- 所有允许变化的纹理类型列表。rgb: 纯色变化；checker: 棋盘格纹理；noise: 噪点图；gradient: 渐变色。
    "randomize_skybox": True,  # by default, randomize skybox too -- 默认也随机化天空盒（背景）。默认开启，会让仿真的背景天空也跟着随机变换颜色和光照。
}

DEFAULT_CAMERA_ARGS = {
    "camera_names": None,  # all cameras are randomized -- 指定要随机化的相机名称列表。设置为 None 表示对场景中所有的相机（如主视角、手眼相机等）都进行随机化。
    "randomize_position": True,  # 是否随机化相机的三维位置（X, Y, Z）。
    "randomize_rotation": True,  # 是否随机化相机的旋转角度（Roll, Pitch, Yaw）。
    "randomize_fovy": True,  # 是否随机化视场角（Field of View，FOVY）。
    "position_perturbation_size": 0.01,  # 位置随机抖动的范围大小。单位是米，0.01 表示相机会在原始位置正负 1 厘米的范围内随机跳动。
    "rotation_perturbation_size": 0.087,  # 旋转随机抖动的范围大小。单位是弧度，0.087 弧度大约等于 5 度。表示相机会围绕原角度随机偏转约 5 度。
    "fovy_perturbation_size": 5.0,  # 视场角（FOV）抖动的范围大小。单位是度，5.0 表示相机的视野开角会在原基础上随机增减 5 度。
}

DEFAULT_LIGHTING_ARGS = {
    "light_names": None,  # all lights are randomized -- 指定要随机化的灯光名称。None 表示对场景中所有的灯（主光源、环境灯）都进行随机化。
    "randomize_position": True,  # 是否随机化灯的位置（让影子的方向发生变化）
    "randomize_direction": True,  # 是否随机化灯的照射方向
    "randomize_specular": True,  # 是否随机化镜面反射强度（决定物体表面高光有多亮）
    "randomize_ambient": True,  # 是否随机化环境光强度（决定阴影处有多黑）
    "randomize_diffuse": True,  # 是否随机化漫反射强度（决定物体固有颜色有多亮）
    "randomize_active": True,  # 是否随机开关灯（模拟灯光闪烁或熄灭的情况）
    # 扰动范围设置：
    "position_perturbation_size": 0.1,  # 位置扰动大小，灯光位置随机移动 0.1 米
    "direction_perturbation_size": 0.35,  # 方向扰动大小，灯光方向随机偏转的角度大小
    "specular_perturbation_size": 0.1,  # 镜面反射的变化范围
    "ambient_perturbation_size": 0.1,  # 环境光的变化范围
    "diffuse_perturbation_size": 0.1,  # 漫反射的变化范围
}

# ps 这是最硬核的部分。它通过修改物理公式里的系数，让机器人感觉到物体“一会儿重、一会儿滑”。
DEFAULT_DYNAMICS_ARGS = {
    # Opt parameters - 可选参数
    "randomize_density": True,  # 随机化密度
    "randomize_viscosity": True,  # 随机化粘度
    "density_perturbation_ratio": 0.1,  # 密度扰动比例
    "viscosity_perturbation_ratio": 0.1,  # 粘度扰动比例
    # Body parameters - 身体参数
    "body_names": None,  # 所有身体都进行随机化
    "randomize_position": True,  # 随机化重心位置（重心不稳）
    "randomize_quaternion": True,  # 随机化四元数（旋转）
    "randomize_inertia": True,  # 随机化转动惯量
    "randomize_mass": True,  # 随机化质量（让物体一会儿轻一会儿重）
    "position_perturbation_size": 0.0015,  # 重心位置微调 1.5 毫米
    "quaternion_perturbation_size": 0.003,  # 四元数扰动大小
    "inertia_perturbation_ratio": 0.02,  # 惯性扰动比例
    "mass_perturbation_ratio": 0.02,  # 质量扰动比例
    # Geom parameters - 几何体参数
    "geom_names": None,  # 所有几何体都进行随机化
    "randomize_friction": True,  # 随机化摩擦力（模拟物体表面有的滑、有的涩）
    "randomize_solref": True,  # 随机化接触约束参数（影响物体碰撞时的软硬程度）
    "randomize_solimp": True,  # 随机化阻抗参数
    "friction_perturbation_ratio": 0.1,  # 摩擦力上下波动 10%
    "solref_perturbation_ratio": 0.1,  # solref扰动比例
    "solimp_perturbation_ratio": 0.1,  # solimp扰动比例
    # Joint parameters - 关节参数
    "joint_names": None,  # 所有关节都进行随机化
    "randomize_stiffness": True,  # 随机化关节刚度（弹簧感）
    "randomize_frictionloss": True,  # 随机化关节摩擦损耗
    "randomize_damping": True,  # 随机化关节阻尼（粘滞感）
    "randomize_armature": True,  # 随机化电机转子惯量
    "stiffness_perturbation_ratio": 0.1,  # 刚度波动 10%
    "frictionloss_perturbation_size": 0.05,  # 摩擦损失扰动大小
    "damping_perturbation_size": 0.01,  # 阻尼扰动大小
    "armature_perturbation_size": 0.01,  # 转子参数扰动大小
}


class DomainRandomizationWrapper(Wrapper):
    """
    Wrapper that allows for domain randomization mid-simulation. -- 允许在仿真过程中进行领域随机化的包装器。

    Args:
        env (MujocoEnv): The environment to wrap. -- 要包装的环境。

        seed (int): Integer used to seed all randomizations from this wrapper. It is
            used to create a np.random.RandomState instance to make sure samples here
            are isolated from sampling occurring elsewhere in the code. If not provided,
            will default to using global random state. -- 用于设置包装器所有随机化的种子整数。
            它用于创建 np.random.RandomState 实例，以确保此处的采样与代码其他地方的采样隔离。如果不提供，则默认使用全局随机状态。

        randomize_color (bool): if True, randomize geom colors and texture colors. -- 如果为 True, 则随机化几何体颜色和纹理颜色

        randomize_camera (bool): if True, randomize camera locations and parameters. -- 如果为 True, 则随机化相机位置和参数

        randomize_lighting (bool): if True, randomize light locations and properties. -- 如果为 True, 则随机化灯光位置和属性

        randomize_dyanmics (bool): if True, randomize dynamics parameters. -- 如果为 True, 则随机化动力学参数

        color_randomization_args (dict): Color-specific randomization arguments. -- 颜色特定的随机化参数

        camera_randomization_args (dict): Camera-specific randomization arguments. -- 相机特定的随机化参数

        lighting_randomization_args (dict): Lighting-specific randomization arguments. --  灯光特定的随机化参数

        dynamics_randomization_args (dict): Dyanmics-specific randomization arguments. -- 动力学特定的随机化参数

        randomize_on_reset (bool): if True, randomize on every call to @reset. This, in
            conjunction with setting @randomize_every_n_steps to 0, is useful to
            generate a new domain per episode. -- 如果为 True, 则在每次调用 @reset 时进行随机化。这与将 @randomize_every_n_steps
            设置为 0 相结合，对于每 episode 生成一个新 domain很有用。

        randomize_every_n_steps (int): determines how often randomization should occur. Set
            to 0 if randomization should happen manually (by calling @randomize_domain). -- 确定随机化的频率。
            如果应该手动发生随机化 (通过调用 @randomize_domain), 则设置为 0

    """

    def __init__(
        self,
        env,
        seed=None,
        randomize_color=True,
        randomize_camera=True,
        randomize_lighting=True,
        randomize_dynamics=True,
        color_randomization_args=DEFAULT_COLOR_ARGS,
        camera_randomization_args=DEFAULT_CAMERA_ARGS,
        lighting_randomization_args=DEFAULT_LIGHTING_ARGS,
        dynamics_randomization_args=DEFAULT_DYNAMICS_ARGS,
        randomize_on_reset=True,  # 每次 reset() 时随机化
        randomize_every_n_steps=1,  # 每隔 N 步随机一次。Demo 里设为 1，所以画面会疯狂闪烁。在实际训练中，我们通常设为 0（只在 reset 时变）。
    ):
        super().__init__(env)

        self.seed = seed
        if seed is not None:
            self.random_state = np.random.RandomState(seed)  # 创建独立的随机状态
        else:
            self.random_state = None  # 使用全局随机状态

        self.randomize_color = randomize_color  # 是否随机化颜色
        self.randomize_camera = randomize_camera  # 是否随机化相机
        self.randomize_lighting = randomize_lighting  # 是否随机化照明
        self.randomize_dynamics = randomize_dynamics  # 是否随机化动力学
        self.color_randomization_args = color_randomization_args  # 颜色随机化参数
        self.camera_randomization_args = camera_randomization_args  # 相机随机化参数
        self.lighting_randomization_args = lighting_randomization_args  # 照明随机化参数
        self.dynamics_randomization_args = dynamics_randomization_args  # 动力学随机化参数
        self.randomize_on_reset = randomize_on_reset  # 重置时是否随机化
        self.randomize_every_n_steps = randomize_every_n_steps  # 每n步随机化一次

        self.step_counter = 0  # 步数计数器

        self.modders = []  # 修改器列表

        # 纹理修改器
        if self.randomize_color:
            self.tex_modder = TextureModder(
                sim=self.env.sim, random_state=self.random_state, **self.color_randomization_args
            )
            self.modders.append(self.tex_modder)

        # 相机修改器
        if self.randomize_camera:
            self.camera_modder = CameraModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.camera_randomization_args,
            )
            self.modders.append(self.camera_modder)

        # 照明修改器
        if self.randomize_lighting:
            self.light_modder = LightingModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.lighting_randomization_args,
            )
            self.modders.append(self.light_modder)

        # 动力学修改器
        if self.randomize_dynamics:
            self.dynamics_modder = DynamicsModder(
                sim=self.env.sim,
                random_state=self.random_state,
                **self.dynamics_randomization_args,
            )
            self.modders.append(self.dynamics_modder)

        self.save_default_domain()  # 保存默认领域参数

    def reset(self):
        """
        Extends superclass method to reset the domain randomizer.

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # undo all randomizations -- 撤销所有随机化
        self.restore_default_domain()

        # normal env reset -- 正常环境重置
        ret = super().reset()

        # save the original env parameters -- 保存原始环境参数
        self.save_default_domain()

        # reset counter for doing domain randomization at a particular frequency -- 重置按特定频率进行领域随机化的计数器
        self.step_counter = 0

        # update sims -- 更新仿真
        for modder in self.modders:
            modder.update_sim(self.env.sim)

        if self.randomize_on_reset:
            # domain randomize + regenerate observation -- 领域随机化 + 重新生成观察
            self.randomize_domain()
            ret = self.env._get_observations()

        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate domain randomization

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        # Step the internal randomization state -- 步进内部随机化状态
        self.step_randomization()

        return super().step(action)

    def step_randomization(self):
        """
        Steps the internal randomization state
        """
        # functionality for randomizing at a particular frequency -- 按特定频率进行随机化的功能
        if self.randomize_every_n_steps > 0:
            if self.step_counter % self.randomize_every_n_steps == 0:
                self.randomize_domain()
        self.step_counter += 1

    def randomize_domain(self):
        """
        Runs domain randomization over the environment. -- 在环境中运行领域随机化。
        """
        for modder in self.modders:
            modder.randomize()

    def save_default_domain(self):
        """
        Saves the current simulation model parameters so that they can be restored later. -- 保存当前仿真模型参数以便稍后恢复。
        """
        for modder in self.modders:
            modder.save_defaults()

    def restore_default_domain(self):
        """
        Restores the simulation model parameters saved in the last call to @save_default_domain. -- 恢复上次调用 @save_default_domain 时保存的仿真模型参数。
        """
        for modder in self.modders:
            modder.restore_defaults()
