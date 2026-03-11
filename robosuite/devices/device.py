import abc  # for abstract base class definitions


class Device(metaclass=abc.ABCMeta):
    """
    Base class for all robot controllers. Defines basic interface for all controllers to adhere to.
    定义所有机器人控制器的基类，为所有控制器定义必须遵守的基本接口
    """

    # @abc.abstractmethod：标记此方法为抽象方法，子类必须实现
    @abc.abstractmethod
    def start_control(self):
        """
        Method that should be called externally before controller can start receiving commands. --  外部调用此方法后，控制器才能开始接收命令
        """

        # 抛出未实现异常，强制子类重写此方法
        raise NotImplementedError

    @abc.abstractmethod
    def get_controller_state(self):
        """
        Returns the current state of the device, a dictionary of pos, orn, grasp, and reset.
        返回设备当前状态，包括位置(pos)、方向(orn)、抓取(grasp)和重置(reset)信息
        """

        # 抛出未实现异常，强制子类重写此方法
        raise NotImplementedError
