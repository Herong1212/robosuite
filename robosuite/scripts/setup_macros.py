"""
This script sets up a private macros file.
The private macros file (macros_private.py) is not tracked by git,
allowing user-specific settings that are not tracked by git.
This script checks if macros_private.py exists.
If applicable, it creates the private macros at robosuite/macros_private.py
"""

import os
import shutil

import robosuite

if __name__ == "__main__":
    # 获取 robosuite 库的基础路径：'/home/robot/robosuite/robosuite'
    base_path = robosuite.__path__[0]
    # 构建 macros.py 文件的完整路径：'/home/robot/robosuite/robosuite/macros.py'
    macros_path = os.path.join(base_path, "macros.py")
    # 构建 macros_private.py 文件的完整路径：'/home/robot/robosuite/robosuite/macros_private.py'
    macros_private_path = os.path.join(base_path, "macros_private.py")

    # 检查源文件是否存在，如果不存在则终止程序
    if not os.path.exists(macros_path):
        print("{} does not exist! Aborting...".format(macros_path))

    # 检查目标文件是否已存在，如果存在则询问用户是否覆盖
    if os.path.exists(macros_private_path):
        ans = input("{} already exists! \noverwrite? (y/n)\n".format(macros_private_path))

        if ans == "y":
            print("REMOVING")
        else:
            exit()

    # 将 macros.py 文件复制为 macros_private.py 文件
    shutil.copyfile(macros_path, macros_private_path)
    print("copied {}\nto {}".format(macros_path, macros_private_path))
    # copied /home/robot/robosuite/robosuite/macros.py to /home/robot/robosuite/robosuite/macros_private.py
