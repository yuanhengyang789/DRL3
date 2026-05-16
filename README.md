# DRL领航

本仓库是一个基于 ROS / Gazebo 的多机器人强化学习与 Velodyne 激光点云仿真项目，包含机器人场景定义、Gazebo 启动配置、以及基于 TD3 的深度强化学习训练环境。

## 项目结构

- `catkin_ws/`
  - `src/multi_robot_scenario/`
    - `launch/`：包含 Gazebo 与 RViz 启动文件
      - `empty_world.launch`
      - `pioneer3dx_multi3.gazebo.launch`
      - `pioneer3dx.gazebo.launch`
      - `pioneer3dx.rviz`
      - `pioneer3dx.urdf.launch`
      - `TD3_world.launch`
      - `TD3.world`, `TD31.world`, `TD32.world`
    - `xacro/`：Pioneer3DX 机器人、传感器、激光、摄像头等机器人模型描述
    - `src/pedestrian_patrol_plugin.cpp`：场景仿真插件源码
    - `package.xml`, `CMakeLists.txt`：ROS 包描述与构建配置

- `TD3/`
  - `velodyne_env.py`：Gazebo + ROS Velodyne 环境封装，适用于强化学习训练
  - `replay_buffer.py`：TD3 训练中使用的经验回放缓冲区
  - `test_velodyne_pid.py`：PID 控制测试脚本
  - `test_velodyne_t.py`：自定义测试脚本
  - `test_velodyne_td3.py`：TD3 算法测试脚本
  - `models/`：训练生成的模型文件
  - `results/`：训练结果与误差数据
  - `runs/`：训练运行输出记录

## 核心功能

- 多机器人 Gazebo 仿真场景：支持 Pioneer3DX 多机器人布局
- Velodyne 点云处理：使用 ROS 订阅 `/velodyne_points`，转换为深度扫描特征
- 强化学习环境：`GazeboEnv` 提供状态、动作、奖励、碰撞检测和目标判定
- 队形控制与避障：支持队形奖励、障碍物约束、目标点导航
- TD3 算法验证：结合 `test_velodyne_td3.py` 进行策略测试与评估

## 运行说明

### 1. 编译 ROS 包

```bash
cd catkin_ws
catkin_make
source devel/setup.bash
```

### 2. 启动 Gazebo 场景

```bash
roslaunch multi_robot_scenario pioneer3dx_multi3.gazebo.launch
```

或使用 TD3 专用场景：

```bash
roslaunch multi_robot_scenario TD3_world.launch
```

### 3. 运行强化学习环境

在 Python 环境中运行 `TD3/velodyne_env.py` 所依赖的训练脚本或测试脚本。例如：

```bash
python3 TD3/test_velodyne_td3.py
```

## 依赖项

- ROS (推荐 ROS Melodic / Noetic)
- Gazebo
- `xacro`
- `gazebo_ros`
- Python 3
- `numpy`, `rospy`, `sensor_msgs`, `gazebo_msgs`, `nav_msgs`, `std_srvs`, `visualization_msgs`
- `squaternion`

## 注意事项

- 本项目的 ROS 包依赖于 `catkin` 构建系统，需在 Linux 环境下运行。
- `TD3/velodyne_env.py` 中默认使用 `gazebo/set_model_state`、`/gazebo/pause_physics`、`/gazebo/unpause_physics`、`/gazebo/reset_world` 等 ROS 服务。
- 如果要训练新模型，请先确认 Gazebo 仿真环境已正常启动，并且 `robot_names` 与 TF / topic 命名一致。

## 目录说明

- `catkin_ws/src/multi_robot_scenario/launch/`：仿真世界与机器人启动配置
- `catkin_ws/src/multi_robot_scenario/xacro/`：机器人模型文件
- `TD3/velodyne_env.py`：强化学习环境接口
- `TD3/models/`：保存的策略模型
- `TD3/results/`：测试误差与结果文件
- `TD3/runs/`：训练日志和运行记录


欢迎根据需求继续补充项目说明，例如训练参数、实际运行命令、测试结果分析等。
