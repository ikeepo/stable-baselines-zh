- # Stable Baselines/用户向导/RL Baselines Zoo

  > *Stable Baselines*官方文档中文版 [Github](https://github.com/DBWangML/stable-baselines-zh) [CSDN](https://blog.csdn.net/The_Time_Runner/article/details/97392656)
  > 尝试翻译官方文档，水平有限，如有错误万望指正

  *[RL Baselines Zoo](https://github.com/araffin/rl-baselines-zoo)*. 是一系列用*Stable Baselines*预训练的强化学习*agents*。它也提供用于训练、评估*agents*、微调超参数、记录视频的基础脚本。

  这个版本库的目标：

  1. 提供一个简单界面用于训练和享用*Rl agents* 
  2. 用基准测试不同强化学习算法
  3. 为每一个环境和RL算法提供调整后超参数
  4. 享受训练好的*agents*带来的种种欢乐

- ## 安装

  1. ### 安装依赖

     ```python
     apt-get install swig cmake libopenmpi-dev zlib1g-dev ffmpeg
     pip install stable-baselines box2d box2d-kengz pyyaml pybullet optuna pytablewriter
     ```

  2. ### 克隆仓库

     ```python
     git clone https://github.com/araffin/rl-baselines-zoo
     ```

- ## 训练Agent

  每个环境的超参数定义在`hyperparameters/algo_name.yml` 

  如果文件中包含环境，你可以如此训练*agent*：

  ```python
  python train.py --algo algo_name --env env_id
  ```

  举例（带*tensorboard*支持）：

  ```python
  python train.py --algo ppo2 --env CartPole-v1 --tensorboard-log /tmp/stable-baselines/
  ```

  针对多环境（一次调用）和用*tensorboard*记录日志进行训练：

  ```python
  python train.py --algo a2c --env MountainCar-v0 CartPole-v1 --tensorboard-log /tmp/stable-baselines/
  ```

  继续训练（这里，载入预训练的*agent*为*Breakout*并连续训练5000步）：

  ```python
  python train.py --algo a2c --env BreakoutNoFrameskip-v4 -i trained_agents/a2c/BreakoutNoFrameskip-v4.pkl -n 5000
  ```

- ## 享用训练好的Agent

  如果存在训练好的*agent*，你可以用下述命令查看其实际应用：

  ```python
  python enjoy.py --algo algo_name --env env_id
  ```

  例如，在5000时间步内效用*Breakout*中的*A2C*：

  ```python
  python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder trained_agents/ -n 5000
  ```

- ## 优化超参数

  我们用 *[Optuna](https://optuna.org/)*优化超参数。

  为PPO2调整超参数，使用随机抽样器和中值修剪器，2个平行工作，预算1000次测试，最多50000步：

  ```python
  python train.py --algo ppo2 --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 \
    --sampler random --pruner median
  ```

- ## Colab Botebook：在线训练

  你可以用*Google [colab notebook](https://colab.research.google.com/drive/1cPGK3XrCqEs3QLqiijsfib9OFht3kObX)*在线训练*agents*。

  > 你可以在仓库 [README](https://github.com/araffin/rl-baselines-zoo)中发现更多关于RL Baselines zoo的信息。例如，如果记录一个训练好agent的视频。
