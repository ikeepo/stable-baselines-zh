- # Stable Baselines/用户向导/RL算法

  > Stable Baselines官方文档中文版 [Github](https://github.com/DBWangML/stable-baselines-zh) [CSDN](https://blog.csdn.net/The_Time_Runner/article/details/97392656)
  > 尝试翻译官方文档，水平有限，如有错误万望指正

  下面这个表格展示了stable baselines项目中采用的所有RL算法及其重要特征：迭代策略、离散/连续行动、多线程

| Name                                                         | Refactored [[1\]](https://stable-baselines.readthedocs.io/en/master/guide/algos.html#f1) | Recurrent | `Box`                                                        | `Discrete` | Multi Processing                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------- | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
| A2C                                                          | ✔️                                                            | ✔️         | ✔️                                                            | ✔️          | ✔️                                                            |
| ACER                                                         | ✔️                                                            | ✔️         | ❌ [[4\]](https://stable-baselines.readthedocs.io/en/master/guide/algos.html#f4) | ✔️          | ✔️                                                            |
| ACKTR                                                        | ✔️                                                            | ✔️         | ❌ [[4\]](https://stable-baselines.readthedocs.io/en/master/guide/algos.html#f4) | ✔️          | ✔️                                                            |
| DDPG                                                         | ✔️                                                            | ❌         | ✔️                                                            | ❌          | ✔️ [[3\]](https://stable-baselines.readthedocs.io/en/master/guide/algos.html#f3) |
| DQN                                                          | ✔️                                                            | ❌         | ❌                                                            | ✔️          | ❌                                                            |
| HER                                                          | ✔️                                                            | ❌         | ✔️                                                            | ✔️          | ❌                                                            |
| GAIL [[2\]](https://stable-baselines.readthedocs.io/en/master/guide/algos.html#f2) | ✔️                                                            | ✔️         | ✔️                                                            | ✔️          | ✔️ [[3\]](https://stable-baselines.readthedocs.io/en/master/guide/algos.html#f3) |
| PPO1                                                         | ✔️                                                            | ❌         | ✔️                                                            | ✔️          | ✔️ [[3\]](https://stable-baselines.readthedocs.io/en/master/guide/algos.html#f3) |
| PPO2                                                         | ✔️                                                            | ✔️         | ✔️                                                            | ✔️          | ✔️                                                            |
| SAC                                                          | ✔️                                                            | ❌         | ✔️                                                            | ❌          | ❌                                                            |
| TD3                                                          | ✔️                                                            | ❌         | ✔️                                                            | ❌          | ❌                                                            |
| TRPO                                                         | ✔️                                                            | ❌         | ✔️                                                            | ✔          | ✔️ [[3\]](https://stable-baselines.readthedocs.io/en/master/guide/algos.html#f3) |

[1]   是否重构以适应`BaseRLModel`类

[2]   只用于TRPO 

[3]   (1,2,3,4)用MPI实现多重处理 

[4]   在项目范围内，(1,2)必做

> 目前任何算法都不支持类似`Dict`或`Tuple`这种非数组空间，除非`HER`与`gym.GoalEnv`一起用，此时会支持`Dict` 

各类行动`gym.spaces`:

- `Box`: 一个包含行动空间中每个点的N维盒子
- `Discrete`: 一组可能的行动，每个时间步中只会采用一个
- `MultiDiscrete`: 一组可能的行动，每个时间步每个离散集中只有一个行动被采用
- `MultiBinary`: 一组可能的行动，每个时间步中任何行动都可能以任何结合方式使用
