- # Stable Baselines/RL算法/Policy Networks

  > *Stable Baselines*官方文档中文版 [Github](https://github.com/DBWangML/stable-baselines-zh) [CSDN](https://blog.csdn.net/The_Time_Runner/article/details/97392656)
  > 尝试翻译官方文档，水平有限，如有错误万望指正

  *Stable-baselines*提供一系列默认策略（*policies*），可与大部分行动空间同用。你可以指定所用模型类的`policy_kwargs`参数来更改默认策略。然后这些`kwargs`参数会传给实例化的策略（参见案例： [Custom Policy Network](https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html#custom-policy)）如果你希望控制更多策略架构，你也可以创建一个自定义环境（具体参见：[Custom Policy Network](https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html#custom-policy)）。

  > `CnnPolicies`只用于图像。`MplPolicies`用于其他特征类型（例如：机器人关节）

  > 警告：
  >
  > 对于所有算法（除了`DDPG`，`TD3`，`SAC`），训练和测试过程中会剪掉连续行动（避免边界溢出错误）

- ## 可用策略

  | 可用策略                                                     | 策略简介                                                     |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | [`MlpPolicy`](https://stable-baselines.readthedocs.io/en/master/modules/policies.html#stable_baselines.common.policies.MlpPolicy) | Policy object that implements actor critic, using a MLP (2 layers of 64) |
  | [`MlpLstmPolicy`](https://stable-baselines.readthedocs.io/en/master/modules/policies.html#stable_baselines.common.policies.MlpLstmPolicy) | Policy object that implements actor critic, using LSTMs with a MLP feature extraction |
  | [`MlpLnLstmPolicy`](https://stable-baselines.readthedocs.io/en/master/modules/policies.html#stable_baselines.common.policies.MlpLnLstmPolicy) | Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction |
  | [`CnnPolicy`](https://stable-baselines.readthedocs.io/en/master/modules/policies.html#stable_baselines.common.policies.CnnPolicy) | Policy object that implements actor critic, using a CNN (the nature CNN) |
  | [`CnnLstmPolicy`](https://stable-baselines.readthedocs.io/en/master/modules/policies.html#stable_baselines.common.policies.CnnLstmPolicy) | Policy object that implements actor critic, using LSTMs with a CNN feature extraction |
  | [`CnnLnLstmPolicy`](https://stable-baselines.readthedocs.io/en/master/modules/policies.html#stable_baselines.common.policies.CnnLnLstmPolicy) | Policy object that implements actor critic, using a layer normalized LSTMs with a CNN feature extraction |

- ## 基础类Base Classes

  ```python
  stable_baselines.common.policies.BasePolicy(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False, obs_phs=None, add_action_ph=False)
  ```

  基础策略对象（*Policy Object*）

  | 参数          | 数据类型                             | 意义                                                 |
  | ------------- | ------------------------------------ | ---------------------------------------------------- |
  | sess          | TensorFlow session                   | 当前TensorFlow会话                                   |
  | ob_space      | Gym Space                            | 环境的观测空间                                       |
  | ac_space      | Gym Space                            | 环境的行动空间                                       |
  | n_env         | int                                  | 运行的环境数量                                       |
  | n_steps       | int                                  | 每个环境运行的步数                                   |
  | n_batch       | int                                  | 运行批次的数量（n_envs * n_steps）                   |
  | reuse         | bool                                 | 策略是否可重用                                       |
  | scale         | bool                                 | 是否缩放输入                                         |
  | obs_phs       | TensorFlow Tensor, TensorFlow Tensor | 一个元组，分别包含观察占位符和已处理观察占位符的重写 |
  | add_action_ph | bool                                 | 是否创建行动占位符                                   |

  - ***action_ph*** 

    tendorflow.Tensor: 行动的占位符，形状（self.n_batch）+ self.ac_space.shape 

  - ***initial_state*** 

    策略的初始状态，对于前馈策略，None。对于迭代策略，shape(self.n_env,)+state_shape的NumPy数组 

  - ***is_discrete*** 

    bool: 行动空间是否离散

  - ***obs_ph*** 

    tendorflow.Tensor: 观测的占位符，shape(self.n_batch,)+self.ob_space.shape

  - ***proba_step(obs, state=None,mask=None)*** 

    返回单步的行动概率

    | 参数  | 数据类型         | 意义                     |
    | ----- | ---------------- | ------------------------ |
    | obs   | [float] or [int] | 环境的当前观察           |
    | state | [float]          | 最新状态（用于迭代策略） |
    | mask  | [float]          | 最新掩码（用于迭代策略） |

    返回：（[*float*]）行动概率 

  - ***processed_obs***

    tendorflow.Tensor: 已处理的观测。shape(self.n_batch,)+self.ob_space.shape

    处理形式取决于观测空间类型以及缩放参数是否传递给构造器；更多信息参见：`observation_input`  

  - ***step(obs, state=None,mask=None)*** 

    返回单步的策略

    | 参数  | 数据类型         | 意义                     |
    | ----- | ---------------- | ------------------------ |
    | obs   | [float] or [int] | 环境的当前观察           |
    | state | [float]          | 最新状态（用于迭代策略） |
    | mask  | [float]          | 最新掩码（用于迭代策略） |

    返回：([*float*], [*float*], [*float*], [*float*]) 行动(*actions*)，值(*values*)，状态(*states*)，*neglogp* 

- ## class ActorCriticPolicy

  ```python
  stable_baselines.common.policies.ActorCriticPolicy(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, scale=False)
  ```

  实现actor critic的策略对象

  | 参数     | 数据类型           | 意义                               |
  | -------- | ------------------ | ---------------------------------- |
  | sess     | TensorFlow session | 当前TensorFlow会话                 |
  | ob_space | Gym Space          | 环境的观测空间                     |
  | ac_space | Gym Space          | 环境的行动空间                     |
  | n_env    | int                | 运行的环境数量                     |
  | n_steps  | int                | 每个环境运行的步数                 |
  | n_batch  | int                | 运行批次的数量（n_envs * n_steps） |
  | reuse    | bool               | 策略是否可重用                     |
  | scale    | bool               | 是否缩放输入                       |

  - ***action*** 

    tendorflow.Tensor: 随机行动，shape(self.n_batch,)+self.ac_space.shape

  - ***deterministic_action*** 

    tendorflow.Tensor:  确定性行动，shape (self.n_batch, ) + self.ac_space.shape 

  - ***neglogp*** 

     tendorflow.Tensor: self.action采样的行动的负的log似然

  - ***pdtype*** 

     概率分布类型(ProbabilityDistributionType)：随机行动分布的类型

  - ***policy*** 

     tendorflow.Tensor: 策略输出，例如logits 

  - ***policy_proba*** 

     tendorflow.Tensor: 概率分布的参数。取决于***pdtype*** 

  - ***proba_distribution*** 

    概率分布(ProbabilityDistribution)：随机行动的分布

  - ***step(obs, state=None, mask=None, deterministic=False)*** 

    返回单个步骤的策略

    | 参数          | 数据类型         | 意义                     |
    | ------------- | ---------------- | ------------------------ |
    | obs           | [float] or [int] | 环境的当前观察           |
    | state         | [float]          | 最新状态（用于迭代策略） |
    | mask          | [float]          | 最新掩码（用于迭代策略） |
    | deterministic | bool             | 是否返回确定性的行动     |

    **返回：**([float], [float], [float], [float]) actions, values, states, neglogp 

  - ***value(obs, state=None, mask=None)***  

    返回单步骤的值

    | 参数  | 数据类型         | 意义                     |
    | ----- | ---------------- | ------------------------ |
    | obs   | [float] or [int] | 环境的当前观察           |
    | state | [float]          | 最新状态（用于迭代策略） |
    | mask  | [float]          | 最新掩码（用于迭代策略） |

    **返回：**([float])行动的关联值 

  - ***value_flat*** 

     tendorflow.Tensor: 价值估计， shape (self.n_batch, )

  - ***value_fn*** 

     tendorflow.Tensor: 价值估计，shape (self.n_batch, 1) 

- ## class FeedForwardPolicy 

  ```python
  stable_baselines.common.policies.FeedForwardPolicy(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None, act_fun=<MagicMock id='139636329165320'>, cnn_extractor=<function nature_cnn>, feature_extraction='cnn', **kwargs)
  ```

  用前馈神经网络实现actor critic的策略对象

  | 参数               | 数据类型                                                    | 意义                                                         |
  | ------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
  | sess               | TensorFlow session                                          | 当前TensorFlow会话                                           |
  | ob_space           | Gym Space                                                   | 环境的观测空间                                               |
  | ac_space           | Gym Space                                                   | 环境的行动空间                                               |
  | n_env              | int                                                         | 运行的环境数量                                               |
  | n_steps            | int                                                         | 每个环境运行的步数                                           |
  | n_batch            | int                                                         | 运行批次的数量（n_envs * n_steps）                           |
  | reuse              | bool                                                        | 策略是否可重用                                               |
  | layers             | [int]                                                       | （弃用，用net_arch替代）策略用神经网络的大小（如果是None，默认[64,64]） |
  | net_arch           | list                                                        | actor-critic策略网络结构的规范（更多细节参见命令mlp_extractor文档） |
  | act_fun            | tf.func                                                     | 神经网络用的激活函数                                         |
  | cnn_extractor      | function (TensorFlow Tensor, **kwargs): (TensorFlow Tensor) | CNN特征提取                                                  |
  | feature_extraction | str                                                         | 特征提取类型（'cnn'或'mlp'）                                 |
  | kwargs             | dict                                                        | 自然CNN特征提取的额外关键字参数                              |

  - ***proba_step(obs, state=None, mask=None)*** 

    返回单步骤的行动概率

    | 参数  | 数据类型         | 意义                     |
    | ----- | ---------------- | ------------------------ |
    | obs   | [float] or [int] | 环境的当前观察           |
    | state | [float]          | 最新状态（用于迭代策略） |
    | mask  | [float]          | 最新掩码（用于迭代策略） |

    **返回：**([float])行动概率

  - ***step(obs, state=None, mask= None, deterministic= False)*** 

    返回单个步骤的策略

    | 参数          | 数据类型         | 意义                     |
    | ------------- | ---------------- | ------------------------ |
    | obs           | [float] or [int] | 环境的当前观察           |
    | state         | [float]          | 最新状态（用于迭代策略） |
    | mask          | [float]          | 最新掩码（用于迭代策略） |
    | deterministic | bool             | 是否返回确定性的行动     |

    **返回：**([float], [float], [float], [float]) actions, values, states, neglogp 

  - ***value(obs, state=None, mask=None)***  

    返回单步骤的值

    | 参数  | 数据类型         | 意义                     |
    | ----- | ---------------- | ------------------------ |
    | obs   | [float] or [int] | 环境的当前观察           |
    | state | [float]          | 最新状态（用于迭代策略） |
    | mask  | [float]          | 最新掩码（用于迭代策略） |

    **返回：**([float])行动的关联值 

- ## class LstmPolicy 

  ```python
  stable_baselines.common.policies.LstmPolicy(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None, net_arch=None, act_fun=<MagicMock id='139636329067856'>, cnn_extractor=<function nature_cnn>, layer_norm=False, feature_extraction='cnn', **kwargs)
  ```

  用LSTM实现*actor critic*的策略对象

  | 参数               | 数据类型                                                    | 意义                                                         |
  | ------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
  | sess               | TensorFlow session                                          | 当前TensorFlow会话                                           |
  | ob_space           | Gym Space                                                   | 环境的观测空间                                               |
  | ac_space           | Gym Space                                                   | 环境的行动空间                                               |
  | n_env              | int                                                         | 运行的环境数量                                               |
  | n_steps            | int                                                         | 每个环境运行的步数                                           |
  | n_batch            | int                                                         | 运行批次的数量（n_envs * n_steps）                           |
  | n_lstm             | int                                                         | LSTM cells的数量（用于迭代策略）                             |
  | reuse              | bool                                                        | 策略是否可重用                                               |
  | layers             | [int]                                                       | （弃用，用net_arch替代）策略用神经网络的大小（如果是None，默认[64,64]） |
  | net_arch           | list                                                        | actor-critic策略网络结构的规范（更多细节参见命令mlp_extractor文档） |
  | act_fun            | tf.func                                                     | 神经网络用的激活函数                                         |
  | cnn_extractor      | function (TensorFlow Tensor, **kwargs): (TensorFlow Tensor) | CNN特征提取                                                  |
  | layer_norm         | bool                                                        | 是否用层标准化LSTMs                                          |
  | feature_extraction | str                                                         | 特征提取类型（'cnn'或'mlp'）                                 |
  | kwargs             | dict                                                        | 自然CNN特征提取的额外关键字参数                              |

  - ***proba_step(obs, state=None, mask=None)*** 

    返回单步骤的行动概率

    | 参数  | 数据类型         | 意义                     |
    | ----- | ---------------- | ------------------------ |
    | obs   | [float] or [int] | 环境的当前观察           |
    | state | [float]          | 最新状态（用于迭代策略） |
    | mask  | [float]          | 最新掩码（用于迭代策略） |

    **返回：**([float])行动概率

  - ***step(obs, state=None, mask= None, deterministic= False)*** 

    返回单个步骤的策略

    | 参数          | 数据类型         | 意义                     |
    | ------------- | ---------------- | ------------------------ |
    | obs           | [float] or [int] | 环境的当前观察           |
    | state         | [float]          | 最新状态（用于迭代策略） |
    | mask          | [float]          | 最新掩码（用于迭代策略） |
    | deterministic | bool             | 是否返回确定性的行动     |

    **返回：**([float], [float], [float], [float]) actions, values, states, neglogp 

  - ***value(obs, state=None, mask=None)***  

    返回单步骤的值

    | 参数  | 数据类型         | 意义                     |
    | ----- | ---------------- | ------------------------ |
    | obs   | [float] or [int] | 环境的当前观察           |
    | state | [float]          | 最新状态（用于迭代策略） |
    | mask  | [float]          | 最新掩码（用于迭代策略） |

    **返回：**([float])行动的关联值 

- ## MLP Policies

- ## class MlpPolicy

  ```python
  stable_baselines.common.policies.MlpPolicy(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs)
  ```

   用*MLP* (2 layers of 64)实现*actor critic*的策略对象

  | 参数     | 数据类型           | 意义                               |
  | -------- | ------------------ | ---------------------------------- |
  | sess     | TensorFlow session | 当前TensorFlow会话                 |
  | ob_space | Gym Space          | 环境的观测空间                     |
  | ac_space | Gym Space          | 环境的行动空间                     |
  | n_env    | int                | 运行的环境数量                     |
  | n_steps  | int                | 每个环境运行的步数                 |
  | n_batch  | int                | 运行批次的数量（n_envs * n_steps） |
  | reuse    | bool               | 策略是否可重用                     |
  | _kwargs  | dict               | 自然CNN特征提取的额外关键字参数    |

- ## class MlpLstmPolicy 

  用通过*MLP*特征提取的*LSTM*实现*actor critic*的策略对象

  | 参数     | 数据类型           | 意义                               |
  | -------- | ------------------ | ---------------------------------- |
  | sess     | TensorFlow session | 当前TensorFlow会话                 |
  | ob_space | Gym Space          | 环境的观测空间                     |
  | ac_space | Gym Space          | 环境的行动空间                     |
  | n_env    | int                | 运行的环境数量                     |
  | n_steps  | int                | 每个环境运行的步数                 |
  | n_batch  | int                | 运行批次的数量（n_envs * n_steps） |
  | reuse    | bool               | 策略是否可重用                     |
  | kwargs   | dict               | 自然CNN特征提取的额外关键字参数    |

- ## class MlpLnLstmPolicy

  ```python
  stable_baselines.common.policies.MlpLnLstmPolicy(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs)
  ```

  用通过*MLP*特征提取标准化的*LSTM*层实现*actor critic*的策略对象

  | 参数     | 数据类型           | 意义                               |
  | -------- | ------------------ | ---------------------------------- |
  | sess     | TensorFlow session | 当前TensorFlow会话                 |
  | ob_space | Gym Space          | 环境的观测空间                     |
  | ac_space | Gym Space          | 环境的行动空间                     |
  | n_env    | int                | 运行的环境数量                     |
  | n_steps  | int                | 每个环境运行的步数                 |
  | n_batch  | int                | 运行批次的数量（n_envs * n_steps） |
  | reuse    | bool               | 策略是否可重用                     |
  | kwargs   | dict               | 自然CNN特征提取的额外关键字参数    |

- ## CNN Policies

- ## class CnnPolicy 

  ```python
  stable_baselines.common.policies.CnnPolicy(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs)
  ```

  用*CNN* (*the nature CNN*)实现*actor critic*的策略对象

  | 参数     | 数据类型           | 意义                               |
  | -------- | ------------------ | ---------------------------------- |
  | sess     | TensorFlow session | 当前TensorFlow会话                 |
  | ob_space | Gym Space          | 环境的观测空间                     |
  | ac_space | Gym Space          | 环境的行动空间                     |
  | n_env    | int                | 运行的环境数量                     |
  | n_steps  | int                | 每个环境运行的步数                 |
  | n_batch  | int                | 运行批次的数量（n_envs * n_steps） |
  | reuse    | bool               | 策略是否可重用                     |
  | kwargs   | dict               | 自然CNN特征提取的额外关键字参数    |

- ## class CnnLstmPolicy 

  用通过*CNN*特征提取处理的*LSTMs*实现*actor critic*的策略对象

  | 参数     | 数据类型           | 意义                               |
  | -------- | ------------------ | ---------------------------------- |
  | sess     | TensorFlow session | 当前TensorFlow会话                 |
  | ob_space | Gym Space          | 环境的观测空间                     |
  | ac_space | Gym Space          | 环境的行动空间                     |
  | n_env    | int                | 运行的环境数量                     |
  | n_steps  | int                | 每个环境运行的步数                 |
  | n_batch  | int                | 运行批次的数量（n_envs * n_steps） |
  | reuse    | bool               | 策略是否可重用                     |
  | kwargs   | dict               | 自然CNN特征提取的额外关键字参数    |

- ## class CnnLnLstmPolicy

  用通过*CNN*特征提取标准化的*LSTM*层实现*actor critic*的策略对象

  | 参数     | 数据类型           | 意义                               |
  | -------- | ------------------ | ---------------------------------- |
  | sess     | TensorFlow session | 当前TensorFlow会话                 |
  | ob_space | Gym Space          | 环境的观测空间                     |
  | ac_space | Gym Space          | 环境的行动空间                     |
  | n_env    | int                | 运行的环境数量                     |
  | n_steps  | int                | 每个环境运行的步数                 |
  | n_batch  | int                | 运行批次的数量（n_envs * n_steps） |
  | reuse    | bool               | 策略是否可重用                     |
  | kwargs   | dict               | 自然CNN特征提取的额外关键字参数    |

