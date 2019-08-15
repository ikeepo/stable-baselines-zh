- # Stable Baselines/RL算法/ACER

  > *Stable Baselines*官方文档中文版 [Github](https://github.com/DBWangML/stable-baselines-zh) [CSDN](https://blog.csdn.net/The_Time_Runner/article/details/97392656)
  > 尝试翻译官方文档，水平有限，如有错误万望指正

  [Sample Efficient Actor-Critic with Experience Replay (ACER)](https://arxiv.org/abs/1611.01224) 结合先前几个算法的思想：它使用多个*workers*（类似A2C），实现重播缓冲（如DQN），用重跟踪来计算Q值估计、重要性采样和信任区间。

- ## 要点核心

  - 原始文献： https://arxiv.org/abs/1611.01224  
  - `python -m stable_baselines.acer.run_atari`在*Atari*游戏以 40M frames = 10M timesteps运行算法。更多选项参见帮助文档（`-h`）  

- ## 适用情况

  - 迭代策略：✔️

  - 多进程：✔️

  - *Gym*空间：

    | Space         | Action | Observation |
    | ------------- | ------ | ----------- |
    | Discrete      | ✔️      | ✔️           |
    | Box           | ❌      | ✔️           |
    | MultiDiscrete | ❌      | ✔️           |
    | MultiBinary   | ❌      | ✔️           |

- ## 案例

  

  ```python
  import gym
  
  from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
  from stable_baselines.common.vec_env import SubprocVecEnv
  from stable_baselines import ACER
  
  # multiprocess environment
  n_cpu = 4
  env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
  
  model = ACER(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("acer_cartpole")
  
  del model # remove to demonstrate saving and loading
  
  model = ACER.load("acer_cartpole")
  
  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()
  ```

- ## 参数

- ***class ACER***

  ```python
  stable_baselines.acer.ACER(policy, env, gamma=0.99, n_steps=20, num_procs=1, q_coef=0.5, ent_coef=0.01, max_grad_norm=10, learning_rate=0.0007, lr_schedule='linear', rprop_alpha=0.99, rprop_epsilon=1e-05, buffer_size=5000, replay_ratio=4, replay_start=1000, correction_term=10.0, trust_region=True, alpha=0.99, delta=1, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False)
  ```

  *ACER (Actor-Critic with Experience Replay)*模型类， https://arxiv.org/abs/1611.01224  

  | 参数                 | 数据类型                 | 意义                                                         |
  | -------------------- | ------------------------ | ------------------------------------------------------------ |
  | policy               | ActorCriticPolicy or str | 所用策略模型（MlpPolicy, CnnPolicy, CnnLstmPolicy, …）       |
  | env                  | Gym environment or str   | 学习所用环境（如果注册在Gym，可以是str）                     |
  | gamma                | float                    | 贴现因子                                                     |
  | n_steps              | int                      | 每次更新每个环境所运行时间步（例如：当n_env是同时运行的环境副本数量时，batch=n_steps*n_env） |
  | num_procs            | int                      | TensorFlow运行的线程数                                       |
  | q_coef               | float                    | Q值的损失权重                                                |
  | ent_coef             | float                    | 信息熵损失的权重                                             |
  | max_grad_norm        | float                    | 最大梯度的裁剪值                                             |
  | learning_rate        | float                    | 学习率                                                       |
  | rprop_alpha          | float                    | RMSProp衰减参数（默认：0.99）                                |
  | rprop_epsilon        | float                    | RMSProp epsilon（RMSProp更新中分母的稳定平方根计算）（默认1e-5） |
  | lr_schedule          | str                      | 更新学习率的调度程序类型（‘linear’, ‘constant’, ‘double_linear_con’, ‘middle_drop’ or ‘double_middle_drop’） |
  | buffer_size          | int                      | 以步数为单位的缓冲区大小                                     |
  | replay_ratio         | float                    | 使用泊松分布，每个策略上重播学习的次数                       |
  | replay_start         | int                      | 缓冲区内时间步的最小数，学习重播之前                         |
  | correction_term      | float                    | 重要性权重裁剪因子（默认10）                                 |
  | trust_region         | bool                     | 算法是否评估新旧策略之间的梯度KL散度，并用其确定时间步（默认True） |
  | alpha                | float                    | 参数的指数移动平均的衰减率                                   |
  | delta                | float                    | 新旧策略间的最大KL三度（默认1）                              |
  | verbose              | int                      | 日志信息级别：0None；1训练信息；2tensorflow调试              |
  | tensorboard_log      | str                      | tensorboard的日志位置（如果时None，没有日志）                |
  | _init_setup_model    | bool                     | 实例化创建过程中是否建立网络（只用于载入）                   |
  | policy_kwargs        | dict                     | 创建过程中传递给策略的额外参数                               |
  | full_tensorboard_log | bool                     | 当使用tensorboard时，是否记录额外日志（这个日志会占用大量空间） |

  - ***action_probability(observation, state=None, mask=None, actions=None, logp=False)*** 

    如果`actions`时`None`，那么从给定观测中获取模型的行动概率分布。

    > 输出取决于行动空间：

    - 离散：每个可能行动的概率
    - *Box*：行动输出的均值和标准差

    然而，如果`actions`不是`None`，这个函数会返回给定行动与参数（观测，状态，...）用于此模型的概率。对于离散行动空间，它返回概率密度；对于连续行动空间，则是概率密度。这是因为在连续空间，概率密度总是*0*，更详细的解释见 <http://blog.christianperone.com/2019/01/>  

    | 参数        | 数据类型   | 意义                                                         |
    | ----------- | ---------- | ------------------------------------------------------------ |
    | observation | np.ndarray | 输入观测                                                     |
    | state       | np.ndarray | 最新状态（可以时None，用于迭代策略）                         |
    | mask        | np.ndarray | 最新掩码（可以时None，用于迭代策略）                         |
    | actions     | np.ndarray | （可选参数）为计算模型为每个给定参数选择给定行动的似然。行动和观测必须具有相同数目（None返回完全动作分布概率） |
    | logp        | bool       | （可选参数）当指定行动，返回log空间的概率。如果action是None，则此参数无效 |

    **返回：**（*np.ndarray*）模型的（*log*）行动概率

  - ***get_env()*** 

    返回当前环境（如果没有定义可以是*None*）

    **返回：**（*Gym Environment*）当前环境

  - ***get_parameter_list()*** 

    获取模型参数的*tensorflow*变量

    包含连续训练（保存/载入）所用的所有必要变量

    **返回：**（*list*）*tensorflow*变量列表

  - ***get_parameters()*** 

    获取当前模型参数作为变量名字典 -> *ndarray* 

    **返回：**（*OrderedDict*）变量名字典 -> 模型参数的*ndarray*

  - ***learn(total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name='A2C', reset_num_timesteps=True)*** 

    返回一个训练好的模型

    | 参数                | 数据类型              | 意义                                                         |
    | ------------------- | --------------------- | ------------------------------------------------------------ |
    | total_timesteps     | int                   | 训练用样本总数                                               |
    | seed                | int                   | 训练用初始值，如果None:保持当前种子                          |
    | callback            | function (dict, dict) | 与算法状态的每步调用的布尔函数。采用局部或全局变量。如果它返回false,训练被终止 |
    | log_interval        | int                   | 记录日志之前的时间步数                                       |
    | tb_log_name         | str                   | 运行tensorboard日志的名称                                    |
    | reset_num_timesteps | bool                  | 是否重置当前时间步数（日志中使用）                           |

    **返回：**(*BaseRLModel*) 训练好的模型

  - ***classmethod load(load_path, env=None, kwargs)*** 

    从文件中载入模型

    | 参数      | 数据类型         | 意义                                                         |
    | --------- | ---------------- | ------------------------------------------------------------ |
    | load_path | str or file-like | 文件路径                                                     |
    | env       | Gym Envrionment  | 载入模型运行的新环境（如果你只是从训练好的模型来做预测可以是None） |
    | kwargs    |                  | 载入过程中能改变模型的额外                                   |

    

  - ***load_parameters(load_path_or_dict, exact_match=True)*** 

    从文件或字典中载入模型参数

    字典关键字是*tensorflow*变量名，可以用`get_parameters`函数获取。如果`exact_match`是`True`,字典应该包含所有模型参数的关键字，否则报错*RunTimeError*。如果是False,只有字典包含的变量会被更新。

    此函数并不载入agent的超参数

    > 警告：
    >
    > 此函数不更新训练器/优化器的变量（例如：*momentum*）。因为使用此函数的这种训练可能会导致低优化结果

    | 参数              | 数据类型         | 意义                                                         |
    | ----------------- | ---------------- | ------------------------------------------------------------ |
    | load_path_or_dict | str or file-like | 保存参数或变量名字典位置->载入的是ndarrays                   |
    | exact_match       | bool             | 如果是True，期望载入关键字包含模型所有变量的字典；如果是False，只载入字典中提及的参数。默认True |

  - ***predict(observation, state=None, mask=None, deterministic=False)*** 

    获取从参数得到的模型行动

    | 参数          | 数据类型   | 意义                                 |
    | ------------- | ---------- | ------------------------------------ |
    | observation   | np.ndarray | 输入观测                             |
    | state         | np.ndarray | 最新状态（可以时None，用于迭代策略） |
    | mask          | np.ndarray | 最新掩码（可以时None，用于迭代策略） |
    | deterministic | bool       | 是否返回确定性的行动                 |

    **返回：**(*np.ndarray, np.ndarray*) 模型的行动和下一状态（用于迭代策略）

  - ***pretrain(dataset, n_epochs=10, learning_rate=0.0001, adam_epsilon=1e-08, val_interval=None)*** 

    用行为克隆预训练一个模型：在给定专家数据集上的监督学习

    目前只支持*Box*和离散空间

    | 参数          | 数据类型      | 意义                                               |
    | ------------- | ------------- | -------------------------------------------------- |
    | dataset       | ExpertDataset | 数据集管理器                                       |
    | n_epochs      | int           | 训练集上的迭代次数                                 |
    | learning_rate | float         | 学习率                                             |
    | adam_epsilon  | float         | adam优化器的$\epsilon$ 值                          |
    | val_interval  | int           | 报告每代的训练和验证损失。默认最大纪元数的十分之一 |

    返回：(*BaseRLModel*) 预训练好的模型

  - ***save(save_path)*** 

    保存当前参数到文件

    | 参数      | 数据类型                | 意义     |
    | --------- | ----------------------- | -------- |
    | save_path | str or file-like object | 保存位置 |

  - ***set_env(env)*** 

    检查环境的有效性，如果是一致的，将其设置为当前环境

    | 参数 | 数据类型        | 意义                |
    | ---- | --------------- | ------------------- |
    | env  | Gym Environment | x学习一个策略的环境 |

  - ***setup_model()*** 

    创建训练模型所必须的函数的*tensorflow*图表
