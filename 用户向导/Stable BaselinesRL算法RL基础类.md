- # Stable Baselines/RL算法/RL基础类

  > *Stable Baselines*官方文档中文版 [Github](https://github.com/DBWangML/stable-baselines-zh) [CSDN](https://blog.csdn.net/The_Time_Runner/article/details/97392656)
  > 尝试翻译官方文档，水平有限，如有错误万望指正

  所有强化学习（*RL*）算法的公共接口

- ## BaseRLModel

  ```python
  class stable_baselines.common.base_class.BaseRLModel(policy, env, verbose=0, *, requires_vec_env, policy_base, policy_kwargs=None)
  ```

  基础RL模型。

  **参数介绍：**

  | 参数             | 类型            | 意义                                                         |
  | ---------------- | --------------- | ------------------------------------------------------------ |
  | policy           | BasePolicy      | 策略对象                                                     |
  | env              | Gym environment | 学习环境（如果已在Gym注册，可以是str。如果为了载入已训练好的模型可以是None） |
  | verbose          | int             | 信息显示级别：0是None；1是训练信息；2是tensorflow debug      |
  | requires_vec_env | bool            | 此模型是否需要矢量化环境                                     |
  | policy_base      | BasePolicy      | 此方法使用的基础策略                                         |

  **函数介绍：** 

  1. ***action_probability()*** 

  ```python
  action_probability(observation, state=None, mask=None, actions=None, logp=False)
  ```

  如果`actions`是`None`，那就从给定观测中获取模型行动的概率分布。

  依据行动空间有两种输出：

  - 离散：每种可能行动的概率
  - *Box*：行动输出的均值和标准差

  然而，如果`actions`不是`None`，这个函数会返回此模型采用给定行动和给定参数（观测，状态，...）的概率。对于离散行动空间，返回概率质量；对于连续行动空间，则是概率密度。这是因为在连续空间，概率质量总是*0*，详细解释见此[链接](http://blog.christianperone.com/2019/01/)。

  | 参数        | 类型       | 意义                                                         |
  | ----------- | ---------- | ------------------------------------------------------------ |
  | observation | np.ndassay | 输入观测                                                     |
  | state       | np.ndarray | 最新状态（可以是None，用于迭代策略）                         |
  | mask        | np.ndarray | 最新掩码（可以是None，用于迭代策略）                         |
  | actions     | np.ndarray | （可选参数）计算模型为每个指定参数选择指定行动的可能性。行动和观测必须具有相同数量（设为None则返回完整的行动概率分布） |
  | logp        | bool       | （可选参数）当指定行动，返回log空间中的概率。如果行动是None则无影响 |

  ***返回：***(*np.ndarray*) 模型的(*log*)行动概率

  2. ***get_env()*** 

     返回当前环境（如果没有定义则返回`None`）

     ***返回：***（*Gym Environment*）当前环境

  3.  ***get_parameter_list()***  

     获取模型参数的*tensorflow*变量

     这包含了连续训练所必须的所有变量（保存/载入）

     ***返回：***（*list*）*tensorflow*变量列表

  4. ***get_parameters()***  

     获取当前模型参数作为变量名字典-->*ndarray* 

     ***返回：***（*OrderedDict*）变量名字典-->模型擦书的*ndarray* 

  5. ***learn()*** 

     ```python
     learn(total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name='run', reset_num_timesteps=True)
     ```

     返回一个训练好的模型

     | 参数            | 类型                  | 意义                                                         |
     | --------------- | --------------------- | ------------------------------------------------------------ |
     | total_timesteps | int                   | 训练样本的总数                                               |
     | seed            | int                   | 训练的初始种子，如果是None：保持当前种子                     |
     | callback        | function (dict, dict) | 算法状态的每一步都调用的布尔函数。它接受本地或全局变量。如果返回False，终止训练 |
     | log_interval    | int                   | 记录日志之前的时间步数                                       |
     | tb_log_name     | str                   | 运行tensorboard日志的名字                                    |

     ***返回：***（*BaseRLModel*）训练好的模型

  6.  ***类方法 load()***  

     ```python
     classmethod load(load_path, env=None, **kwargs)  
     ```

     从文件中载入模型

     | 参数      | 类型             | 意义                                                         |
     | --------- | ---------------- | ------------------------------------------------------------ |
     | load_path | str or file-like | 参数保存位置                                                 |
     | env       | Gym Envrionment  | 载入模型运行的环境（如果你只是需要从一个已训练模型进行预测可以是是None） |
     | kwargs    |                  | 载入模型时对模型有改变作用的其他参数                         |

  7. ***load_parameters()***  

     ```python
     load_parameters(load_path_or_dict, exact_match=True) 
     ```

     从文件或字典中载入模型参数

     字典关键字应该时*tensorflow*变量名称，可以用`get_parameters`函数获取。如果`exact_match`为*True*，字典应该包含所有模型参数的关键字。否则，出现*RunTimeError*。如果时*False*，只有字典内的参数会被更新。

     此函数没有载入*agent*的超参数。

     > 警告：
     >
     > 此函数没有更新训练器/优化器参数（例如动量）。因为使用此函数后的这种训练会导致不太理想的结果。

     | 参数              | 类型                     | 意义                                                         |
     | ----------------- | ------------------------ | ------------------------------------------------------------ |
     | load_path_or_dict | str or file-like or dict | 参数保存位置或参数字典                                       |
     | exact_match       | bool                     | 如果是True，期望载入字典包含此模型的所有参数。如果是False，只为字典提到的变量载入参数，默认True |

  8. ***predict()*** 

     ```python
     predict(observation, state=None, mask=None, deterministic=False) 
     ```

     从一个观测得到模型的行动

     | 参数          | 类型       | 意义                                 |
     | ------------- | ---------- | ------------------------------------ |
     | observation   | np.ndarray | 输入观测                             |
     | state         | np.ndarray | 最新状态（可以是None，用于迭代策略） |
     | mask          | np.ndarray | 最新掩码（可以是None，用于迭代策略） |
     | deterministic | bool       | 是否返回确定性的行动                 |

     ***返回：***（*np.ndarray, np.ndarray*）模型的行动和下个状态（用在迭代策略）

  9. ***pretrain()***  

     ```python
     pretrain(dataset, n_epochs=10, learning_rate=0.0001, adam_epsilon=1e-08, val_interval=None) 
     ```

     用行为克隆预训练一个模型：在给定专家数据集监督学习

     目前只支持*Box*和离散空间。

     | 参数          | 类型          | 意义                                                  |
     | ------------- | ------------- | ----------------------------------------------------- |
     | dataset       | ExpertDataset | 数据集管理器                                          |
     | n_epochs      | int           | 在训练集上的迭代次数                                  |
     | learning_rate | float         | 学习率                                                |
     | adam_epsilon  | float         | 是*adam*优化器的$\epsilon$值                          |
     | val_interval  | int           | 报告每*n*轮训练和验证的损失。默认，最大代数的十分之一 |

     ***返回：***（*BaseRLModel*）预训练模型

  10. ***save(save_path)*** 

      将当前参数保存到文件

      | 参数      | 类型                    | 意义     |
      | --------- | ----------------------- | -------- |
      | save_path | str or file-like object | 保存位置 |

  11. ***set_env(env)***  

      验证环境的有效性，如果是连贯的，设为当前环境。

      | 参数 | 类型            | 意义           |
      | ---- | --------------- | -------------- |
      | env  | Gym Environment | 学习策略的环境 |

  12. ***setup_model()***  

      创建训练模型所需的所有函数和*tensorflow*图表