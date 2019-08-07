- # Stable Baselines/用户向导/预训练（行为克隆）

  > *Stable Baselines*官方文档中文版 [Github](https://github.com/DBWangML/stable-baselines-zh) [CSDN](https://blog.csdn.net/The_Time_Runner/article/details/97392656)
  > 尝试翻译官方文档，水平有限，如有错误万望指正

  通过.pretrain()方法，你可以源自专家的轨迹预训练RL策略，因此加速训练。

  行为克隆（BC）处理模仿学习的问题，例如用专家示范，作为一个监督学习问题。这就是说，给出的专家轨迹（观测-行动对），训练策略网络以再生专家行为：对于一个给定观测，策略必采用专家采取的行动。

  专家轨迹可以是人类展示，来自其他控制器的轨迹或来自已训练好的RL agents的轨迹。

  > 预训练模型目前只支持Box和Discrete空间

  > 为避免内存问题，处理图像数据集不同于其他数据集。来自专家衍射的图像必须来位于文件夹中，而不在专家Numpy归档文件中。

- ## 产生专家轨迹

  这里，我们会训练一个Rl模型，然后用这个agent产生专家轨迹。实际应用中，产生专家轨迹并不需要训练RL agent。

  下述案例只是为了展示`pretrain()`特征。

  然而，沃恩建议用户看一下`generate_expert_traj()`函数（位于*gail/dataset/*文件夹）的源代码，以了解专家数据集（查看下面的概述）的数据结构以及如何记录轨迹。

  ```python
  from stable_baselines import DQN
  from stable_baselines.gail import generate_expert_traj
  
  model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
        # Train a DQN agent for 1e5 timesteps and generate 10 trajectories
        # data will be saved in a numpy archive named `expert_cartpole.npz`
  generate_expert_traj(model, 'expert_cartpole', n_timesteps=int(1e5), n_episodes=10)
  ```

  这里有个调用专家控制器的额外例子，它被传递给函数而不是RL模型。想法是这个可调用的是一个PID控制器，询问人类玩家...

  ```python
  import gym
  
  from stable_baselines.gail import generate_expert_traj
  
  env = gym.make("CartPole-v1")
  # Here the expert is a random agent
  # but it can be any python function, e.g. a PID controller
  def dummy_expert(_obs):
      """
      Random agent. It samples actions randomly
      from the action space of the environment.
  
      :param _obs: (np.ndarray) Current observation
      :return: (np.ndarray) action taken by the expert
      """
      return env.action_space.sample()
  # Data will be saved in a numpy archive named `expert_cartpole.npz`
  # when using something different than an RL expert,
  # you must pass the environment object explicitely
  generate_expert_traj(dummy_expert, 'dummy_expert_cartpole', env, n_episodes=10)
  ```

- ## 用行为克隆预训练一个模型

  用通过前述脚本产生的`expert_cartpole.npz`数据集：

  ```python
  from stable_baselines import PPO2
  from stable_baselines.gail import ExpertDataset
  # Using only one expert trajectory
  # you can specify `traj_limitation=-1` for using the whole dataset
  dataset = ExpertDataset(expert_path='expert_cartpole.npz',
                          traj_limitation=1, batch_size=128)
  
  model = PPO2('MlpPolicy', 'CartPole-v1', verbose=1)
  # Pretrain the PPO2 model
  model.pretrain(dataset, n_epochs=1000)
  
  # As an option, you can train the RL agent
  # model.learn(int(1e5))
  
  # Test the pre-trained model
  env = model.get_env()
  obs = env.reset()
  
  reward_sum = 0.0
  for _ in range(1000):
          action, _ = model.predict(obs)
          obs, reward, done, _ = env.step(action)
          reward_sum += reward
          env.render()
          if done:
                  print(reward_sum)
                  reward_sum = 0.0
                  obs = env.reset()
  
  env.close()
  ```

- ## 专家数据集的数据结构

  专家数据集是一个`.npz`格式的归档文件。数据集保存为Python字典格式，带有关键字：`actions`, `episode_returns`, `rewards`, `obs`, `episode_starts`.

  在图像的案例中，`obs`包含到图像的相对路径。

  obs，actions：shape(N * L,) + S 

  其中，N = # episodes， L = episode 长度，S是环境观测/行动空间。 对于离散空间S = (1, )

- ```python
  class stable_baselines.gail.ExpertDataset(expert_path=None, traj_data=None, train_fraction=0.7, batch_size=64, traj_limitation=-1, randomize=True, verbose=1, sequential_preprocessing=False)
  ```

  用行为克隆或*GAIL*的数据集。

  专家数据集的结构是一个*dict*，保存为"`.npz`"档案文件。此字典包含关键字'`actions`', '`episode_returns`', '`rewards`', '`obs`', '`episode_starts`'。对应值具有跨事件链接的数据：第一轴是时间步，其余的轴索引到数据中。在图像案例中，'obs'包含到图像的相对路径，从图像压缩中节省空间。

  **参数**：

  - **expert_path** – (str) 到轨迹数据的路径 (.npz file). 与轨迹数据相互排斥.

  - **traj_data** – (dict) 轨迹数据, 以上面描述过的格式，与轨迹路径相互排斥。

  - **train_fraction** – (float) 训练的验证分割(0 to 1)用于通过行为克隆 (BC)来预训练

  - **batch_size** – (int) 行为克隆的最小批量尺寸

  - **traj_limitation** – (int) 采用的轨迹数量(如果是 -1, 全部载入)

  - **randomize** – (bool) 数据集是否需要被打乱

  - **verbose** – (int) 冗余信息

  - **sequential_preprocessing** – (bool) 是否采用子进程进行数据预处理 (对于CI虽然慢，但是节省内存)

  - ```python
    get_next_batch(split=None)
    ```

    从数据集获取批处理批次

    **参数：**split - (str)数据拆分的类型(可以是None, 'train', 'val')

    **返回：**(np.ndarray, np.ndarray)输入和标签

  - ```python
    init_dataloader(batch_size)
    ```

    初始化GAIL使用的数据载入器。

    **参数：**batch_size - (int) 

  - ```python
    log_info()
    ```

    记录数据集的信息

  - ```python
    plot()
    ```

    显示事件返回值的直方图

  - ```python
    prepare_pickling()
    ```

    退出程序以`pickle`数据集

- ```python
  class stable_baselines.gail.DataLoader(indices, observations, actions, batch_size, n_workers=1, infinite_loop=True, max_queue_len=1, shuffle=False, start_process=True, backend='threading', sequential=False, partial_minibatch=True)
  ```

  一种用于预处理观测值(dataloader)（包括图像）的自定义数据载入器，并将其传递给网络。

  dataloader的源代码<https://github.com/araffin/robotics-rl-srl> (MIT licence) 作者:Antonin Raffin, René Traoré, Ashley Hill

  **参数：**

  - **indices** – ([int]) 观测值索引的list 

  - **observations** – (np.ndarray) 观测值或者图像路径 

  - **actions** – (np.ndarray) 行动 

  - **batch_size** – (int) 最小批内的样本数 

  - **n_workers** – (int) 预处理器的数量(为载入图像)

  - **infinite_loop** – (bool) 是否有可重置的迭代器  

  - **max_queue_len** – (int) 可同时处理的最小批的最大数量 

  - **shuffle** – (bool) 每代之后重洗最小批 

  - **start_process** – (bool) 开始预处理程序(默认: True)

  - **backend** – (str) 工作库后端(在最新版本是‘`multiprocessing`’, ‘`sequential`’, ‘`threading`’ or ‘`loky`’ 其中之一)

  - **sequential** – (bool) 不用子进程预处理数据(对于*CI*虽然慢，但是节省内存)

  - **partial_minibatch** – (bool) 允许局部最小批(比`batch_size`样本更小的`minibatches`)

  - ```python
    sequential_next()
    ```

    预处理的顺序版本

  - ```python
    start_process()
    ```

    开始预处理程序

- ```python
  stable_baselines.gail.generate_expert_traj(model, save_path=None, env=None, n_timesteps=0, n_episodes=100, image_folder='recorded_images')
  ```

  训练专家控制器（如果需要）并记录专家轨迹。

  > 目前只支持*Box* 和*Discrete*空间

  **参数：**

  - **model** – (RL 模型或调用) 专家模型，如果需要被训练，你需要传入 `n_timesteps > 0`.
  - **save_path** – (str) 不带扩展的专家数据集存储路径 (ex: ‘expert_cartpole’ -> creates ‘expert_cartpole.npz’). 如果不指定则不会保存，只是返回生成的专家轨迹。基于图像的环境必须指定此参数.
  - **env** – (gym.Env) 环境，如果没有定义则会使用模型环境 
  - **n_timesteps** – (int) 训练时间步的次数
  - **n_episodes** – (int) 记录的轨迹步数
  - **image_folder** – (str) 当使用图形是，用于记录图像的文件夹。

  **返回：**

  (dict)生成的专家轨迹