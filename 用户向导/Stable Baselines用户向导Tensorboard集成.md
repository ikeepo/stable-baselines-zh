- # Stable Baselines/用户向导/Tensorboard集成

  > Stable Baselines官方文档中文版 [Github](https://github.com/DBWangML/stable-baselines-zh) [CSDN](https://blog.csdn.net/The_Time_Runner/article/details/97392656)
  > 尝试翻译官方文档，水平有限，如有错误万望指正

- ## 初阶用法

  与*RL baselines*一起使用*Tensorboard*，你只需为*RL*智体简单定义一个`log`位置即可：

  ```python
  import gym
  from stable_baselines import A2C
  model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
  model.learn(total_timesteps=10000)
  ```

  或者加载现存模型之后（日志路径默认未保存）：

  ```python
  import gym
  
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import A2C
  
  env = gym.make('CartPole-v1')
  env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
  
  model = A2C.load("./a2c_cartpole.pkl", env=env, tensorboard_log="./a2c_cartpole_tensorboard/")
  model.learn(total_timesteps=10000)
  ```

  学习函数一经调用，你可以在训练过程中或训练完成后，通过以下命令监视RL智体：

  ```python
  tensorboard --logdir ./a2c_cartpole_tensorboard/
  ```

  也可添加过去的日志文件夹：

  ```python
  tensorboard --logdir ./a2c_cartpole_tensorboard/;./ppo2_cartpole_tensorboard/
  ```

  它可展示诸多信息：模型图表、事件奖励、模型损失、观察、其他一些模型的特有参数。

  ![](D:\Wangdb\Typora\Typora图片\1.png)

  ![](D:\Wangdb\Typora\Typora图片\2.png)

  ![](D:\Wangdb\Typora\Typora图片\3.png)

- ## 日志更新

  使用回调函数，你用*TensorBoard*可轻松更新日志。这里是一个如何更新额外张量或随机标量的简单案例：

  ```python
  import tensorflow as tf
  import numpy as np
  
  from stable_baselines import SAC
  
  model = SAC("MlpPolicy", "Pendulum-v0", tensorboard_log="/tmp/sac/", verbose=1)
  # Define a new property to avoid global variable
  model.is_tb_set = False
  
  
  def callback(locals_, globals_):
      self_ = locals_['self']
      # Log additional tensor
      if not self_.is_tb_set:
          with self_.graph.as_default():
              tf.summary.scalar('value_target', tf.reduce_mean(self_.value_target))
              self_.summary = tf.summary.merge_all()
          self_.is_tb_set = True
      # Log scalar value (here a random variable)
      value = np.random.random()
      summary = tf.Summary(value=[tf.Summary.Value(tag='random_value', simple_value=value)])
      locals_['writer'].add_summary(summary, self_.num_timesteps)
      return True
  
  
  model.learn(50000, callback=callback)
  ```

- ## 主干集成

  终端展示的所有信息（默认日志）也可在*tensorboard*展示。为此，你需要定义几个环境变量：

  ```python
  # formats are comma-separated, but for tensorboard you only need the last one
  # stdout -> terminal
  export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard'
  export OPENAI_LOGDIR=path/to/tensorboard/data
  ```

  用下述命令配置日志程序：

  ```python
  from stable_baselines.logger import configure
  
  configure()
  ```

  然后启动*tensorboard*：

  ```python
  tensorboard --logdir=$OPENAI_LOGDIR
  ```

  