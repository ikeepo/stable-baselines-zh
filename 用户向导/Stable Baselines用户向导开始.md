- # Stable Baselines/用户向导/开始

- > Stable Baselines官方文档中文版 [Github](https://github.com/DBWangML/stable-baselines-zh)   [CSDN](https://blog.csdn.net/The_Time_Runner/article/details/97392656)   

  大多数强化学习算法包都试图采用sklearn风格语法。

  下面是一个简单的案例，展示如何在Cartpole环境中训练和运行PPO2.

  ```python
  import gym
  
  from stable_baselines.common.policies import MlpPolicy
  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines import PPO2
  
  env = gym.make('CartPole-v1')
  env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
  
  model = PPO2(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=10000)
  
  obs = env.reset()
  for i in range(1000):
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()
  ```

  或者，如果环境已在Gym注册、策略也已注册，仅仅用liner训练一个模型：

  ```python
  # 用一行代码定义并训练一个RL agent
  from stable_baselines import PPO2
  model = PPO2('MlpPolicy', 'CartPole-v1').learn(10000)
  ```

  ![](D:\Wangdb\Typora\Typora图片\RL agent.gif)

