- # Stable Baselines/用户向导/使用自定义环境

  > Stable Baselines官方文档中文版 [Github](https://github.com/DBWangML/stable-baselines-zh) [CSDN](https://blog.csdn.net/The_Time_Runner/article/details/97392656)
  > 尝试翻译官方文档，水平有限，如有错误万望指正

  在自定义环境使用*RL baselines*，只需要遵循*gym*接口即可。

  也就是说，你的环境必须实现下述方法（并且继承自*OpenAI Gym*类）：

  > 如果你用图像作为输入，输入值必须在[0,255]因为当用CNN策略时观测会被标准化（除以255让值落在[0,1]）

  ```python
  import gym
  from gym import spaces
  
  class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
  
    def __init__(self, arg1, arg2, ...):
      super(CustomEnv, self).__init__()
      # Define action and observation space
      # They must be gym.spaces objects
      # Example when using discrete actions:
      self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
      # Example for using image as input:
      self.observation_space = spaces.Box(low=0, high=255,
                                          shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
  
    def step(self, action):
      ...
    def reset(self):
      ...
    def render(self, mode='human', close=False):
      ...
  ```

  然后你就可以用其训练一个RL智体：

  ```python
  # Instantiate and wrap the env
  env = DummyVecEnv([lambda: CustomEnv(arg1, ...)])
  # Define and Train the agent
  model = A2C(CnnPolicy, env).learn(total_timesteps=1000)
  ```

  这里有一份创建自定义*Gym*环境的[在线教程](https://github.com/openai/gym/blob/master/docs/creating-environments.md)。

  视需求，你还可以像*gym*注册环境，这可让用户实现一行创建*Rl*智体（并用`gym.make()`实例化环境）。

  本项目中，为测试方便，我们在这个[文件夹](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/identity_env.py)下创建了名为`IdentityEnv`自定义环境。[这里](https://github.com/hill-a/stable-baselines/blob/master/tests/test_identity.py)有一个如何使用的案例展示。