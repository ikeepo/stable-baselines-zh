- # Stable Baselines/用户向导/处理NaNs和infs

  > *Stable Baselines*官方文档中文版 [Github](https://github.com/DBWangML/stable-baselines-zh) [CSDN](https://blog.csdn.net/The_Time_Runner/article/details/97392656)
  > 尝试翻译官方文档，水平有限，如有错误万望指正

  在指定环境下训练模型的过程中，当遇到输入或者从RL模型中返回的*NaN*或*inf*时，*RL*模型有完全崩溃的可能。

- ## 原因和方式

  问题出现后，*NaNs*和*infs*不会崩溃，而是简单的通过训练传递，直到所有的浮点数收敛到*NaN*或*inf*。这符合IEEE浮点运算标准(*IEEE754*)，标准指出：

  > 可能出现的物种异常：
  >
  > - 无效的操作符（$\sqrt{-1}$, inf$*$1, NaN mod 1, ...）返回NaN
  > - 除以0：
  >   - 如果运算对象非零（1/0， -2/0, ...）返回 $\pm inf$ 
  >   - 如果运算对象是零（0/0）返回NaN 
  > - 上溢（指数太高而无法表示）返回$\pm inf$ 
  > - 下溢（指数太低而无法表示）返回$0$ 
  > - 不精确（以2为底时不能准确表示，例如1/5）返回四舍五入值（例如：`assert (1/5) * 3 == 0.6000000000000001`）

  只有*除以0*会报错，其他方式只会静静传递。

  在*Python*中，除以0会报如下错：`ZeroDivisionError: float division by zero`，其他会忽略。

  *Numpy*中默认警告：`RuntimeWarning: invalid value encountered`但不会停止代码。

  最差的情况，*Tensorflow*不会提示任何信息

  ```python
  import tensorflow as tf
  import numpy as np
  
  print("tensorflow test:")
  
  a = tf.constant(1.0)
  b = tf.constant(0.0)
  c = a / b
  
  sess = tf.Session()
  val = sess.run(c)  # this will be quiet
  print(val)
  sess.close()
  
  print("\r\nnumpy test:")
  
  a = np.float64(1.0)
  b = np.float64(0.0)
  val = a / b  # this will warn
  print(val)
  
  print("\r\npure python test:")
  
  a = 1.0
  b = 0.0
  val = a / b  # this will raise an exception and halt.
  print(val)
  ```

  不幸的是，大多数浮点运算都是用*Tensorflow*和*Numpy*处理的，这意味着当无效值出现时，你很可能得不到任何警告。

- ## Numpy参数

  *Numpy*有方便处理无效值的方法：`numpy.seterr`，它是为*Python*进程定义的，决定它如何处理浮点型错误。

  ```python
  import numpy as np
  np.seterr(all='raise')  # define before your code.
  print("numpy test:")
  a = np.float64(1.0)
  b = np.float64(0.0)
  val = a / b  # this will now raise an exception instead of a warning.
  print(val)
  ```

  不过这也会避免浮点数的溢出问题：

  ```python
  import numpy as np
  
  np.seterr(all='raise')  # define before your code.
  
  print("numpy overflow test:")
  
  a = np.float64(10)
  b = np.float64(1000)
  val = a ** b  # this will now raise an exception
  print(val)
  ```

  不过无法避免传递问题：

  ```python
  import numpy as np
  
  np.seterr(all='raise')  # define before your code.
  
  print("numpy propagation test:")
  
  a = np.float64('NaN')
  b = np.float64(1.0)
  val = a + b  # this will neither warn nor raise anything
  print(val)
  ```

- ## Tensorflow参数

  *Tensorflow*会增加检查以侦测和处理无效值：`tf.add_check_numerics_ops`和 `tf.check_numerics`，然而，他们会增加*Tensorflow*图表处理，增加运算时间。

  ```python
  import tensorflow as tf
  
  print("tensorflow test:")
  
  a = tf.constant(1.0)
  b = tf.constant(0.0)
  c = a / b
  
  check_nan = tf.add_check_numerics_ops()  # add after your graph definition.
  
  sess = tf.Session()
  val, _ = sess.run([c, check_nan])  # this will now raise an exception
  print(val)
  sess.close()
  ```

  这也会避免浮点数溢出问题：

  ```python
  import tensorflow as tf
  
  print("tensorflow overflow test:")
  
  check_nan = []  # the list of check_numerics operations
  
  a = tf.constant(10)
  b = tf.constant(1000)
  c = a ** b
  
  check_nan.append(tf.check_numerics(c, ""))  # check the 'c' operations
  
  sess = tf.Session()
  val, _ = sess.run([c] + check_nan)  # this will now raise an exception
  print(val)
  sess.close()
  ```

  捕捉传播问题：

  ```python
  import tensorflow as tf
  
  print("tensorflow propagation test:")
  
  check_nan = []  # the list of check_numerics operations
  
  a = tf.constant('NaN')
  b = tf.constant(1.0)
  c = a + b
  
  check_nan.append(tf.check_numerics(c, ""))  # check the 'c' operations
  
  sess = tf.Session()
  val, _ = sess.run([c] + check_nan)  # this will now raise an exception
  print(val)
  sess.close()
  ```

- ## VecChecNan包装器

  为查明无效值源自何时何处，*stable-baselines*提出`VecChecknan`包装器。

  它会监控行动、观测、奖励，指明那种行动和观测导致了无效值以及从何处出现。

  ```python
  import gym
  from gym import spaces
  import numpy as np
  
  from stable_baselines import PPO2
  from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
  
  class NanAndInfEnv(gym.Env):
      """Custom Environment that raised NaNs and Infs"""
      metadata = {'render.modes': ['human']}
  
      def __init__(self):
          super(NanAndInfEnv, self).__init__()
          self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
          self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
  
      def step(self, _action):
          randf = np.random.rand()
          if randf > 0.99:
              obs = float('NaN')
          elif randf > 0.98:
              obs = float('inf')
          else:
              obs = randf
          return [obs], 0.0, False, {}
  
      def reset(self):
          return [0.0]
  
      def render(self, mode='human', close=False):
          pass
  
  # Create environment
  env = DummyVecEnv([lambda: NanAndInfEnv()])
  env = VecCheckNan(env, raise_exception=True)
  
  # Instantiate the agent
  model = PPO2('MlpPolicy', env)
  
  # Train the agent
  model.learn(total_timesteps=int(2e5))  # this will crash explaining that the invalid value originated from the environment.
  ```

- ## RL模型超参数

  依据你的超参数，NaN可能更经常出现。一个极好的例子：<https://github.com/hill-a/stable-baselines/issues/340>   

  要明白，虽然在大多数案例中默认超参数看起来可以跑通，不过在你的环境下很难最优。如果是这样，搞清楚每个超参数如何影响模型，以便你可以调参以得到稳定模型。或者，你可以尝试自动调参（参见[RL Zoo](https://blog.csdn.net/The_Time_Runner/article/details/98597248)）。

- ## 数据集中的缺失值

  如果你的环境产生自外部数据集，确保数据集中不含*NaNs*。因为有时候数据集中会用*NaNs*代替缺失值。

  这里有一些关于如何查找NaNs的阅读材料：[点击链接](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)。

  以及用其他方式查找缺失值：[点击链接](https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4)。

  

