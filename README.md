# Sometimes deep, Sometimes learning
A collection of DL experiments and notes
原始工程地址：https://github.com/JannesKlaas/sometimes_deep_sometimes_learning/blob/master/reinforcement_pytorch.ipynb

### own:
参考博客： [https://www.zhihu.com/question/26408259][如何用简单例子讲解 Q - learning 的具体过程？]
- Q-learning形象解释：主要包括状态(state)、动作(action)、奖励(reward)三要素
Q其相当于一张表格，其记录着各种状态下的采取的action。这里用来存储表格的是使用深度网络来存储。
根据不同state输出对应的action分数值，取最大的值max值作为action。

- 大概思路：输入是10*10的整幅图像reshape成1*100的数据，其中1指的是fruit跟basket的位置。
输出：三个动作，左，静止，右
训练思路：其是先根据当前状态state_t，通过model输出一个max的action,然后根据这个action产生新的
state_tp1并获取当前action产生的reward。并把这些存到memory，这里大概存了500个，每盘游戏10步结束，所以这里的memory
大概存了50盘游戏。训练的时候从这里随机抽其对应batch_size的数据训练。
- 用于训练的inputs，targets数据产生方式，跟训练技巧：
把最低端的reward不断上传到上面，在游戏未结束时，inputs是先前的状态，targets是下一个新状态的输出，其使用了一个gamma比例系数，用来权衡历史跟现在的
重要性。
这种训练可以理解为，当前状态，如果采取了当前网络的action，后产生的新状态获取的reward会怎么样。网络训练的时候其步骤顺序肯定是从fruits在底部的时候该怎么走先学习到，然后不断的往上
走，获取上层该怎么走的技能。因为game over的时候其reward_t是明确跟准确的，所以刚开始最底下状态的时候是肯定可以学习到的，然后不断的往上学习。
例子解释：
当fruits在最底部行(倒数第一行)学习完该怎么走时，当其作为state_tp1时，则(倒数第二行)state_t能够获取的targets也变的准确，这样子model能学习网在state_t状态下该怎么走，
接着(倒数第三行)新的state_t，而(倒数第二行)新的state_tp1，进行model学习在state_t状态下该怎么走。
以此不断循环向上，则可以完成整个步骤的学习。
