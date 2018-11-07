
import tensorflow as tf
import numpy as np
import gym

'''
Because of Lacking of Fundatmental,Reading in furture days
'''

'''
gym is a openSource Tool of OpenAI
the notion of Reforcement Learning in gym consist of
three part: 1,Environment  2,Action  3,Reward
'''

#----------------------------------------------------------------------------
'''
#这个例子只作为参考使用，没什么算法价值

env = gym.make(id='CartPole-v0')
env.reset()

random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    # env.render() 将环境的矩阵渲染出来，如果环境矩阵是图像
    env.render()
    # env.step() 相当于执行一次 Action
    observation,reward,done,_ = env.step(np.random.randint(0,2))
    reward_sum += reward

    if done:
        random_episodes += 1
        print('Reward for this episode was: ',reward_sum)
        reward_sum = 0
        env.reset()
'''
#----------------------------------------------------------------------------------
'''
信息不够，理解不充分
'''
env = gym.make(id='CartPole-v0')

# H 应该是 隐层的神经元数目
H = 50
batch_size = 25
learning_rate = 0.1
# 这是环境信息返回的维度，即是 observation 的维度
D = 4
# gamma 是对于不同时间节点的 reward 的递减系数，相当于reward_sum = Sum( Reward_i * gamma**i )
gamma = 0.99
# tf.trainable_variables() 返回所有当前计算图中 在获取变量时未标记 trainable=False 的变量集合
# 即是返回的是需要训练的变量列表
tvars = tf.trainable_variables()

observations = tf.placeholder(dtype=tf.float32,shape=[None,D],name='input_x')
w1 = tf.get_variable(name='w1',shape=[D,H])
layer1 = tf.nn.relu(tf.matmul(observations,w1))
w2 = tf.get_variable(name='w2',shape=[H,1])
score = tf.matmul(layer1,w2)
probability = tf.nn.sigmoid(score)

adam = tf.train.AdamOptimizer(learning_rate)
w1Grad = tf.placeholder(dtype=tf.float32,name='batch_grad1')
w2Grad = tf.placeholder(dtype=tf.float32,name='batch_grad2')
batchGrad = [w1Grad,w2Grad]

# apply_gradients((gradient, variable)) 使用(gradient, variable)列表
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

def discount_rewards(r):
    # np.zero_like: Return an array of zeros with the same shape and type as a given array
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[r] = running_add
    return discounted_r

input_y = tf.placeholder(dtype=tf.float32,shape=[None,1],name='inpuy_y')
advantages = tf.placeholder(dtype=tf.float32,name='reward_signal')
loglik = tf.log(input_y*(input_y - probability) + (1-input_y)*(input_y+probability) )
loss = -tf.reduce_mean(loglik * advantages)

newGrads = tf.gradients(ys=loss,xs=tvars)

xs,ys,drs = [],[],[]
reward_sum = 0
episode_number = 1
total_episodes = 10000

with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)

    observation = env.reset()

    # 这里的 gradBuffer 是作为作为 grad 的缓冲区，开始初始化为0
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:
        if reward_sum/batch_size >100 or rendering == True:
            env.render()
            rendering = True

        # 这里 x 是每个输入的结果
        x = np.reshape(observation,[1,D])

        tfprob = sess.run(fetches=probability,
                          feed_dict={
                              observations:x
                          })
        action = 1 if np.random.uniform() < tfprob else 0

        # xs 整合 每个输入结果
        xs.append(x)
        y = 1-action
        ys.append(y)

        # env.step() 对于我而言就是个 黑箱
        observation,reward,done,info = env.step(action)
        reward_sum += reward
        drs.append(reward)

        if done :
            episode_number +=1
            # 这里先不考虑 np.vstack 的作用，它是用于维度拼接与转换的
            # 这里的 vstack 相当于将 每所有输入整合为 [ None , D]
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs,ys,drs = [],[],[]

            discount_epr = discount_rewards(epr)
            discount_epr -= np.mean(discount_epr)
            discount_epr /= np.std(discount_epr)

            tGrad = sess.run(fetches=newGrads,
                             feed_dict={
                                 observations:epx,
                                 input_y:epy,
                                 advantages:discount_epr
                             })
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if episode_number % batch_size == 0:
                sess.run(fetches=updateGrads,
                         feed_dict={
                             # gradBuffer = [<w1-grad> , <w2-grad>]
                             w1Grad:gradBuffer[0],
                             w2Grad:gradBuffer[1]
                         })

                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                print('Average reward for episode %d : %f ' % (episode_number,reward_sum/batch_size))

                if reward_sum/batch_size > 200:
                    print('Task solve in ',episode_number,' times')
                    break

                reward_sum = 0
            env.reset()
















