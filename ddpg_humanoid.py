import gym
import gym_humanoid0
import numpy as np
import collections

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate, Reshape, Conv2D, Lambda
from tensorflow.keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor

class _coll:
    def __init__(self):
        self.data=collections.OrderedDict()
    def add(self,name,value):
        if not name in self.data:
            self.data[name]=[]
        self.data[name].append(value)
    def result(self):
        return list(map(np.array,self.data.values()))


class XProcessor(Processor):
    def process_state_batch(self, batch):
        c=_coll()
        if type(batch)==np.ndarray:
            for b in batch.flatten():
                if type(b)==collections.OrderedDict:
                    for k,v in b.items():
                        c.add(k,v)
                else:
                    print (type(b),b)
                    raise Exception()
                                    
        return c.result()


ENV_NAME="HumanoidContinuous-v0"

env=gym.make(ENV_NAME)

nb_actions=env.action_space.shape[0]

servos=Input(shape=env.observation_space["servos"].shape,name="servos")
sensors=Input(shape=env.observation_space["sensors"].shape,name="sensors")
eye_left=Input(shape=env.observation_space["eye_left"].shape,name="eye_left")
eye_right=Input(shape=env.observation_space["eye_right"].shape,name="eye_right")

view=Concatenate(axis=-1)([Reshape((100,100,1))(eye_left),Reshape((100,100,1))(eye_right)])
v0=view#Lambda(lambda x: (x-128.0)/128)(view)
v1=Conv2D(5,3,activation="tanh")(v0)
v2=Conv2D(7,3,activation="tanh")(v1)
v3=Conv2D(9,5,activation="tanh")(v2)
v4=Conv2D(11,7,activation="tanh")(v3)
vo=Flatten()(v4)

ser0=Dense(100,activation="tanh")(servos)
ser1=Dense(100,activation="tanh")(ser0)

sen0=Dense(100,activation="tanh")(sensors)
sen1=Dense(100,activation="tanh")(sen0)

center=Concatenate(axis=-1)([vo,ser1,sen1])

a0=Dense(nb_actions*2,activation="tanh")(center)
ao=Dense(nb_actions,activation="linear")(a0)
actor=Model([servos,sensors,eye_left,eye_right],ao)
print (actor.summary())

action_input=Input(shape=(nb_actions,),name="action_input")
ca=Concatenate(axis=-1)([center,action_input])
c0=Dense(nb_actions*2,activation="tanh")(ca)
co=Dense(1,activation="linear")(c0)
critic=Model([action_input,servos,sensors,eye_left,eye_right],co)
print (critic.summary())


memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.processor=XProcessor()
agent.compile(Adam(lr=.0001, clipnorm=1.), metrics=['mae'])

agent.load_weights('ddpg_{}_weights.h5f'.format(ENV_NAME))


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=100000, visualize=True, verbose=1, nb_max_episode_steps=1000)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
#agent.load_weights('ddpg_{}_weights.h5f'.format(ENV_NAME))

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)

