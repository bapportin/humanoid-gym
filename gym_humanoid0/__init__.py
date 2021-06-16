from gym.envs.registration import register

register(
    id='HumanoidContinuous-v0',
    entry_point='gym_humanoid0.envs:HumanoidContinuousEnv'
)

register(
    id='HumanoidDiscrete-v0',
    entry_point='gym_humanoid0.envs:HumanoidDiscreteEnv'
)
