import gymnasium as gym

# env = gym.make('CartPole-v0')
env = gym.make('Acrobot-v1', render_mode="human")
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
