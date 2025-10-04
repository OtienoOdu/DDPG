import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="human")

angle_thresh = 0    

for i_episode in range(2):
    observation = env.reset()
    for t in range(300):
        env.render()
        env.step(env.action_space.sample()) # take a random action
        
env.close()
