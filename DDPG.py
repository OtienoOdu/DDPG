import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Ornstein-Uhlenbeck Noise for exploration
class OUNoise:
    def __init__(self, action_dim, mean=0, std_dev=0.2, theta=0.15, dt=1e-2):
        self.action_dim = action_dim
        self.mean = mean
        self.std_dev = std_dev
        self.theta = theta
        self.dt = dt
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mean

    def sample(self):
        x = self.state
        dx = self.theta * (self.mean - x) * self.dt + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state = x + dx
        return self.state


# Actor Model
def create_actor(state_dim, action_dim, action_bound):
    inputs = layers.Input(shape=(state_dim,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(action_dim, activation="tanh")(out)
    outputs = outputs * action_bound
    return tf.keras.Model(inputs, outputs)

# Critic Model
def create_critic(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    concat = layers.Concatenate()([state_input, action_input])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    return tf.keras.Model([state_input, action_input], outputs)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        # Initialize buffers with correct shapes
        self.state_buffer = np.zeros((self.buffer_capacity, state_dim))
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim))

    def store(self, state, action, reward, next_state):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state

        self.buffer_counter += 1

    def sample(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert NumPy arrays to TensorFlow tensors
        states = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)

        return states, actions, rewards, next_states


# Update Target Network
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


# Main Training Loop
def train_ddpg(env_name, episodes=100, noise_std_dev=0.2, action_bound=1, tau=0.005, gamma=0.99):
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]

    # Handle action space type
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n  # Number of discrete actions
        is_discrete = True
    else:
        action_dim = env.action_space.shape[0]
        is_discrete = False

    # Create actor and critic
    if not is_discrete:
        actor = create_actor(state_dim, action_dim, action_bound)
        critic = create_critic(state_dim, action_dim)
        target_actor = create_actor(state_dim, action_dim, action_bound)
        target_critic = create_critic(state_dim, action_dim)

        # Copy weights
        target_actor.set_weights(actor.get_weights())
        target_critic.set_weights(critic.get_weights())

    # Replay buffer and noise
    buffer = ReplayBuffer(state_dim, action_dim)
    ou_noise = OUNoise(action_dim, std_dev=noise_std_dev) if not is_discrete else None

    # Optimizers
    if not is_discrete:
        actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    for ep in range(episodes):
        state, _ = env.reset()
        if ou_noise:
            ou_noise.reset()
        episodic_reward = 0

        while True:
            # Select action
            if is_discrete:
                action = env.action_space.sample()  # Random action for discrete spaces
            else:
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                action = actor(state_tensor)
                noise = ou_noise.sample()
                action = tf.clip_by_value(action + noise, -action_bound, action_bound)

            # Perform action
            next_state, reward, terminated, truncated, _ = env.step(action if is_discrete else action[0])
            done = terminated or truncated
            buffer.store(state, action, reward, next_state)

            # Train only if the environment is continuous and the buffer is ready
            if not is_discrete and buffer.buffer_counter > buffer.batch_size:
                states, actions, rewards, next_states = buffer.sample()

                # Update Critic
                with tf.GradientTape() as tape:
                    target_actions = target_actor(next_states)
                    y = rewards + gamma * target_critic([next_states, target_actions])
                    critic_value = critic([states, actions])
                    critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                grads = tape.gradient(critic_loss, critic.trainable_variables)
                critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

                # Update Actor
                with tf.GradientTape() as tape:
                    actions = actor(states)
                    critic_value = critic([states, actions])
                    actor_loss = -tf.math.reduce_mean(critic_value)
                grads = tape.gradient(actor_loss, actor.trainable_variables)
                actor_optimizer.apply_gradients(zip(grads, actor.trainable_variables))

                # Update Target Networks
                update_target(target_actor.variables, actor.variables, tau)
                update_target(target_critic.variables, critic.variables, tau)

            state = next_state
            episodic_reward += reward

            if done:
                print(f"Episode {ep + 1}: Reward = {episodic_reward}")
                break
    env.close()


# Training Examples
if __name__ == "__main__":
    print("Training on Pendulum-v1...")
    train_ddpg(env_name="Pendulum-v1", episodes=50)

    print("Training on Acrobot-v1...")
    train_ddpg(env_name="Acrobot-v1", episodes=50)

    print("Training on CartPole-v1...")
    train_ddpg(env_name="CartPole-v1", episodes=50)
