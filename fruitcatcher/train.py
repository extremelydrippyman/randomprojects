import numpy as np
import random
from fruit_catcher_env import FruitCatacherEnv
from q_model import build_model

env = FruitCatacherEnv()
model = build_model()

#Hyperparameters; variables set before training to influence how the model acts
gamma = 0.95 #How much future rewards are worth
epsilon = 1.0 #Initial exploration rate (fully random at first with a 1.0)
epsilon_min = 0.1 #Minimum exploration
epsilon_decay = 0.985 #How fast exploration decays
episodes = 100 #Number of training games
batch_size = 32 #Number of samples for each training step
memory = [] #Store past experiences

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(env.height):
        if np.random.rand() < epsilon:
            action = np.random.randint(3) #Random action, explore
        else:
            #Predict Q-values (best choice) for current state (basket pos and fruit pos)
            q_values = model.predict(np.expand_dims(state, axis=0), verbose = 0)
            #verbose=0 hides training output, 1 shows a progress bar, 2 prints one line per episode
            action = np.argmax(q_values[0]) #Choose the best predicted action

        next_state, reward, done = env.step(action)
        memory.append((state,action,reward,next_state,done))
        state = next_state
        total_reward += reward

        if done:
            break

    if len(memory) >= batch_size:
        batch = random.sample(memory, batch_size) # take a random sample when length is full
        for s, a, r, ns, d in batch: #state, action, reward, next state, done
            target = r
            if not d:
                target = r + gamma * np.max(model.predict(np.expand_dims(ns, axis=0), verbose=0)) #pick the best choice, use bellman equation
            target_f = model.predict(np.expand_dims(s, axis=0), verbose=0)
            target_f[0][a] = target
            model.fit(np.expand_dims(s, axis=0), target_f, epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode+1} | Total Reward: {total_reward} | Epsilon: {epsilon:.2f}")
    if (episode + 1) % 25 == 0:
        model.save_weights(f"checkpoint_ep{episode+1}.weights.h5")
        print(f"💾 Checkpoint saved for episode {episode+1}")


model.save_weights("fruitcatcher/model_weights.weights.h5")