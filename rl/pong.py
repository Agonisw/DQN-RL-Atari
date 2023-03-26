import gym
import cv2

import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from network import DuelCNN
from collections import deque

env_name = "PongDeterministic-v4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_models = False
Train = False
Load_model = True 
Load_file_ep = 90000
Batch_size = 64
mem_size = 40000
Gamma = 0.97
Alpha = 0.00025 
Epsilon_decay = 0.99 
Render = True  


class Agent:
    def __init__(self, environment):
        """
        Hyperparameters definition for Agent
        """
        self.state_size_h = environment.observation_space.shape[0]
        self.state_size_w = environment.observation_space.shape[1]
        self.state_size_c = environment.observation_space.shape[2]

        self.action_size = environment.action_space.n

        self.target_h = 80  
        self.target_w = 64  

        self.crop_dim = [20, self.state_size_h, 0, self.state_size_w] 

        self.gamma = Gamma  
        self.alpha = Alpha  

        self.epsilon = 1  
        self.epsilon_decay = Epsilon_decay 
        self.epsilon_minimum = 0.05 

        self.memory = deque(maxlen=50000)

        # Create two model for DDQN algorithm
        self.online_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(device)
        self.target_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(device)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.alpha)

    def process_image(self, image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
        frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  # Cut 20 px from top
        frame = cv2.resize(frame, (self.target_w, self.target_h))  # Resize
        frame = frame.reshape(self.target_w, self.target_h) / 255  # Normalize

        return frame

    def select_action(self, state):
        act_protocol = 'Explore' if random.uniform(0, 1) <= self.epsilon else 'Exploit'
        if act_protocol == 'Explore':
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
                q_values = self.online_model.forward(state)  # (1, action_size)
                action = torch.argmax(q_values).item()  # Returns the indices of the maximum value of all elements
        return action

    def train(self):
        if len(agent.memory) < mem_size:
            loss, max_q = [0, 0]
            return loss, max_q
        state, action, reward, next_state, done = zip(*random.sample(self.memory, Batch_size))

        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        # Convert them to tensors
        state = torch.tensor(state, dtype=torch.float, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=device)
        action = torch.tensor(action, dtype=torch.long, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        done = torch.tensor(done, dtype=torch.float, device=device)

        # Make predictions
        state_q_values = self.online_model(state)
        next_states_q_values = self.online_model(next_state)
        next_states_target_q_values = self.target_model(next_state)

        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        # Use Bellman function to find expected q value
        expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)

        # Calc loss with expected_q_value and q_value
        loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, torch.max(state_q_values).item()

    def storeResults(self, state, action, reward, nextState, done):
        """
        Store every result to memory
        """
        self.memory.append([state[None, :], action, reward, nextState[None, :], done])

    def adaptiveEpsilon(self):
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    environment = gym.make(env_name)  # Get env
    agent = Agent(environment)  # Create Agent

    if Load_model:
        agent.online_model.load_state_dict(torch.load("./models/pong-"+str(Load_file_ep)+".pkl"))

        with open("./models/pong-"+str(Load_file_ep)+'.json') as outfile:
            param = json.load(outfile)
            agent.epsilon = param.get('epsilon')

        startEpisode = Load_file_ep + 1

    else:
        startEpisode = 1

    last_100_ep_reward = deque(maxlen=100) 
    total_step = 1  
    for episode in range(startEpisode, 100000):

        startTime = time.time()  # Keep time
        state = environment.reset()  # Reset env

        state = agent.process_image(state)  # Process image

        state = np.stack((state, state, state, state))

        total_max_q_val = 0  # Total max q vals
        total_reward = 0  # Total reward for each episode
        total_loss = 0  # Total loss for each episode
        for step in range(100000):

            if Render:
                environment.render()  # Show state visually

            action = agent.select_action(state)  # Act
            next_state, reward, done, info = environment.step(action)  # Observe

            next_state = agent.process_image(next_state)  # Process image

            next_state = np.stack((next_state, state[0], state[1], state[2]))

            agent.storeResults(state, action, reward, next_state, done)  # Store to mem

            state = next_state 

            if Train:
             
                loss, max_q_val = agent.train()  
            else:
                loss, max_q_val = [0, 0]

            total_loss += loss
            total_max_q_val += max_q_val
            total_reward += reward
            total_step += 1
            if total_step % 1000 == 0:
                agent.adaptiveEpsilon()  # Decrase epsilon

            if done:  # Episode completed
                currentTime = time.time()  # Keep current time
                time_passed = currentTime - startTime  # Find episode duration
                current_time_format = time.strftime("%H:%M:%S", time.gmtime())  # Get current dateTime as HH:MM:SS
                epsilonDict = {'epsilon': agent.epsilon}  # Create epsilon dict to save model as file

                if save_models and episode % 10 == 0:  # Save model as file
                    weightsPath = "./models/pong-" + str(episode) + '.pkl'
                    epsilonPath = "./models/pong-" + str(episode) + '.json'

                    torch.save(agent.online_model.state_dict(), weightsPath)
                    with open(epsilonPath, 'w') as outfile:
                        json.dump(epsilonDict, outfile)

                if Train:
                    agent.target_model.load_state_dict(agent.online_model.state_dict())

                last_100_ep_reward.append(total_reward)
                avg_max_q_val = total_max_q_val / step

                outStr = "Episode:{} Time:{} Reward:{:.2f} Loss:{:.2f} Last_100_Avg_Rew:{:.3f} Avg_Max_Q:{:.3f} Epsilon:{:.2f} Duration:{:.2f} Step:{} CStep:{}".format(
                    episode, current_time_format, total_reward, total_loss, np.mean(last_100_ep_reward), avg_max_q_val, agent.epsilon, time_passed, step, total_step
                )

                print(outStr)

                if save_models:
                    outputPath = "./models/pong-" + "out" + '.txt'  # Save outStr to file
                    with open(outputPath, 'a') as outfile:
                        outfile.write(outStr+"\n")

                break
