# DQN-RL-Atari
The Atari game chose Pong environment, which originated in 1972 in the United States, a game simulating two people playing table tennis.

1. The training of reinforcement learning has no ready-made samples, but the intelligence collects the corresponding (state, action, reward) samples in the interaction with the environment for trial-and-error learning, learning in the process of playing the game, and game control is a sequential decision problem, MDP modeling can be a good solution to this problem.


2. ![image-20230326211446406](/Users/weimaolin/Library/Application Support/typora-user-images/image-20230326211446406.png)

   The game watch is an image of 210 by 160, which is the data graph of the game screen.


3. ALE was built on Stella, an open source Atari 2600 emulator. It allowed users to interact with the Atari 2600 by receiving joystick actions, sending screen/RAM messages, and emulating the platform. ALE provides a game-handling layer that turns each game into a standard reinforcement learning problem by marking cumulative scores and whether the game is over. By default, each observation contains a single game screen: frame: a 2D array of 7bit pixels, 160 pixels wide x 210 pixels high. The action_space contains up to 18 discrete actions, which are defined by the joystick controller. The game-handling layer also specifies the minimum set of actions that need to be played for a particular game. When running, the simulator produces 60 frames per second, with a top speed of 6,000 frames per second. The reward on each time-step is defined on a game basis and is usually specified by the difference in score/points between frames. One episode starts at the first frame after the reset command and ends when the game ends. The game-handling layer also provides the ability to terminate an episode after a predefined number of frames. Users can thus access dozens of games through a single common interface, and can easily add new games.


4. The action space is six discrete Spaces, and what this means is six decision variables, like up, down, left, right.![image-20230326211719448](/Users/weimaolin/Library/Application Support/typora-user-images/image-20230326211719448.png)


5. ```python
   def process_image(self, image):
           frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
           frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  # Cut 20 px from top
           frame = cv2.resize(frame, (self.target_w, self.target_h))  # Resize
           frame = frame.reshape(self.target_w, self.target_h) / 255  # Normalize
   
           return frame
   ```
   
   The code that handles pre-proceesing, as shown above, first converts the image into a grayscale image -- because colors, which count as honor information, are not needed to make decisions -- and then clippings the image to a specified size.
   
   Finally, divide by 255 to shrink to 0-1, because the neural network is sensitive to 0-1 values.
   
6. ```python
   #CNN with Duel Algo. https://arxiv.org/abs/1511.06581
   class DuelCNN(nn.Module):
       """
       CNN with Duel Algo. https://arxiv.org/abs/1511.06581
       """
       def __init__(self, h, w, output_size):
           super(DuelCNN, self).__init__()
           self.conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)
           self.bn1 = nn.BatchNorm2d(32)
           convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
           self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
           self.bn2 = nn.BatchNorm2d(64)
           convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
           self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
           self.bn3 = nn.BatchNorm2d(64)
           convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)
   
           linear_input_size = convw * convh * 64  # Last conv layer's out sizes
   
           self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
           self.Alrelu = nn.LeakyReLU()  # Linear 1 activation funct
           self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)
   
           self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
           self.Vlrelu = nn.LeakyReLU()  # Linear 1 activation funct
           self.Vlinear2 = nn.Linear(in_features=128, out_features=1)  # Only 1 node
   ```

   This is a neural network structure consisting of four convolution layers and two linear layers.

7 . The actions of paddle on the left are determined by the input action parameters, which are predicted by DQN. Whereas the actions of paddle on the right are controlled by a simple computer ai.
   How to win: The first person to score 20 points wins.
   There is also the need to return to the current state of the game, including:
   reward C.
   Game frame;
   The current score;
   Whether the round is over;
   Current action action.
   Among them, the reward is designed by ourselves. Generally speaking, the reward must be positive when the score is scored, but it must be negative when the score is lost. How to design better depends on our own experience and debugging.


   8. ![image-20230326212122317](/Users/weimaolin/Library/Application Support/typora-user-images/image-20230326212122317.png)

   DQN uses the formula above to select updates
   Q Network optimization target = reward + gamma * maximum Q for the next state
   Change the network in the direction of maximum reward
   Calculate the Q values of all state-behavior pairs, and take the maximum Q value of state-behavior value to update the value function of each state, which I understand as the value of each state. In DQN, I think the corresponding one is to use the maximum Q value of the next state as a part of the target, and use the Q value of the state-behavior pair obtained by the estimated network to calculate loss with the target, so as to achieve the purpose of updating parameters. Updating parameters is equivalent to updating Q table, that is, updating Q value.
   The way to get the optimal strategy is to take the behavior that has the maximum value in each state in the Q function.
   target = bonus + gamma * Maximum Q for the next state
   The fit, essentially, fits the q value of this state-action pair to the reward value + gamma * the maximum Q value of the next state, i.e., You want to estimate that the value of the action the network gets at this state (q represents the value of the action) is close to the reward value + gamma * maximum Q of the next state. The bonus value + gamma * the maximum Q value of the next state means the maximum value of the next state (q value) from using this action in the current state + the immediate bonus from using this action in the current state, which is the cumulative future bonus of this s-a.

9. Because it's propagating through the Behrman equation, it accumulates. Even a very small error at each step can eventually lead to a very large error on a long sequence task. And one thing they need to know is that the error in this part comes from two parts, and each part is inevitable.
   1. Approximate error of neural network, as long as you use neural network, error is there.
   2. The estimation error of the Bermann equation, which exists as long as you update it in the form of the Beryan equation.
