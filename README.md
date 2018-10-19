# Rllab Ant Locomotion with Low Drift

This repository is used to train rllab's Ant agent to move in a straight line along any of the four directions. The default reward structure encourages the agent to move forward at its maximum speed, however, it leads to sideways drift in agent's movement. This behavior can be noticed easily in almost any of the videos of ant locomotion trained in either OpenAI gym or rllab. This is a problem when the agent is required to navigate areas with narrow corridors, for example, maze environments in rllab as the agent will keep on hitting the walls and instead will drag along it. We reduced this drift by modifing the reward function and reducing the gear settings. 

The policy gradient algorithms used here are taken from the repository of [Ilya Kostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr).

## Installation

### Dependencies:
* rllab - [Forked version](https://github.com/sumitsk/rllab) 
* OpenAI gym 0.7.4
* mujoco-py 1.50
* Pytorch 0.4

We use the above-listed versions, however, the latest versions may also be compatible.

## Results
### Trained models
The final policies are present in `results` directory. The corresponding learned behaviors are:

![alt text](https://github.com/sumitsk/results/posx.gif "East")
![alt text](https://github.com/sumitsk/results/negx.gif "West")
![alt text](https://github.com/sumitsk/results/posy.gif "North")
![alt text](https://github.com/sumitsk/results/negy.gif "South")

Note that the agent moves in a straight line and deviates very little in all four cases.

## The Team
* Sumit Kumar
* Vigneshram Krishnamoorthy
* Suriya Narayanan Lakshmanan

## Contact
For any queries, feel free to post an issue or contact at sumitsk@cmu.edu.