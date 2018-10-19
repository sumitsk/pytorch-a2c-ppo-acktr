# Rllab Ant Locomotion without Drift

This repository is used to train rllab's Ant agent to move in a straight line along any of the four directions. The default reward structure encourages the agent to move forward at its maximum speed, however, it leads to sideways drift in agent's movement. This behavior can be noticed easily in almost any of the videos of ant locomotion trained in either OpenAI gym or rllab. This is a problem when the agent is required to navigate areas with narrow corridors, for example, maze environments in rllab as the agent will keep on hitting the walls and will instead get stuck or drag along it. We reduced this drift by modifying the reward function and adding a drift-dependent penalty term. Furthermore, we reduced the gear settings of the agent to reduce its maximum speed. 

The policy gradient algorithms used here are taken from the repository of [Ilya Kostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr).

## Installation

### Dependencies:
* rllab - [Forked repository](https://github.com/sumitsk/rllab) 
* OpenAI gym 0.7.4
* mujoco-py 1.50
* Pytorch 0.4
* TensorboardX 

We use the above-listed versions, however, the latest versions may also be compatible.

## Results
### Trained models
The final trained policies are present in the `policies` directory. The corresponding learned behaviors are:
![East](videos/posx.gif)
![West](videos/negx.gif)
![North](videos/posy.gif)
![South](videos/negy.gif)

Note that the agent moves in a straight line and deviates very little in all four cases.

## The Team
* Sumit Kumar
* Vigneshram Krishnamoorthy
* Suriya Narayanan Lakshmanan

All the members are Graduate Student at Carnegie Mellon University.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


## Contact
For any queries, feel free to post an issue or contact at sumitsk@cmu.edu.