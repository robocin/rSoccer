# IEE VSSS gym environments

Creating new environments is done over inheritance of the VSSS base class through the following steps:
+ Setting the field type (0 for the 3 vs 3 competition field size, and 1 for 5 vs 5), the number of blue and yellow robots and the simulation time step.
+ Defining the obsevation and action spaces as [gym.Spaces](https://gym.openai.com/docs/#spaces) types.
+ Implement the base class methods:
    + _get_commands
    + _frame_to_observations
    + _calculate_reward_and_done
    + _get_initial_positions_frame
+ Register the environment and set the maximum number of steps in an episode on [**__init__**](../__init__.py) 

The base environment define normalization methods using field size and robot parameters values which are used on the available environments.

# Available Envs
<!-- - **VSSFIRA-v0** [Needs to run with FIRASIm] -->
<!-- - **VSSMAOpp-v0** [Needs a attacker model trained on VSS-v0] -->
- [**VSS-v0**](#vss-v0)

# VSS-v0
In this environment each team has 3 robots, in which the id 0 blue robot is controlled through setting the desired wheel speed, while the other robots receive actions sampled from a Ornstein Uhlenbeck process. The episode ends when a goal occurs.

![VSS-v0 environment rendering gif](../../.github/resources/vss.gif)

- ## Observations:
    - Box(40,)
    - Value Range: [-1.25, 1.25] (Normalized)

    | Index        	| Observation                	|
    |--------------	|----------------------------	|
    | 0            	| Ball X                     	|
    | 1            	| Ball Y                     	|
    | 2            	| Ball Vx                    	|
    | 3            	| Ball Vy                    	|
    | 4 + (7 * i)  	| id i Blue Robot X          	|
    | 5 + (7 * i)  	| id i Blue Robot Y          	|
    | 6 + (7 * i)  	| id i Blue Robot sin(theta) 	|
    | 7 + (7 * i)  	| id i Blue Robot cos(theta) 	|
    | 8 + (7 * i)  	| id i Blue Robot Vx         	|
    | 9  + (7 * i) 	| id i Blue Robot Vy         	|
    | 10 + (7 * i) 	| id i Blue Robot v_theta    	|
    | 25 + (5 * i) 	| id i Yellow Robot X        	|
    | 26 + (5 * i) 	| id i Yellow Robot Y        	|
    | 27 + (5 * i) 	| id i Yellow Robot Vx       	|
    | 28 + (5 * i) 	| id i Yellow Robot Vy       	|
    | 29 + (5 * i) 	| id i Yellow Robot v_theta  	|

- ## Actions:
    - Box(2,)
    - Value Range: [-1, 1]

    | Index | Action        |
    |-------|---------------|
    | 0     | Wheel 0 speed |
    | 1     | Wheel 1 speed |

- ## Rewards:
    - Move
    - Ball potential gradient
    - Energy
    - Goal
- ## Done:
    When a goal happens
