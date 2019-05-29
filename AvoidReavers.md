## Avoid An Obstacle : Avoid-Reavers 
 
The purpose of the scenario is to safely move to the destination while avoiding some moving obstacles. 
The starting point is the upper left of the map and the destination is the lower right.  
In this scenario, the agent is Terran dropship, and the obstacles are 3 Protoss reavers.
Dropship is a air-transport unit of terran race and it has relatively low speed and acceleration. 
And the obstacle, protoss reaver, which is one of the slowest units in the game, moves randomly in this scenario so that it makes more difficult to predict their movement.   
Also, since the size of the map is very limited (320 X 320 Pixels) and the sizes of the units are not small (Dropship : 49 X 37 Pixel, River : 32 X 32 Pixel) the agents must move carefully for each step. 
Lastly, the starting point of the agent is the area where the protoss river can not get close, so it is also important to use it properly

<p style="text-align:center;">
    <img style="max-width:49%; height:14em" src="/SAIDA_RL/assets/image/Avoid_reavers_explained.png" alt="Avoid_Reaver.gif">
    <img style="max-width:49%; height:14em" src="/SAIDA_RL/assets/image/Avoid_Reaver.gif" alt="Avoid_reavers_explained.png">
</p>

## Action Space 

The dropship(agent) must decide how far they will move and which direction to move for each step.
The agent does not budge when a movement command is given to outside area of the map(Dark area in the picture). 
Therefore, it is very important to determine the appropriate move distance.
If the move distance is too small, it may take a long time to reach the goal, and if the move distance is too large, the attempt to move to the outside area of the map may increase.

 Action Type  | Type | Output | Description |
---- | ---- | ---- | ---- |
0 | Discrete | one integer between 0 ~ action size | Output number means the moving direction. 0 means 3 o'clock. The total number of action spaces is defined as 360 / (move dist). The last number of action means stop moving |
1 | Continuous | x, y, action number | ● X, Y : 2D coordinates, <br/> ● the action number 0 : move, 1 : attack move |
2 | Continuous | theta, radian, action number | ● theta : the angle started from 3 0'clock  <br/>● radian : distance to move toward the theta, <br/>● action number 0 : move, 1 : attack move |

## Observation 

Important observations in the avoidance scenario are the location and acceleration of each unit.
Especially, since the terran dropship and protoss river has a slow speed so that they have a some disadvantage to change the direction suddenly, 
knowing these information and predicting the next state is crucial to overcome this problem. 

- Base Information of unit type

Observation name  | Description | Type|
 ---- | ---- | ---- | 
acceleration | The unit's accelation amount | int32 |
armor |The amount of armor the unit type has |int32 
cooldown_max |The amount of base cooldown applied to the unit after an attack |int32 
damage_amount |The base amount of damage that this weapon can deal per attack |int32 
damage_factor |The intended number of missiles/attacks that are used. This is used to multiply with the damage amount to obtain the full amount of damage for an attack. |int32 
energy_max |The maximum amount of energy for this unit type| int32 
hp_max |The maximum amount of hit points for this unit type| int32 
seek_range |The range at which this unit type will start targeting enemy units |int32 
shield_max |The maximum amount of shield points for this unit type| int32 
sight_range |The sight range of this unit type |int32 
top_speed| The unit type's top movement speed with no upgrades |double 
weapon_range |The maximum attack range of the weapon, measured in pixels |int32 

- Current Informaiton of all units in game

Observation name  | Description | Type |
 ---- | ---- | ---- | 
unit_type |The type of unit |string 
hp |The current amount of hit points for this unit |int32 
shield |The current amount of shield points for this unit| int32 
energy |The current amount of energy points for this unit |int32 
cooldown |The current amount of cooldown for this unit |int32 
pos_x |The X coordinates for current position of this unit. The position is   roughly the center if the unit |int32 
pos_y |The Y coordinates for current position of this unit. The position is   roughly the center if the unit |int32 
velocity_x |The X component of the unit's velocity, measured in pixels per frame |double 
velocity_y |The Y component of the unit's velocity, measured in pixels per frame| double 
angle |The unit's facing direction in radians |double 
accelerating | True, if the current unit is accelerating. |bool 
braking | True, if the current unit is slowing down to come to a stop |bool 
attacking |True, if this unit is currently attacking something |bool 
is_attack_frame |True, if this unit is currently playing an attack animation |bool 

## Reward
 
In the case of this scenario, the agent can not defeated.
However, when the agent dropship collides with protoss river or when it is ordered to move to outside area of the map it receives negative reward.
So, the reward you can get from the scenario is simple.
To improve the learning speed or performance, it is essential for the user to properly reshape the reward using the observation information.
Below is the reward delivered in the scenario.

- +1 : when agent reaches the goal.
- -1 : when agent collides with enemy
- -0.1 : when it ordered to move to outside area of the map

## Environment configuration

You should know those parameters when you create the environment.

```python
from saida_gym.starcraft.avoidReavers import AvoidReavers
env = AvoidReavers(version=0, action_type=0, frames_per_step=24, move_angle=30, move_dist=2, verbose=0)
```

{% include_relative Environment.md %}