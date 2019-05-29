## Survival : Escape from Observers

This scenario is similar to Avoid Reaver, but much more difficult.  
It is a scenario in which an agent (Zerg Scourge) that started at the bottom of the map should move to the top of the map to avoid more than 100 protoss observers randomly moving in the entire area of the map.
The episode ends when the agent collides with protoss observers or arrives at the top of the map.
This map is relatively difficult as following reasons.

- larger size of map. : The size of the map is 128 X 128 tiles, much larger than 10 x 10 of the river cover.
- more obstacles. : There are three obstacles to river cover, but this map should reach the top by avoiding more than 100 observers.
- terminate condition of the episode : Unlike River Avoidance, when an agent hits an obstacle (observer), the episode ends.
- randomness : There is randomness due to the nature of StarCraft map, so that there is a slight chance an episode won't end even if the agent collides with observers.

<p style="text-align:center;">
    <img style="max-width:32%; height:12em" src="/SAIDA_RL/assets/image/observer_escape_1.PNG" alt="observer_escape_1.PNG">
    <img style="max-width:32%; height:12em" src="/SAIDA_RL/assets/image/observer_escape_2.PNG" alt="observer_escape_2.PNG">
    <img style="max-width:32%; height:12em" src="/SAIDA_RL/assets/image/Avoid_Observer.gif" alt="observer_escape_3.PNG">
</p>

## Action Space 

In this scenario, only thing to decide is where to move. We provide three types of action space which has different output values.
To conquer this scenario with high probability of success, you may need to use the continuous action space rather than the discrete action space.

 Action Type  | Type | Output | Description |
---- | ---- | ---- | ---- |
0 | Discrete | one integer between 0 ~ action size | Output number means the moving direction. 0 means 3 o'clock. The total number of action spaces is defined as 360 / (move-dist). The last number of action means stop moving |
1 | Continuous | x, y, action number | ● X, Y : 2D coordinates, <br/> ● the action number 0 : move, 1 : attack move |
2 | Continuous | theta, radian, action number | ● theta : the angle started from 3 0'clock  <br/> ● radian : distance to move toward specific direction <br/> ● action number 0 : move, 1 : attack move |

## Observation 

To reach the goal without collision, the agent must know the information of close enemies such as position, velocity, acceleration, etc.
But some information like hit points, shield, energy, something has to do with combat may not necessary in this scenario.
Therefore, pre-processing of the raw observation into meaningful input features is essential.

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

- Current informaiton of all units in game

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

In the case of that scenario, when the agent reaches the top of the map or collides with an observer the episode ends.  
What is unusual thing in the scenario is that there is the safe area where the agent never die even if it collides with observers.  
So if you don't reshape the reward properly, you may see the scourge trying to stay at the safe area to avoid dying.    

- +5 : when agent reaches the goal.
- -5 : when agent collides with enemy
- +0.5 : others

## Environment configuration

You should know those parameters when you create the environment.

```python
from saida_gym.starcraft.avoidObserver import AvoidObserver
env = AvoidObserver(version=0, action_type=2, frames_per_step=6, verbose=0)
```
{% include_relative Environment.md %}
