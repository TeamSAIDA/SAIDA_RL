## Grid World

Grid World, a two-dimensional plane (5x5), is one of the easiest and simplest environments to test reinforcement learning algorithm.  
 In this environment, agents can only move up, down, left, right in the grid, and there are traps in some tiles.  
The agent starts at the fixed start position and when it arrives at the goal or trap, episode ends.

<p style="text-align:center;">
    <img style="max-width:49%; height:14em" src="/SAIDA_RL/assets/image/Grid_world.PNG" alt="Grid_world.png">
    <img style="max-width:49%; height:14em" src="/SAIDA_RL/assets/image/Grid_World.gif" alt="Grid_World.gif">
</p>

As you can see from the picture, every time the Q-table is updated, Q-values in tiles is changing.

## Action Space

4 discrete action space

- 0 : Move to the up 
- 1 : Move to the down
- 2 : Move to the left
- 3 : Move to the right

## Reward

- -1 : Move to red dot
- +1 : Move to green dot

