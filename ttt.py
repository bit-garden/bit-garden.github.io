# nims
import random
from browser import window

tile_size = 500/5

# 5 random 2 to 5 long heaps
stacks = {x: 
    Tile('|', width=tile_size, height=tile_size, 
      x=x[0]*tile_size, y=x[1]*tile_size)
  for x in (((j, i) for j in range(random.randint(2, 5))) for i in range(5)) for x in x}
  
# bind click events
@bind('click', *stacks.values())
def on_click(ev):
  # find index of clicked element
  idx = next(k for k, v in stacks.items() if ev.target.inside(v))

  # remove the clicked one on to the end of the row
  for i in range(idx[0], 5):
    if (i, idx[1]) not in stacks:
      break

    stacks[i, idx[1]].remove()
    del stacks[i, idx[1]]

    # if you remove the last stick, you lose
    if not stacks:
      window.alert('You lose')
  
popup(list(stacks.values()), 500, 500)


# tic tac toe
from browser import window

# Check pattern for possible wins.
check_pattern = ((0, 4, 8), (2, 4, 6), (0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8))
width = 3

tile_size = 500/3

grid = tuple(
    Tile('', height=tile_size, width=tile_size,
      x=i%width*tile_size, y=i//width*tile_size) 
  for i in range(9))  # 9 tiles in the game
    
turns = ['x', 'o']  # these flip as the turns go by
done = []
  
@bind('click', *grid)
def on_click(ev):
  if done:  # if done, stop
    return
    
  idx = next(idx for idx, i in enumerate(grid) if ev.target.inside(i))
  target = grid[idx]
  
  if target.text:  # if tile taken, stop
    return
  
  # set text of UI element, then change turns
  target.text = turns[0]
  turns[:] = turns[1], turns[0]
  
  # check for win
  for i in check_pattern:
    if grid[i[0]].text and grid[i[0]].text == grid[i[1]].text == grid[i[2]].text:
      window.alert(f'{turns[1]} has won!')
      done.append(True)  # the game is over, no more playing
  
popup(grid, 500, 500)
