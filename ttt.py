import random
from browser import window
from browser.html import *

class Tile(DIV):
  def __init__(self, text, x=0, y=0, width=0, height=0, display='inline', 
      opacity=1, axis=('left', 'top'), font_size='24px', **kw):
    for k, v in (
        (axis[0], x),  (axis[1], y),
        ('width', width), ('height', height),
        ('display', display), ('opacity', opacity),
        ('box-sizing', 'border-box'), ('position', 'absolute')):
      self.style[k] = v
      
    super().__init__(DIV(text, 
      style={
        'position': 'absolute',
        'left': '50%', 'top': '50%',
        'transform': 'translate(-50%, -50%)',
        'font-size': font_size}))
        
  @property
  def text(self):
    return self.children[0].text
  @text.setter
  def text(self, value):
    self.children[0].text = value
    
def popup(root, width='auto', height='auto'):
  if isinstance(root, (tuple, list, str, int, float, set, dict)):
    root = DIV(root)
  d = DIALOG(root)
  root.style['width'] = width
  root.style['height'] = height
      
  def on_click(ev):
    d.remove()
  d.bind('close', on_click)
    
  doc <= d
  
  d.showModal()
  
  return d

#---------------------------------

# nims
tile_size = 500/5

# 5 random 2 to 5 long heaps
stacks = {x: 
    Tile('|', width=tile_size, height=tile_size, 
      x=x[0]*tile_size, y=x[1]*tile_size)
  for x in (((j, i) for j in range(random.randint(2, 5))) for i in range(5)) for x in x}
  
# bind click events
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
for i in stacks.values():
  i.bind('click', on_click)
  
popup(list(stacks.values()), 500, 500)

# tic tac toe

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
for i in grid:
  i.bind('click', on_click)
  
popup(grid, 500, 500)
