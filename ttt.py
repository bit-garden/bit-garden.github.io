# needs slot, align, grouper
# extra return messages are seen by @watch
@watch
class TTT:

  # Tile object
  @slot
  class Tile:
    value: str = ''
  
  # Check pattern for win conditions. This simplifies the loop later.
  check_pattern = [(0, 4, 8), (2, 4, 6), (0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8)]
  
  width = 3
  
  def __init__(self, grid=None, turns=None):
    self.grid = [self.Tile(i) for i in grid] if grid else [self.Tile() for i in range(9)]
    self.turns = turns or ['x', 'y']
    
    self.done = False
    
  # events
  def on_move(self, player, x, y, idx, target): pass
  def on_win(self, winner): pass
  def on_next_turn(self, who): pass
  
  # ---
    
  def move(self, x, y):
    if self.done:
      return 'Game already over'
      
    target = self.grid[y*self.width+x]
    
    # if slot not occupied
    if not target.value:
      target.value = self.turns[0]
      self.on_move(self.turns[0], x, y, y*self.width+x, target)
      
      # flip turns
      self.turns[:] = [self.turns[1], self.turns[0]]
      self.on_next_turn(self.turns[0])
      
      self.check_win()
    else:
      return 'Spot already taken'
      
  def check_win(self):
    for i in self.check_pattern:
      if self.grid[i[0]].value != '' and\
          self.grid[i[0]].value == self.grid[i[1]].value == self.grid[i[2]].value:
        self.done = True
        self.on_win(self.grid[i[0]])
        return True
    return False
    
  def __repr__(self):
    return 'TTT()'
  
  def __str__(self):
    g = (i.value for i in self.grid)
    return align(grouper(g, self.width, ''))

t = TTT()
for i in range(9):
  t.move(i, 0)