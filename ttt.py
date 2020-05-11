# needs slot, align, grouper
class TTT:

  # Tile object
  Tile = slot('Tile', 'value', '')
  
  # Check pattern for win conditions. This simplifies the loop later.
  check_pattern = [(0, 4, 8), (2, 4, 6), (0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8)]
  
  width = 3
  
  def __init__(self, grid=None, turns=None):
    self.grid = [cls.Tile(i) for i in grid] if grid else [self.Tile() for i in range(9)]
    self.turns = turns or ['x', 'y']
    
    self.done = False
    
  # events
  def on_move(self, player, x, y, idx, target): pass
  def on_win(self, winner): pass
  
  # ---
    
  def move(self, x, y):
    if self.done:
      return
      
    target = self.grid[y*self.width+x]
    
    # if slot not occupied
    if not target.value:
      target.value = self.turns[0]
      self.on_move(self.turns[0], x, y, y*self.width+x, target)
      
      # flip turns
      self.turns = [self.turns[1], self.turns[0]]
      
      self.check_win()
      
  def check_win(self):
    for i in self.check_pattern:
      if self.grid[i[0]].value != '' and\
          self.grid[i[0]].value == self.grid[i[1]].value == self.grid[i[2]].value:
        self.done = True
        self.on_win(self.grid[i[0]])
    
  def __repr__(self):
    g = (i.value for i in self.grid)
    return align(grouper(g, self.width, ''))
