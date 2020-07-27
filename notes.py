from itertools import zip_longest, permutations, groupby
from operator import attrgetter

#    
# --- PURE PYTHON ---
# there should be no platform specific code here

class Q:
  def __call__(self, arg):
    if callable(arg):
      def f(*l, **kw):
        _ret = arg(*l, **kw)
        args = ', '.join(j for j in (', '.join(repr(i) for i in l), ', '.join(f'{k}={v!r}' for k, v in kw.items())) if j)
        print(f'{arg.__qualname__}({args}) -> {_ret}')
        return _ret
        
      f.__name__ = arg.__name__
      f.__qualname__ = arg.__qualname__
      return f
    print(arg)
    return arg
    
  def __or__(self, arg):
    print(arg)
    return arg
  
  __truediv__ = __div__ = __or__

q = Q()


class COLOR:
  crimson = (220, 20, 60)
  emerald = (0, 89, 11)
  cobalt = (0, 71, 171)
  gold = (255, 215, 0)
  red = (255, 0, 0)
  blue = (0, 0, 255)
  green = (0, 255, 0)
  
def grouper(iterable, n, fillvalue=None):
  """ Collect data into fixed-length chunks or blocks. """
  # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
  args = [iter(iterable)] * n
  return zip_longest(*args, fillvalue=fillvalue)

def slot(f, *components, verbose=False):
  """ A simple c-struct like generator. 
  
  This does not inherit functions and constructs a new class from scratch
    using exec.
  """
  if callable(f):
    args = tuple({x for x in (vars(i) for i in f.__mro__ if i is not object) for x in x if not x.startswith('__')})
    defaults = tuple(getattr(f, i) for i in args)
    supers = tuple(i.__qualname__ for i in f.__mro__ if i is not object and i is not f)
    components = '__components__ = ' + repr(supers)
    name = f.__name__ 
    
    largs = ', '.join(args[:-len(defaults)] if defaults else args)
  else:
    components = list(components)
    slots = components.pop(0)
    defaults = components
    args = tuple(slots.split(' '))
    largs = ', '.join(args[:-len(defaults)] if defaults else args)
    name = f
    supers = ()

  # Gather list of collections
  collections = {k: repr(v) for k, v in zip(args[-len(defaults):], defaults)
      if type(v) in (list, tuple, dict, set)}

  # Set default signature.
  kwargs = ', '.join(f'{k}={repr(None if k in collections else v)}' for k, v in zip(args[-len(defaults):], defaults))
  all_args = ', '.join(i for i in [largs, kwargs] if i)

  # Self setters.
  sargs = '\n    '.join(f'self.{k} = {v}' for k, v in zip(args, (f'{collections[i]} if {i} is None else {i}' if i in collections else i for i in args)))

  slot_template = f'''
class {name}:
  __slots__ = {repr(args)}
  {components if supers else ''}
  def __init__(self,
      {all_args}):
    {sargs}

  def __iter__(self):
    yield from ({", ".join(f"self.{i}" for i in args)})
    
  def __repr__(self):
    return f"""{{self.__class__.__name__}}({{", ".join(f"{{k}}={{v!r}}" 
        for k, v in zip(self.__slots__, self))}})"""
'''.strip()
  # Verbosity to print out generated code.
  if verbose:
    print(slot_template)

  local = {}
  exec(slot_template, local)
  return local[name]


def parse_note(s):
  title = None
  items = {}
  current_group = None
  parent_done = False

  for row in s.strip().splitlines():
    if not title and row.startswith('['):
      title = row.strip().lstrip('[').rstrip(']').strip()
      continue

    sub = False
    done = False

    if row.startswith(' '):
      sub = True
      row = row.strip(' ')

    if row.startswith('#'):
      done = True
      row = row.lstrip('#').strip()

    if not sub and row.strip():
      parent_done = done
      current_group = items.get(row, {'done': done, 'items': {}})
      items[row] = current_group
      continue

    if sub and parent_done:
      done = True

    current_group['items'][row] = done
  
  return {'title': title, 'items': items}

def note2str(n):
  if n['title']:
    out = f"[ {n['title']} ]\n"
  else:
    out = ''
  for k, v in n['items'].items():
    out += f"{'# ' if v['done'] else ''}{k}\n"
    for kk, vv in v['items'].items():
      out += f"  {'# ' if (vv or v['done']) else ''}{kk}\n"
  return out.strip()

def note2sortedstr(n):
  def item_sort(i):
    return i[1], i[0].lower()
  def group_sort(i):
    return i[1]['done'], i[0].lower()

  if n['title']:
    out = f"[ {n['title']} ]\n"
  else:
    out = ''
  for k, v in sorted(n['items'].items(), key=group_sort):
    out += f"{'# ' if v['done'] else ''}{k}\n"
    for kk, vv in sorted(v['items'].items(), key=item_sort):
      out += f"  {'# ' if (vv or v['done']) else ''}{kk}\n"
  return out.strip()
  
def align(l, *attrs, zip=zip):
  """ Aligns lists into a table like structure for neatness.
  
  This also aligns number like values to the right.
  """

  l = (*l,)  # exhaust generators
  if isinstance(l[0], dict):
    h = attrs or l[0].keys()
    l = [[f'{j}' for j in i] for i in zip(h, *([i[k] for k in h] for i in l))]
  elif isinstance(l[0], (list, tuple)):
    l = [[f'{j}' for j in i] for i in zip(*l)]
  else:
    h = attrs or (l[0].__slots__ if hasattr(l[0], '__slots__') else vars(l[0]))
    l = [[f'{j}' for j in i] for i in zip(h, *([getattr(i, k) for k in h] for i in l))]

  for col in l:
    column_width = len(max(col, key=len))
    for idx, c in enumerate(col):
      try:
        float(c)
        col[idx] = c.rjust(column_width)
      except:
        col[idx] = c.ljust(column_width)
  
  # transpose back and return.
  return '\n'.join(' | '.join(i).rstrip() for i in zip(*l))

def table(s, types={}, char='|'):
  ds = ([j.strip() for j in i.split(char)] 
    for i in s.strip().lstrip('#').splitlines()
      if not i.strip().startswith('#'))

  h = next(ds)
  if types:
    for i in ds:
      yield {k: types[k](v or types[k]()) if k in types else v
          for k, v in zip(h, i)}
  else:
    yield from (dict(zip(h, i)) for i in ds)

def matrix(s, types=None, char='|'):
  ds = ([j.strip() for j in i.split(char)] 
    for i in s.strip().splitlines()
      if not i.strip().startswith('#'))

  if not types:
    yield from ds
  elif isinstance(types, (list, tuple)):
    for i in ds:
      yield [k(v or k()) for k, v in zip(types, i)]
  else:
    for i in ds:
      yield [types(v or types()) for v in i]


def chunk(l, n=1):
  current_batch = []
  for item in l:
    current_batch.append(item)
    if len(current_batch) == n:
      yield current_batch
      current_batch = []
  if current_batch:
    yield current_batch

def gatherby(l, k=None, m=None):
  gb = {}

  if m:
    for i in l:
      gb.setdefault(k(i), []).append(m(i))
  elif k:
    for i in l:
      gb.setdefault(k(i), []).append(i)
  else:
    def k(i):
      return i
    for i in l:
      gb.setdefault(i, []).append(i)

  return gb
    



    
# --- color tools
# dark material height to tint
opacities = {0: 0, 1: .05, 2: .07, 3: .08, 4: .09, 6: .11, 8: .12, 12: .14, 16: .15, 24: .16}
font_opacities = {'high': .87, 'medium': .6, 'disabled': .38}
font_error = (176, 0, 32, 1) #B00020

def tint(val, r, g, b, *_):
  """ Tints rgb values to apply 'light'. """
  rt = r + (val * (255 - r))
  gt = g + (val * (255 - g))
  bt = b + (val * (255 - b))
  return (rt, gt, bt, *_)

def shade(val, r, g, b, *_):
  """ Tints rgb values to apply 'dark'. """
  rt = r - r*val 
  gt = g - g*val 
  bt = b - b*val 
  return (rt, gt, bt, *_)
 
def is_dark(r, g, b, *_):
  """ Determine color too dark. """
  # HSP (Highly Sensitive Poo) equation from http://alienryderflex.com/hsp.html
  hsp =  0.299 * r**2 + 0.587 * g**2 + 0.114 * b**2
  return 127.5 >= hsp**(1/2.0)
  
def hilo(a, b, c):
  """ Used in compliment function. """
  if c < b: b, c = c, b
  if b < a: a, b = b, a
  if c < b: b, c = c, b
  return a + c

#def complement(r, g, b, *_):
#  """ Get complement color. """
#  k = hilo(r, g, b)
#  return (*(k - u for u in (r, g, b)), *_)
  
# --- timer methods
import time
#import asyncio
    
class Ticker:
  """ Holds reference to last diff or tick of time. 
  
  Helps account for delay in code execution to keep interval based
    functions as close as possible to their intended interval.
  """
  def __init__(self):
    self.last_tick = time.time()
    self.last_diff = self.last_tick
    
  def tick(self, sec):
    n = time.time()
    diff = self.last_tick + sec - n
    self.last_tick += sec
    
    return max(diff, 0)
    
  def diff(self):
    _diff = time.time() - self.last_diff
    self.last_diff += _diff
    
    return _diff
    
def delay(d, running=None):
  if not running:
    running = [None, 'running']
  def _delay(f, *l, **kw):
    async def __delay():
      await asyncio.sleep(d)
      if running:
        running[:] = [f(*l, **kw), 'done']
      else:
        running[:] = [None, 'cancelled']
    asyncio.create_task(__delay())
    return f
  return _delay
  
def Delay(d, f, *l, **kw):
  task = [None, 'running']
  delay(d, task)(f, *l, **kw)
  return task
  
def debounce(d):
  """ Prevent excuting function until delay has been waited out since last attempt. """
  _delay = None
  def _debounce(f):
    def __debounce(*l, **kw):
      nonlocal _delay
      if _delay:
        _delay.clear()
      _delay = Delay(d, f, *l, **kw)
    return __debounce
  return _debounce
  
def throttle(d):
  """ Limit rate of how fast a function can be called. """
  _delay, last, _l, _kw = None, None, [], {}
  def _throttle(f):
    def do():
      nonlocal last
      last = time.time()
      f(*_l, **_kw)
    def __throttle(*l, **kw):
      nonlocal _delay, last, _l, _kw
      if _delay == [None, 'running']:
        _l, _kw = l, kw
      elif last and time.time() - last < d:
        _l, _kw = l, kw
        _delay = Delay(d - (time.time() - last), do)
      else: 
        last = time.time()
        f(*l, **kw)
    return __throttle
  return _throttle
  
async def loop(d, f, running=True):
  t = Ticker()
  while f(t.diff()) is not False and running:
    await asyncio.sleep(t.tick(d))

def stagger(l, d):
  d = d/len(l)
  def _stagger(f):
    for idx, i in enumerate(l, 1):
      if isinstance(i, (tuple, list)):
        Delay(idx*d, f, *i)
      elif isinstance(i, (dict)):
        Delay(idx*d, f, **i)
      else:
        Delay(idx*d, f, i)
  return _stagger
  
#decode hex to tuple
#tuple(bytes.fromhex("aabbccaa"))
      
def axis_to_tracks(axis, viewport, offset=None):
  """ Used to generate a non-uniform grid. """
  axis = [[v*b if isinstance(b, float) else b for b in a] 
    for v, a in zip(viewport, axis)]

  for v, a in zip(viewport, axis):
    if (count:=len([i for i in a if i is ...])):
      fill = (v - sum(i for i in a if i is not ...))/count
      while ... in a:
        a[a.index(...)] = fill

  axis = [[round(sum(a[:i])) for i in range(len(a)+1)] for a in axis]
  axis = [[(a[i]+o, a[i+1]+o) for i in range(len(a)-1)] 
      for a, o in zip(axis, offset or [0]*len(axis))]
  
  return axis

def tracks_to_coords(tracks, cell, ecell=None, padding=None):
  """ Returns pairs of start and end of each axis in tracks. """
  return [(t[c][0]+p, t[ec or c][1]-p*2) 
      for c, ec, t, p in zip(
          cell, ecell or [None]*len(cell), 
          tracks, padding or [0]*len(cell))]

def point_to_cell(tracks, point=(0, 0)):
  """ Get cell that point is in on tracks. """
  return [next((idx for idx, i in enumerate(t) if i[0] <= p <= i[1]), None)
      for t, p in zip(tracks, point)]
      
#todo
# radial tools
import math

class Radial_menu:
  def __init__(self, x, y, items, min_distance=0):
    self.x = x
    self.y = y
    self.items = items
    self.min_distance = min_distance

  def __getitem__(self, key):
    if math.dist((self.x, self.y), key) >= self.min_distance:
      return self.deg_to_item(self.get_angle(*key))
    return None

  def get_angle(self, x2, y2):
    angle = math.degrees(math.atan2(y2 - self.y, x2 - self.x)) + 90
    if angle < 0:
      angle += 360
    return angle

  def deg_to_item(self, deg):
    unit_size = 360 // len(self.items)
    offset = deg + unit_size/2

    sel = int(offset//unit_size)
    if sel >= len(self.items):
      sel = 0

    return self.items[sel]

  def deg_to_idx(self, deg):
    unit_size = 360 // len(self.items)
    offset = deg + unit_size/2

    sel = int(offset//unit_size)
    if sel >= len(self.items):
      sel = 0

    return sel

  def get_pos(self, radius):
    unit_size = math.radians(360/len(self.items))
    for idx, i in enumerate(self.items):
      x = self.x + radius * math.cos(unit_size*idx - math.pi/2)
      y = self.y + radius * math.sin(unit_size*idx - math.pi/2)
      yield i, x, y

class Spiral_menu:
  def __init__(self, x, y, items, min_distance=0):
    self.x = x
    self.y = y
    self.items = items
    self.min_distance = min_distance
    self.loop = 0

    self.last_peek = 0

  def __getitem__(self, key):
    if math.dist((self.x, self.y), key) >= self.min_distance:
      return self.deg_to_item(self.get_angle(*key))
    return None

  def get_angle(self, x2, y2):
    angle = math.degrees(math.atan2(y2 - self.y, x2 - self.x)) + 90
    if angle < 0:
      angle += 360
    return angle

  def deg_to_item(self, deg):
    unit_size = 360 // 12
    deg += self.loop*360
    offset = deg + unit_size/2

    sel = int(offset//unit_size)
    sel = max(min(sel, len(self.items)-1), 0)

    return self.items[sel]

  def deg_to_idx(self, deg):
    unit_size = 360 // 12
    deg += self.loop*360
    offset = deg + unit_size/2

    sel = int(offset//unit_size)
    #return max(min(sel, len(self.items)-1), 0)
    return max(sel, 0)

  def get_near(self, x, y):
    around = 6
    spiral_shift = 25/around

    ang = self.get_angle(x, y)

    if ang - 180 > self.last_peek:
      self.loop = max(self.loop-1, 0)
      #print('loop down')
    if ang + 180 < self.last_peek:
      self.loop += 1
      #print('loop up')

    self.last_peek = ang

    idx = self.deg_to_idx(ang)
    
    if idx < around:
      items = self.items[:around*2+1]
    else:
      items = self.items[idx-around:idx+around+1]
    items += [None]*(around*2+1 - len(items))

    shifts = [[i, (spiral_shift*iidx)] for iidx, i in enumerate(items, max(-around, -idx))]

    unit_size = math.radians(360/12)
    radius = 100
    for _idx, i in enumerate(shifts):
      x = self.x + (radius - i[1]) * math.cos(unit_size*_idx - math.pi/2 + unit_size*(max(idx-around, 0)))
      y = self.y + (radius - i[1]) * math.sin(unit_size*_idx - math.pi/2 + unit_size*(max(idx-around, 0)))
      #i[1:] = (x, y)
      shifts[_idx] = {'item': i[0], 'x': x, 'y': y}

    return idx, shifts

  
# --- misc ---
'(,)(?=(?:[^"]|"[^"]*")*$)' # csv: capture unencolsed ',' chars


clear = '\x1b[2K'
goup = '\x1b[1A'

def lprint(user_string="", x=0, y=0):
  # print at x,y in term
  print(f"\033[{y+1};{x+1}f{user_string}", flush=True, end='')

def terminal_size():
  # get term size
  import fcntl, termios, struct
  h, w, hp, wp = struct.unpack('HHHH',
    fcntl.ioctl(0, termios.TIOCGWINSZ,
    struct.pack('HHHH', 0, 0, 0, 0)))
  return w, h
      
# this is very slow in brython. Offload to native js if possible
#import re
#def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
#  return [int(text) if text.isdigit() else text.lower()
#          for text in _nsre.split(s)]

# --- For browser --- 
# include like normal imports
# hacky exec version of itemgetter for speed
# slower creation, but faster execute for more than 1 item
#def itemgetter(*items):
#  template = f'''def f(i): return {", ".join(f"i['{i}']" for i in items)}'''
#  loc = {}
#  exec(template, {}, loc)
#  return loc['f']
#
## hacky exec version of attrgetter for speed
## slower creation, but faster execute
#def attrgetter(*items):
#  template = f'''def f(i): return {", ".join(f"i.{i}" for i in items)}'''
#  loc = {}
#  exec(template, {}, loc)
#  return loc['f']
#  
#class groupby:
#  # [k for k, g in groupby('AAAABBBCCDAABBB')] --> A B C D A B
#  # [list(g) for k, g in groupby('AAAABBBCCD')] --> AAAA BBB CC D
#  def __init__(self, iterable, key=None):
#    if key is None:
#      key = lambda x: x
#    self.keyfunc = key
#    self.it = iter(iterable)
#    self.tgtkey = self.currkey = self.currvalue = object()
#  def __iter__(self):
#    return self
#  def __next__(self):
#    self.id = object()
#    while self.currkey == self.tgtkey:
#      self.currvalue = next(self.it)  # Exit on StopIteration
#      self.currkey = self.keyfunc(self.currvalue)
#    self.tgtkey = self.currkey
#    return (self.currkey, self._grouper(self.tgtkey, self.id))
#  def _grouper(self, tgtkey, id):
#    while self.id is id and self.currkey == tgtkey:
#      yield self.currvalue
#      try:
#        self.currvalue = next(self.it)
#      except StopIteration:
#        return
#      self.currkey = self.keyfunc(self.currvalue)
#
#class zip_longest:
#  def __init__(self, *args, fillvalue = None):
#    self.args = [iter(arg) for arg in args]
#    self.fillvalue = fillvalue
#    self.units = len(args)
#  
#  def __iter__(self):
#    return self
#  
#  def __next__(self):
#    temp = []
#    nb = 0
#    for i in range(self.units):
#      try:
#        temp.append(next(self.args[i]))
#        nb += 1
#      except StopIteration:
#        temp.append(self.fillvalue)
#    if nb==0:
#      raise StopIteration
#    return tuple(temp)
#
#def partial(func, *args, **keywords):
#  def newfunc(*fargs, **fkeywords):
#    newkeywords = {**keywords, **fkeywords}
#    return func(*args, *fargs, **newkeywords)
#  newfunc.func = func
#  newfunc.args = args
#  newfunc.keywords = keywords
#  return newfunc
#  
#class permutations:
#  def __init__(self, iterable, r = None):
#    self.pool = tuple(iterable)
#    self.n = len(self.pool)
#    self.r = self.n if r is None else r
#    self.indices = list(range(self.n))
#    self.cycles = list(range(self.n, self.n - self.r, -1))
#    self.zero = False
#    self.stop = False
#
#  def __iter__(self):
#    return self
#
#  def __next__(self):
#    indices = self.indices
#    if self.r > self.n:
#      raise StopIteration
#    if not self.zero:
#      self.zero = True
#      return tuple(self.pool[i] for i in indices[:self.r])
#    
#    i = self.r - 1
#    while i >= 0:
#      j = self.cycles[i] - 1
#      if j > 0:
#        self.cycles[i] = j
#        indices[i], indices[-j] = indices[-j], indices[i]
#        return tuple(self.pool[i] for i in indices[:self.r])
#      self.cycles[i] = len(indices) - i
#      n1 = len(indices) - 1
#      assert n1 >= 0
#      num = indices[i]
#      for k in range(i, n1):
#        indices[k] = indices[k+1]
#      indices[n1] = num
#      i -= 1
#    raise StopIteration
## this is specific to handling html ui
#from browser import window
#from browser.html import *
#
#def bind(events, *elements):
#  """ @bind('event event2', el, el) """
#  def _bind(f):
#    for event in events.split(' '):
#      for element in elements:
#        element.bind(event, f)
#    return f
#  return _bind
#  
#def bind_once(events, *elements):
#  """ @bind_once('event event2', el, el) 
#  
#  This removes the binding once any event is fired on any of the elements.
#  This can be used to stop doubleclicking causing errors.
#  """
#  def _bind(f):
#    def _unbind(*l, **kw):
#      for event in events.split(' '):
#        for element in elements:
#          element.unbind(event, _unbind)
#      return f(*l, **kw)
#        
#    for event in events.split(' '):
#      for element in elements:
#        element.bind(event, _unbind)
#    return f
#    
#  return _bind
#
#def popup(root, width='auto', height='auto'):
#  """ Popup dialog using <dialog>. """
#  if isinstance(root, (tuple, list, str, int, float, set, dict)):
#    if isinstance(root, str):
#      root = DIV(root)
#    else:
#      root = DIV(repr(root))
#    
#  d = DIALOG(root)
#  root.style['width'] = width
#  root.style['height'] = height
#      
#  @bind('close', d)
#  def _(ev):
#    d.remove()
#    cm_editbox.focus()
#    
#  doc <= d
#  d.showModal()
#  return d
#  
#class Popup:
#  """ Context for popup function. """
#  def __init__(self, *l, **kw):
#    self.d = popup(*l, **kw)
#    
#  def __enter__(self):
#    return self
#    
#  def __exit__(self, *l):
#    self.d.close()
#    
#class El:
#  """ Context to remove element on error. """
#  def __init__(self, element, parent=None):
#    self.element = element
#    if parent:
#      parent <= self.element
#  
#  def __enter__(self):
#    return self.element
#    
#  def __exit__(self, *l):
#    self.element.remove()
#    
#class Box(DIV):
#  def __init__(self, *l, x=0, y=0, width=0, height=0, display='inline', 
#      opacity=1, axis=('top', 'left'), **kw):
#    super().__init__(*l, **kw)
#
#    self.axis = axis
#    self.x = x
#    self.y = y
#    self.width = width
#    self.height = height
#    self.display = display
#    self.opacity = opacity
#    
#    for k, v in (
#        ('box-sizing', 'border-box'),
#        ('position', 'absolute')):
#      self.style[k] = v
#
#  @property
#  def opacity(self):
#    return self.style['opacity']
#  @opacity.setter
#  def opacity(self, value):
#    self.style['opacity'] = value
#
#  @property
#  def display(self):
#    return self.style['display']
#  @display.setter
#  def display(self, value):
#    self.style['display'] = value
#
#  @property
#  def x(self):
#    return self.style[self.axis[1]]
#  @x.setter
#  def x(self, value):
#    self.style[self.axis[1]] = value
#
#  @property
#  def y(self):
#    return self.style[self.axis[0]]
#  @y.setter
#  def y(self, value):
#    self.style[self.axis[0]] = value
#
#  @property
#  def width(self):
#    return self.style['width']
#  @width.setter
#  def width(self, value):
#    self.style['width'] = value
#
#  @property
#  def height(self):
#    return self.style['height']
#  @height.setter
#  def height(self, value):
#    self.style['height'] = value
#      
#class Icon(I):
#  def __init__(self, *l, font_size='32px', **kw):
#    super().__init__(*l, **{'Class': 'material-icons', **kw})
#    
#    self.style['font-size'] = font_size
#    
#class Tile(Box):
#  def __init__(self, text='', font_size='24px', **kw):
#    for k, v in (
#        ('overflow', 'hidden'),):
#      self.style[k] = v
#        
#    super().__init__(DIV(text, 
#      style={
#        'position': 'absolute',
#        'left': '50%',
#        'top': '50%',
#        'transform': 'translate(-50%, -50%)',
#        'font-size': font_size,
#        'text-align': align}), **kw)
#
#  @property
#  def text(self):
#    return self.children[0].text
#  @text.setter
#  def text(self, value):
#    self.children[0].text = value
#
#
## update to handle new axis methods
#coords_map = ('left', 'width'), ('top', 'height')
#def gmap(instance, *l, **kw):
#  for k, v in zip(coords_map, tracks_to_coords(*l, **kw)):
#    instance.style[k[0]] = v[0]
#    instance.style[k[1]] = v[1] - v[0]
#    
## --- 
#
#def make_ripple(element, base_color=(18,18,18,1), no_key=False):
#  """ Adds ripple click effect to an element. """
#  mutator = tint if is_dark(*base_color) else shade
#  
#  hover_color = mutator(opacities[8], *base_color[:3], 1)
#  pulse_color = mutator(.4, *base_color[:3], 1)
#  
#  glows = [(*(base_color[:3]), i) for i in (.14, .12, .2)]
#  
#  @bind('mouseout blur', element)
#  def prep_ripple(*_):
#    base_sty = {'background-color': f'rgba{base_color}', 
#      'background-position': 'center', 
#      'box-shadow': 'none',
#      'transition': 'all 0.3s',
#      'transition-timing-function': 'cubic-bezier(0.4, 0.0, 0.2, 1)'}
#    for k, v in base_sty.items():
#      element.style[k] = v
#  prep_ripple()
#
#  @bind('mouseover mouseup focus', element)
#  def pre_ripple(*_):
#    hover_sty = {'transition': 'all 0.3s',
#      'box-shadow': f'0 4px 5px 0 rgba{glows[0]}, 0 1px 10px 0 rgba{glows[1]}, 0 2px 4px -1px rgba{glows[2]}',
#      'background': f'rgba{hover_color} radial-gradient(circle, transparent 1%, rgba{hover_color} 1%) center/15000%'}
#
#    for k, v in hover_sty.items():
#      element.style[k] = v
#
#  @bind('mousedown', element)
#  def do_ripple(*_):
#    active_sty = {'background-color': f'rgba{pulse_color}',
#      #'box-shadow': f'0 4px 5px 0 rgba(220, 20, 60, 0.14), 0 1px 10px 0 rgba(220, 20, 60, 0.12), 0 2px 4px -1px rgba(220, 20, 60, 0.20)',
#      'background-size': '100%', 'transition': 'all 0s'}
#
#    for k, v in active_sty.items():
#      element.style[k] = v
#    
#    Delay(.05, pre_ripple)
#
#  # manually call a ripple pulse effect.
#  def _do_ripple(*_):
#    pre_ripple()
#    Delay(.4, do_ripple)
#    Delay(.45, pre_ripple)
#    Delay(.6, prep_ripple)
#  
#  def _key_ripple(ev):
#    if ev.key=='Enter' or ev.key==' ':
#      do_ripple()
#  
#  if not no_key:
#    element.bind('keydown', _key_ripple)
#  
#  return element, _do_ripple
