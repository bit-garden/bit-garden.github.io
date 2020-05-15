# --- For browser --- include like normal imports
# hacky exec version of itemgetter for speed
# slower creation, but faster execute for more than 1 item
def itemgetter(*items):
  template = f'''def f(i): return {", ".join(f"i['{i}']" for i in items)}'''
  loc = {}
  exec(template, {}, loc)
  return loc['f']

# hacky exec version of attrgetter for speed
# slower creation, but faster execute
def attrgetter(*items):
  template = f'''def f(i): return {", ".join(f"i.{i}" for i in items)}'''
  loc = {}
  exec(template, {}, loc)
  return loc['f']
  
class groupby:
  # [k for k, g in groupby('AAAABBBCCDAABBB')] --> A B C D A B
  # [list(g) for k, g in groupby('AAAABBBCCD')] --> AAAA BBB CC D
  def __init__(self, iterable, key=None):
    if key is None:
      key = lambda x: x
    self.keyfunc = key
    self.it = iter(iterable)
    self.tgtkey = self.currkey = self.currvalue = object()
  def __iter__(self):
    return self
  def __next__(self):
    self.id = object()
    while self.currkey == self.tgtkey:
      self.currvalue = next(self.it)  # Exit on StopIteration
      self.currkey = self.keyfunc(self.currvalue)
    self.tgtkey = self.currkey
    return (self.currkey, self._grouper(self.tgtkey, self.id))
  def _grouper(self, tgtkey, id):
    while self.id is id and self.currkey == tgtkey:
      yield self.currvalue
      try:
        self.currvalue = next(self.it)
      except StopIteration:
        return
      self.currkey = self.keyfunc(self.currvalue)

class zip_longest:
  def __init__(self, *args, fillvalue = None):
    self.args = [iter(arg) for arg in args]
    self.fillvalue = fillvalue
    self.units = len(args)
  
  def __iter__(self):
    return self
  
  def __next__(self):
    temp = []
    nb = 0
    for i in range(self.units):
      try:
        temp.append(next(self.args[i]))
        nb += 1
      except StopIteration:
        temp.append(self.fillvalue)
    if nb==0:
      raise StopIteration
    return tuple(temp)

def partial(func, *args, **keywords):
  def newfunc(*fargs, **fkeywords):
    newkeywords = {**keywords, **fkeywords}
    return func(*args, *fargs, **newkeywords)
  newfunc.func = func
  newfunc.args = args
  newfunc.keywords = keywords
  return newfunc
  
class permutations:
  def __init__(self, iterable, r = None):
    self.pool = tuple(iterable)
    self.n = len(self.pool)
    self.r = self.n if r is None else r
    self.indices = list(range(self.n))
    self.cycles = list(range(self.n, self.n - self.r, -1))
    self.zero = False
    self.stop = False

  def __iter__(self):
    return self

  def __next__(self):
    indices = self.indices
    if self.r > self.n:
      raise StopIteration
    if not self.zero:
      self.zero = True
      return tuple(self.pool[i] for i in indices[:self.r])
    
    i = self.r - 1
    while i >= 0:
      j = self.cycles[i] - 1
      if j > 0:
        self.cycles[i] = j
        indices[i], indices[-j] = indices[-j], indices[i]
        return tuple(self.pool[i] for i in indices[:self.r])
      self.cycles[i] = len(indices) - i
      n1 = len(indices) - 1
      assert n1 >= 0
      num = indices[i]
      for k in range(i, n1):
        indices[k] = indices[k+1]
      indices[n1] = num
      i -= 1
    raise StopIteration
    
# --- PURE PYTHON ---
# there should be no platform specific code here

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

def parse_by(string_in, delimiter=',', quotes=('"', "'")):
  """ Splits text by lines accounting for quotes. """
  
  last = ''  # last valid quote character
  last_idx = 0  # position of last quote character
  ignore_next_char = False  # ignore next char. Used for backslashes
  current_row = []  # contains currnet row until it's yielded

  for idx, i in enumerate(string_in):
    if ignore_next_char: 
      ignore_next_char = False
      continue
    if i in quotes and not last:
      last = i
    elif i in quotes and i == last:
      last = ''
    #elif i == '\\':
    #  ignore_next_char = True
    elif not last and i in (delimiter, '\n'):
      current_row.append(''.join(string_in[last_idx: idx].strip()))
      if i == '\n':
        yield current_row
        current_row = []
      last_idx = idx + 1
  current_row.append(''.join(string_in[last_idx:].strip()))
  yield current_row

def slot(f, args='', *defaults, verbose=False):
  """ A simple c-struct like generator. 
  
  This can be used to decorate annotated classes as well.
  This does not inherit functions and constructs a new class from scratch
    using exec.
  """
  
  # Switch between decorator or normal call handling.
  if callable(f):
    args = tuple(f.__annotations__.keys())
    defaults = [getattr(f, i) for i in args if hasattr(f, i)]
    name = f.__name__
  else:
    args = tuple(args.split(' '))
    name = f or 'Slot_base'
  largs = ', '.join(args[:-len(defaults)] if defaults else args)

  # Gather list of collections
  collections = {k: repr(v) for k, v in zip(args[-len(defaults):], defaults)
      if type(v) in (list, tuple, dict, set)}

  # Set default signature.
  kwargs = ', '.join(f'{k}={repr(None if k in collections else v)}' 
      for k, v in zip(args[-len(defaults):], defaults))
  all_args = ', '.join(i for i in [largs, kwargs] if i)

  # Self setters.
  sargs = '\n    '.join(f'self.{k} = {v}' 
      for k, v in zip(args, (f'{collections[i]} if {i} is None else {i}' 
      if i in collections else i for i in args)))

  slot_template = f'''
class {name}:
  __slots__ = {repr(args)}
  def __init__(self,
      {all_args}):
    {sargs}

  def __repr__(self):
    return f"""{{self.__class__.__name__}}({{", ".join(f"{{k}}={{getattr(self, k)!r}}" 
        for k in self.__slots__)}})"""
'''.strip()

  if verbose:
    print(slot_template)

  local = {}
  exec(slot_template, local)
  return local[name]
  
# used with align function
def is_num(string_in):
  """ True if string is able to be casted to float. """
  try:
    float(string_in)
    return True
  except:
    return False

def align(*l):
  """ Aligns lists into a table like structure for neatness.
  
  This also aligns number like values to the right.
  """
  
  # transpose and fill in table.
  l = [[*i] for i in zip_longest(*(x for x in l for x in x), fillvalue='')]
  for col in l:
    column_width = max(len(f'{i}') for i in col)
    col[:] = [f'{i}'.rjust(column_width) if is_num(i) else f'{i}'.ljust(column_width) 
        for i in col]
  
  # transpose back and return.
  return '\n'.join(' | '.join(i).rstrip() for i in zip(*l))
  
def attr_map(instance, **kw):
  """ Map dict to object attributes. (like dict.update())"""
  for k, v in kw.items():
    setattr(instance, k, v)

def recycle(items, data, how=attr_map):
  """ Recycles items using data. (like reusing ui elements) """
  for i, d in zip_longest(items, data, fillvalue={}):
    if isinstance(d, dict):
      how(i, **d)
    else:
      how(i, **{k: getattr(d, k) for k in d.__slots__})
      
def parse(pattern, s):
  """ Extract text from strings using pattern.

  parse('[]:[]', '12:31') --> ['12', '31']
  """
  
  if pattern.startswith('[]'):
    pattern = ' ' + pattern
    s = ' ' + s
  ends = False
  if pattern.endswith('[]'): ends = True
  pattern = pattern.split('[]')
  out = []
  for idx, p in enumerate(pattern):
    if ends and idx == len(pattern)-1:
      out.append(s)
    else:
      _s = s.partition(p)
      out.append(_s[0])
      s = _s[2]
  return out[1:]

def lcast(l, *kinds):
  """ Cast list of data. 
  
  If kinds is only one type, cast all elements in l to that type.
  If more than one type, it maps the types to l.
  """
  
  if len(kinds) == 1:
    return (kinds[0](i) for i in l)
  return (k(i) for k, i in zip(kinds, l))
  
def fix_name(s):
  """ Return a safe string for Table to use. """
  safe = 'abcdefghijklmnopqurstuvwxyz0123456789'
  s = s.lower()
  if s[0] in '0123456789':
    s = '_' + s
  return ''.join(i if i in safe else '_' for i in s)

def Table(s, char='|', extras=[], cast=None):
  """ Splits string by lines and char.
  
  This returns Row objects with attributes that match the columns defined in the
    pipe table.
  Extra fields are generated from extras.
  Fields can be casted automatically.
  """
  
  ds = ([j.strip() for j in i.split(char)] for i in s.strip().lstrip('#').splitlines()
      if (_i:=i.strip()) and not _i.startswith('#'))
  heads = next(ds)
  Row = slot('Row', ' '.join(fix_name(i) for i in heads+extras), *['' for i in extras])
  if cast:
    return [Row(*lcast(i, *cast)) for i in ds]
  else:
    return [Row(*i) for i in ds]
    
class Observable:
  """ Observable base class for observable pattern. """
  __slots__ = 'callbacks',
  
  def __init__(self, *events):
    self.callbacks = {i:[] for i in events}

  # Decorator and normal function
  def bind(self, event, f=None):
    if f:
      self.callbacks[event].append(f)
    else:
      def _bind(f):
        self.callbacks[event].append(f)
        return f
      return _bind

  def unbind(self, event, f):
    self.callbacks[event].remove(f)

  def emit(self, event, *l, **kw):
    for c in self.callbacks[event]:
      c(self, *l, **kw)
      
func_type = type(lambda:True)
watch_depth = -1
def watch(f):
  """ Wrap class/functions to print out call graph live as the functions are called. """
  if isinstance(f, type):
    class _watch(f):
      if hasattr(f, '__slots__'):
        __slots__ = f.__slots__
        
      def __setattr__(self, key, value):
        if (hasattr(self, key) and 
            getattr(self, key) != value):
          print(f'{"| "*(watch_depth+1)}{f.__qualname__}.__setattr__({self!r}, {key!r}, {value!r})')
        super().__setattr__(key, value)

    defs = [i for i in {*dir(_watch)} - {*dir(object)} 
        if not i.startswith('__') and callable(getattr(_watch, i))]

    for i in defs:
      setattr(_watch, i, watch(getattr(_watch, i)))

  elif not isinstance(f, func_type):
    def _watch(self, *l, **kw):
      global watch_depth
      watch_depth += 1
      print(f'{"| "*watch_depth}{f.__qualname__}('\
          + ', '.join([repr(j) for j in l] + [f'{k}={v!r}' for k, v in kw.items()])+ ')')
      ret = f(*l, **kw)
      if ret is not None:
        print(f'{"| "*(watch_depth)}|-> {ret!r}')
      watch_depth -= 1
      return ret

  else:
    def _watch(*l, **kw):
      global watch_depth
      watch_depth += 1
      print(f'{"| "*watch_depth}{f.__qualname__}('\
          + ', '.join([repr(j) for j in l] + [f'{k}={v!r}' for k, v in kw.items()])+ ')')
      ret = f(*l, **kw)
      if ret is not None:
        print(f'{"| "*(watch_depth)}|-> {ret!r}')
      watch_depth -= 1
      return ret

  _watch.__name__ = f.__name__
  _watch.__qualname__ = f.__qualname__

  return _watch
      
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

def complement(r, g, b, *_):
  """ Get complement color. """
  k = hilo(r, g, b)
  return (*(k - u for u in (r, g, b)), *_)
  
# --- timer methods
import time
from browser import aio as asyncio

class timer:
  """ Timer context to time fragments of code. """
  
  def __init__(self, label='', func=print):
    self.func = func
    self.label = label

  def __enter__(self):
    self.start = time.time()
    return self

  def __exit__(self, *l):
    self.func(f'{self.label} {time.time() - self.start}')
    
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
    asyncio.run(__delay())
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
from math import atan2, pi
def get_angle(x1, y1, x2, y2):
  angle = atan2(y1 - y2, x2 - x1) * 180 / pi - 90
  if angle < 0:
    angle += 360
  return angle

def deg_to_item(l, deg):
  unit_size = 360 // len(l)
  offset = deg + unit_size/2

  sel = int(offset//unit_size)
  if sel >= len(l):
    sel = 0

  return l[sel]

def spiral_to_item(l, deg, loop):
  unit_size = 360 // 12
  deg += loop*360
  offset = deg + unit_size/2

  sel = int(offset//unit_size)

  sel = max(min(sel, len(l)-1), 0)

  return l[sel]
  
# --- misc ---
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
# this is specific to handling html ui
from browser import window
from browser.html import *

def bind(events, *elements):
  """ @bind('event event2', el, el) """
  def _bind(f):
    for event in events.split(' '):
      for element in elements:
        element.bind(event, f)
    return f
  return _bind
  
def bind_once(events, *elements):
  """ @bind_once('event event2', el, el) 
  
  This removes the binding once any event is fired on any of the elements.
  This can be used to stop doubleclicking causing errors.
  """
  def _bind(f):
    def _unbind(*l, **kw):
      for event in events.split(' '):
        for element in elements:
          element.unbind(event, _unbind)
      return f(*l, **kw)
        
    for event in events.split(' '):
      for element in elements:
        element.bind(event, _unbind)
    return f
    
  return _bind

def popup(root, width='auto', height='auto'):
  """ Popup dialog using <dialog>. """
  if isinstance(root, (tuple, list, str, int, float, set, dict)):
    if isinstance(root, str):
      root = DIV(root)
    else:
      root = DIV(repr(root))
    
  d = DIALOG(root)
  root.style['width'] = width
  root.style['height'] = height
      
  @bind('close', d)
  def _(ev):
    d.remove()
    cm_editbox.focus()
    
  doc <= d
  d.showModal()
  return d
  
class Popup:
  """ Context for popup function. """
  def __init__(self, *l, **kw):
    self.d = popup(*l, **kw)
    
  def __enter__(self):
    return self
    
  def __exit__(self, *l):
    self.d.close()
    
class El:
  """ Context to remove element on error. """
  def __init__(self, element, parent=None):
    self.element = element
    if parent:
      parent <= self.element
  
  def __enter__(self):
    return self.element
    
  def __exit__(self, *l):
    self.element.remove()
    
def box(*l, base=DIV, **kw):
  """ Simple box sized element meant to be freely positioned on the page. """
  sty = kw.get('style', {})
  sty['box-sizing'] = 'border-box'
  sty['display'] = 'inline'
  sty['position'] = 'absolute'
  kw['style'] = sty
  return base(*l, **kw)
  
def icon(_icon, *l, **kw):
  return I(_icon, *l, Class='material-icons', **kw)
  
def tile(text='', _icon='', title='', background='', 
    font_size='24px', icon_size='64px', align='center', base=DIV, **kw):
  sty = kw.get('style', {})
  sty['overflow'] = 'hidden'
  kw['style'] = sty
  
  cla = kw.get('Class', '').split(' ')
  cla += ['animate']
  kw['Class'] = ' '.join(cla)
  
  out = []
  out.append(DIV(style={
      'width': '100%',
      'height': '100%',
      'background': f'url({background})' or 'none',
      'background-size': 'cover',
      'background-position': 'center',
      #'z-index': '-1',
      'position': 'absolute',
      'left': 0,
      'top': 0,
      }))
      
  out.append(icon(_icon or '', 
    style={
      'position': 'absolute',
      'left': '50%',
      'top': '50%',
      'transform': 'translate(-50%, -50%)',
      'font-size': icon_size}))
      
  out.append(DIV(text or '', 
    style={
      'position': 'absolute',
      'left': '50%' if align=='center' else '0px',
      'top': '50%',
      'transform': 'translate(-50%, -50%)' if align=='center' else 'translate(0%, -50%)',
      'font-size': font_size,
      'text-align': align}))

  if text and _icon:
    out[-1].attrs['class'] = 'show_over animate'
    out[-2].attrs['class'] = 'hide_over material-icons animate'
  else:
    out[-1].attrs['class'] = 'animate'
    out[-2].attrs['class'] = 'material-icons animate'
    
  out.append(DIV(f'{title}' or '', 
    style={
      'position': 'absolute',
      'left': 0,
      'bottom': 0,
      'display': 'initial' if title else 'none',
      'background': 'rgba(0,0,0,.5)',
      'padding': '8px'}))
      
  return box(out, base=base, **kw)

def update_tile(instance, text='', _icon='', title='', background='', **kw):
  ui_background, ui_icon, ui_text, ui_title = instance.children
  
  ui_background.style['background-image'] = f'url({background})' if background else 'none'
  ui_icon.text = _icon or ''
  ui_text.text = text or ''
  
  if text and _icon:
    ui_text.attrs['class'] = 'show_over animate'
    ui_icon.attrs['class'] = 'hide_over material-icons animate'
  else:
    ui_text.attrs['class'] = 'animate'
    ui_icon.attrs['class'] = 'material-icons animate'
  
  ui_title.text = title or ''
  ui_title.style['display'] = 'initial' if title else 'none'
  
  if not any([text, _icon, title, title, background]):
    instance.style['display'] = 'none'
  else:
    instance.style['display'] = 'initial'

# update to handle new axis methods
coords_map = 'left', 'width', 'top', 'height'
def gmap(instance, *l, **kw):
  for k, v in zip(coords_map, (x for x in tracks_to_coords(*l, **kw) for x in x)):
    instance.style[k] = v
    
# --- 

def make_ripple(element, base_color=(18,18,18,1), no_key=False):
  """ Adds ripple click effect to an element. """
  mutator = tint if is_dark(*base_color) else shade
  
  hover_color = mutator(opacities[8], *base_color[:3], 1)
  pulse_color = mutator(.4, *base_color[:3], 1)
  
  glows = [(*(base_color[:3]), i) for i in (.14, .12, .2)]
  
  @bind('mouseout blur', element)
  def prep_ripple(*_):
    base_sty = {'background-color': f'rgba{base_color}', 
      'background-position': 'center', 
      'box-shadow': 'none',
      'transition': 'all 0.3s',
      'transition-timing-function': 'cubic-bezier(0.4, 0.0, 0.2, 1)'}
    for k, v in base_sty.items():
      element.style[k] = v
  prep_ripple()

  @bind('mouseover mouseup focus', element)
  def pre_ripple(*_):
    hover_sty = {'transition': 'all 0.3s',
      'box-shadow': f'0 4px 5px 0 rgba{glows[0]}, 0 1px 10px 0 rgba{glows[1]}, 0 2px 4px -1px rgba{glows[2]}',
      'background': f'rgba{hover_color} radial-gradient(circle, transparent 1%, rgba{hover_color} 1%) center/15000%'}

    for k, v in hover_sty.items():
      element.style[k] = v

  @bind('mousedown', element)
  def do_ripple(*_):
    active_sty = {'background-color': f'rgba{pulse_color}',
      #'box-shadow': f'0 4px 5px 0 rgba(220, 20, 60, 0.14), 0 1px 10px 0 rgba(220, 20, 60, 0.12), 0 2px 4px -1px rgba(220, 20, 60, 0.20)',
      'background-size': '100%', 'transition': 'all 0s'}

    for k, v in active_sty.items():
      element.style[k] = v
    
    Delay(.05, pre_ripple)

  # manually call a ripple pulse effect.
  def _do_ripple(*_):
    pre_ripple()
    Delay(.4, do_ripple)
    Delay(.45, pre_ripple)
    Delay(.6, prep_ripple)
  
  def _key_ripple(ev):
    if ev.key=='Enter' or ev.key==' ':
      do_ripple()
  
  if not no_key:
    element.bind('keydown', _key_ripple)
  
  return element, _do_ripple