import 'dart:math';

/// Regex pattern to splice CSV text into a list.
final RegExp csvSplit = RegExp(r',(?=([^"]*"[^"]*")*[^"]*$)');

/// Return all regex matched strings.
List<String> reFindAll(String s, RegExp r) =>
    [for (var m in r.allMatches(s)) m.group(0) as String];

/// Terminal escapes to clear screen.
const String clear = '\x1B[2J\x1B[0;0H';

/// Print [user_string] at [x] and [y].
void lprint<T>(T userString, [int x = 0, int y = 0]) =>
    print('\x1B[${y + 1};${x + 1}H${userString.toString()}');

/// Read stream in, line by line.
//Stream<String> readLine(source) =>
//    source.transform(utf8.decoder).transform(const LineSplitter());

/// Sleepy time.
Future<void> sleep(int ms) => Future.delayed(Duration(milliseconds: ms));

/// A UUID generator.
///
/// This will generate unique IDs in the format:
///
///     f47ac10b-58cc-4372-a567-0e02b2c3d479
///
/// The generated uuids are 128 bit numbers encoded in a specific string format.
/// For more information, see
/// [en.wikipedia.org/wiki/Universally_unique_identifier](http://en.wikipedia.org/wiki/Universally_unique_identifier).
class Uuid {
  static final Random _random = Random();

  /// Generate a version 4 (random) uuid. This is a uuid scheme that only uses
  /// random numbers as the source of the generated uuid.
  static String generateV4() {
    // Generate xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx / 8-4-4-4-12.
    var special = 8 + _random.nextInt(4);

    return '${_bitsDigits(16, 4)}${_bitsDigits(16, 4)}-'
        '${_bitsDigits(16, 4)}-'
        '4${_bitsDigits(12, 3)}-'
        '${_printDigits(special, 1)}${_bitsDigits(12, 3)}-'
        '${_bitsDigits(16, 4)}${_bitsDigits(16, 4)}${_bitsDigits(16, 4)}';
  }

  static String _bitsDigits(int bitCount, int digitCount) =>
      _printDigits(_generateBits(bitCount), digitCount);

  static int _generateBits(int bitCount) => _random.nextInt(1 << bitCount);

  static String _printDigits(int value, int count) =>
      value.toRadixString(16).padLeft(count, '0');
}

/// Simple reference wrapper.
class Node<T> {
  T data;

  Node(this.data);

  String toString() => data.toString();
}

/// Base class for registering callbacks.
class Listenable {
  List<void Function()> _listeners = [];

  void add(void Function() f) => _listeners.add(f);

  void remove(void Function() f) => _listeners.remove(f);

  bool get hasListeners => _listeners.isNotEmpty;

  void dispose() => _listeners.clear();

  void notify() {
    for (var f in _listeners) f();
  }
}

/// Listenable property changes.
class ListenableValue<T> extends Listenable {
  T _value;

  ListenableValue(T value) : _value = value;

  T get value => _value;

  set value(T newValue) {
    if (newValue == _value) return;
    _value = newValue;
    notify();
  }
}

/// Break List into sublists of [n] length.
Iterable<List<E>> chunk<E>(List<E> l, int n) sync* {
  for (int i = 0; i < l.length ~/ n; i++) yield l.sublist(n * i, n * (i + 1));

  // Give remainder.
  if (l.length % n != 0) yield l.sublist(l.length - (l.length % n));
}

/// Group elements of [l] into a map by the results of [key].
Map<T, List<S>> groupBy<S, T>(Iterable<S> l, T Function(S) key) {
  Map<T, List<S>> m = {};
  for (var i in l) (m[key(i)] ??= []).add(i);
  return m;
}

/// Return like [groupBy] but is grouped by consecutive results from [key].
Iterable<MapEntry<T, List<S>>> chunkBy<S, T>(
    Iterable<S> l, T Function(S) key) sync* {
  T lastKey = key(l.first);
  T k = lastKey;
  List<S> lastGroup = [];

  for (S i in l) {
    k = key(i);
    if (k == lastKey) {
      lastGroup.add(i);
    } else {
      if (lastGroup.isNotEmpty) yield MapEntry(lastKey, lastGroup);
      lastKey = k;
      lastGroup = [i];
    }
  }
  if (lastGroup.isNotEmpty) yield MapEntry(lastKey, lastGroup);
}

/// Toggles an item in a list.
void toggleList<T>(List<T> l, T e) => l.contains(e) ? l.remove(e) : l.add(e);

/// Returns items of references from a map.
Iterable<V?> proxyList<K, V>(Map<K, V> m, Iterable<K> l) sync* {
  for (var i in l) yield m[i];
}

// math stuff

/// Degrees to radians.
double degreesToRads(double deg) => (deg * pi) / 180.0;

/// Radians to degrees.
double radsToDegrees(double rad) => (rad * 180.0) / pi;

/// Add rotate and angle to points.
extension on Point {
  Point<double> rotate(double t) =>
      Point(x * cos(t) - y * sin(t), x * sin(t) + y * cos(t));

  double angleTo(Point p) => atan2(p.y - y, p.x - x);
}

/// A utility class for representing two-dimensional positions.
///
/// This version is mutable with some extra functions.
class MutablePoint<T extends num> {
  T x, y;

  MutablePoint(this.x, this.y);

  String toString() => 'MutablePoint(x: $x, y: $y)';

  bool operator ==(Object other) =>
      other is MutablePoint && x == other.x && y == other.y;

  MutablePoint<T> operator +(MutablePoint<T> other) =>
      MutablePoint<T>((x + other.x) as T, (y + other.y) as T);

  MutablePoint<T> operator -(MutablePoint<T> other) =>
      MutablePoint<T>((x - other.x) as T, (y - other.y) as T);

  MutablePoint<T> operator *(num /*T|int*/ factor) =>
      MutablePoint<T>((x * factor) as T, (y * factor) as T);

  double get magnitude => sqrt(x * x + y * y);

  double distanceTo(MutablePoint<T> other) {
    var dx = x - other.x;
    var dy = y - other.y;
    return sqrt(dx * dx + dy * dy);
  }

  T squaredDistanceTo(MutablePoint<T> other) {
    var dx = x - other.x;
    var dy = y - other.y;
    return (dx * dx + dy * dy) as T;
  }

  MutablePoint<double> rotate(double t) =>
      MutablePoint(x * cos(t) - y * sin(t), x * sin(t) + y * cos(t));

  double angleTo(MutablePoint p) => atan2(p.y - y, p.x - x);
}

// animations

/// Simple lerp.
double lerp(num a, num b, double t) => a * (1.0 - t) + b * t;

/// Simple in out curve.
double inOutQuart(num a, num b, double t) {
  var p = t * 2;
  double _t = 0;
  if (p < 1) {
    _t = 0.5 * p * p * p * p;
  } else {
    p -= 2;
    _t = -0.5 * (p * p * p * p - 2.0);
  }
  return a * (1.0 - _t) + b * _t;
}

/// Animation driver.
class Animator {
  List<Animation> anims = [];
  List<Animation> toAdd = [];
  List<Animation> toRemove = [];

  Animation add(Animation a) {
    toAdd.add(a);
    return a;
  }

  void remove(Animation a) {
    toRemove.add(a);
  }

  void tick(int step) {
    for (var i in anims) {
      if (i.cancel) {
        toRemove.add(i);
        continue;
      }

      if (i.reverse)
        i.step -= step;
      else
        i.step += step;

      var t = i.step / i.duration;
      if (t > 1.0) {
        if (i.repeat) {
          t %= 1;
        } else if (i.repeatReverse) {
          i.reverse = !i.reverse;
          t = 1 - (t % 1);
        } else {
          t = 1.0;
          toRemove.add(i);
        }
      } else if (t < 0 && i.repeatReverse) {
        i.reverse = !i.reverse;
        t = -t;
      }
      i.f(t);
    }

    for (var i in toRemove) {
      anims.remove(i);
      i.onDone?.call();
    }

    toRemove.clear();

    for (var i in toAdd) anims.add(i);

    toAdd.clear();
  }
}

/// Animations to be driven.
class Animation {
  double duration;
  double step = 0.0;
  void Function(double t) f;
  bool cancel = false;
  bool reverse = false;
  bool repeat;
  bool repeatReverse;
  void Function()? onDone;

  Animation(this.duration, this.f,
      {this.repeat = false, this.repeatReverse = false});
}

/// Main loop driver.
///
/// Provides the difference in time between calls.
Future<void> mainloop(int delay, bool Function(int) f) async {
  int stamp = DateTime.now().millisecondsSinceEpoch, last = stamp, diff = 0;
  while (true) {
    stamp = DateTime.now().millisecondsSinceEpoch;
    diff = stamp - last;
    last = stamp;
    // if f returns false, fall out of loop.
    if (!f(diff)) break;
    await sleep(-stamp % delay);
  }
}

// ecs

/// Entity Component System.
///
/// Archetypes are groups of entities that have a specified set of components.
/// These are what systems loop though.
class ECS<U> {
  Map<Type, ComponentArray> componentArrays = {};
  Map<U, Set<Type>> entitySignature = {};
  Map<Set<Type>, List<U>> archetypes = {};

  /// Id generator. This can be UUID or random/sequential ints, etc.
  U Function() makeId;

  /// Id disposer.
  void Function(U id) disposeID;

  ECS(this.makeId, this.disposeID);

  /// Short hand to create uuid and attach components.
  ///
  /// If the ComponentArray isn't already cached, the array will have a dynamic
  /// data type due to runtimeType not being usable in generics.
  U entity([List components = const []]) {
    var u = makeId();
    for (var c in components) {
      var T = c.runtimeType;
      (componentArrays[T] ??= ComponentArray<dynamic, U>()).items[u] = c;
      (entitySignature[u] ??= {}).add(T);
    }
    updateArchetypes(u);
    return u;
  }

  /// Dispose entity.
  void dispose(U u) {
    for (var T in entitySignature[u] ?? {}) componentArrays[T]!.items.remove(u);
    entitySignature.remove(u);
    updateArchetypes(u);
    disposeID(u);
  }

  /// Add Component to entity and update signature and archetypes.
  void addComponent<T>(U u, T c) {
    (componentArrays[T] ??= ComponentArray<T, U>()).items[u] = c;
    (entitySignature[u] ??= {}).add(c.runtimeType);
    updateArchetypes(u);
  }

  /// Opposite of addComponent.
  T? removeComponent<T>(U u) {
    T? _comp = componentArrays[T]?.items.remove(u);
    entitySignature[u]?.remove(T);
    updateArchetypes(u);
    return _comp;
  }

  /// Update archetypes.
  void updateArchetypes(U u) {
    var signature = entitySignature[u] ?? {};
    for (var e in archetypes.entries)
      if (signature.containsAll(e.key)) {
        if (!e.value.contains(u)) e.value.add(u);
      } else {
        if (e.value.contains(u)) e.value.remove(u);
      }
  }

  /// Hard fetch for component.
  T call<T>(U u) => componentArrays[T]!.items[u]!;

  /// Soft fetch for component. Will return null of missing type or id.
  T? maybe<T>(U u) => componentArrays[T]?.items[u];

  /// Returns list of ids that match [types].
  ///
  /// This will create the archetype of it doesn't already exist.
  /// example: `for (var e in ecs.of(const {type, type2}))`
  List<U> of(Set<Type> types) => archetypes[types] ??= [
        for (var e in entitySignature.entries)
          if (e.value.containsAll(types)) e.key
      ];
}

class ComponentArray<T, U> {
  Map<U, T> items = {};

  String toString() => 'ComponentArray<$T, $U>(items: $items)';
}
