import 'dart:collection';

/// Base class for registering callbacks.
class Listenable {
  List<void Function()> _listeners = [];

  void add(void Function() f) => _listeners.add(f);
  void remove(void Function() f) => _listeners.remove(f);
  bool get hasListeners => _listeners.length != 0;
  void dispose() => _listeners.clear();

  void notify() {
    for (void Function() f in _listeners) f();
  }
}

class ListenableValue<T> extends Listenable {
  T _value;
  ListenableValue(T value) : _value = value;
  String toString() => value.toString();

  T get value => _value;
  set value(T newValue) {
    if (newValue != _value) {
      _value = newValue;
      notify();
    }
  }
}

/// Call [update] and supply the time spent between runs and return the time until next cycle.
Future<void> aloop(int Function(int) update) async {
  Stopwatch sw = Stopwatch()..start();
  int d = 0;

  while (true) {
    d = update(sw.elapsedMilliseconds);
    if (d == -1) break;
    sw.reset();
    await Future.delayed(Duration(milliseconds: d));
  }
}

/// Simple component to add an arbitrary string to an entity.
class Tag {
  String tag;
  Tag([this.tag = '']);

  Tag.fromJson(Map<String, dynamic> json) : tag = json['tag'];
  Map<String, dynamic> toJson() => {'tag': tag};

  String toString() => 'Tag(tag: $tag)';
}

/// Simple Entity Component System.
class ECS {
  int entityIndex = -1;
  List<int> freeEntities = [];
  HashMap<int, Set<Type>> entitySignature = HashMap();
  int get entitiesAlive => entitySignature.length;

  HashMap<Type, ComponentArray> components = HashMap();

  List<System> systems = [];

  /// Creates and adds [components] to related ComponentArrays and systems.
  int createEntity([List<dynamic> components = const <dynamic>[]]) {
    int e = freeEntities.isEmpty ? ++entityIndex : freeEntities.removeAt(0);

    if (components.length > 0) {
      Set<Type> signature = {for (var c in components) c.runtimeType};
      entitySignature[e] = signature;
      for (var c in components) {
        assert(this.components.keys.contains(c.runtimeType),
            'Component ${c.runtimeType} not registered');
        (this.components[c.runtimeType])!.add(e, c);
      }

      for (System s in systems)
        if (signature.containsAll(s.signature)) s.entities.add(e);
    }

    return e;
  }

  /// Remove entity from arrays and systems.
  void disposeEntity(int e) {
    for (Type T in entitySignature[e]!) this.components[T]!.remove(e);
    for (System s in systems) s.entities.remove(e);
    entitySignature.remove(e);
    freeEntities.add(e);
  }

  // ---

  /// Add [ComponentArray].
  void addComponent<T>() => components[T] = ComponentArray<T>();

  /// Add a single [T] component to [entity].
  void addEntityComponent<T>(int entity, T component) {
    Set<Type> signature = entitySignature[entity]!;
    signature.add(T);
    updateSignature(entity, signature);
    components[T]!.add(entity, component);
  }

  /// Remove component from array.
  T removeEntityComponent<T>(int entity) {
    Set<Type> signature = entitySignature[entity]!;
    signature.remove(T);
    updateSignature(entity, signature);
    return components[T]!.remove(entity);
  }

  /// Get single component for entity.
  T getEntityComponent<T>(int entity) => components[T]!.get(entity);

  /// Same as [getEntityComponent].
  T call<T>(int entity) => components[T]!.get(entity);

  /// Reverse lookup an [entity] by a [component].
  int getEntityByComponent<T>(T component) {
    return components[T]!
            .indexToEntity[components[T]!.items.indexOf(component)] ??
        -1;
  }

  /// Get all components of entity by [component].
  List<dynamic> getEntityComponentsByComponent<T>(T component) =>
      getEntityComponents(getEntityByComponent(component));

  /// Get list of components for [entity].
  List<dynamic> getEntityComponents(int entity) => <dynamic>[
        for (Type T in entitySignature[entity]!) components[T]!.get(entity)
      ];

  // ---

  /// Add system for [signature] typed entities.
  ///
  /// If no function is supplied, then the system is disabled and just acts as a list.
  /// Returning the list directly can allow for a faster loop to be written outside of the [ECS.update].
  List<int> addSystem(Set<Type> signature, [void Function(List<int>)? f]) {
    System s = System(signature, f);
    systems.add(s);
    return s.entities;
  }

  /// Call all system updates to entities.
  void update() {
    for (System s in systems) if (s.f != null) s.f!(s.entities);
  }

  /// Update [signature] for [entity].
  ///
  /// This will also add/remove entities from the relevant systems too.
  void updateSignature(int entity, Set<Type> signature) {
    for (System s in systems)
      if (signature.containsAll(s.signature)) {
        if (!s.entities.contains(entity)) s.entities.add(entity);
      } else
        s.entities.remove(entity);
  }

  void load<T>(Map<String, dynamic> json, Function f) {
    components[T]!.fromJson(json, f);
    for (var i in components[T]!.entityToIndex.keys)
      (entitySignature[i] ??= {}).add(T);
  }
  //Map<String, dynamic> toJson<T>() => components[T]!.toJson();

  //ECS entityIndex freeEntities components
  fromJson(Map<String, dynamic> json) {
    entityIndex = json['entityIndex'];
    freeEntities = json['freeEntities'].cast<int>();
  }

  Map<String, dynamic> toJson() => {
        'entityIndex': entityIndex,
        'freeEntities': freeEntities,
        'components': {
          for (var i in components.entries) i.key.toString(): i.value.toJson()
        }
      };

  void resyncSystems() {
    for (var i in systems) i.entities.clear();
    for (var e in entitySignature.entries) updateSignature(e.key, e.value);
  }

  toString() =>
      'ECS(\n  entitiesAlive: $entitiesAlive, \n  components: $components, \n  systems: $systems\n)';
}

/// Struct for systems.
class System {
  Set<Type> signature;
  void Function(List<int>)? f;
  List<int> entities = [];

  System(this.signature, this.f);

  toString() =>
      'System(\n    f: $f, \n    signature: $signature, \n    size: ${entities.length}\n  )';
}

class ComponentArray<T> {
  int get size => items.length;

  /// What is in the array.
  List<T> items = [];

  /// Bidirectional mapping to look up specific instances of components.
  HashMap<int, int> entityToIndex = HashMap();
  HashMap<int, int> indexToEntity = HashMap();

  /// Add a component and associate it with an Entity ID.
  void add(int entity, T component) {
    int newIndex = size;

    entityToIndex[entity] = newIndex;
    indexToEntity[newIndex] = entity;

    //items[newIndex] = component;
    items.add(component);
  }

  /// Remove and re-pack component array.
  ///
  /// This actually takes the last item in the array, moves it to the deleted item slot,
  /// then updates the index and entity maps.
  T remove(int entity) {
    // Copy element at end into deleted element's place to maintain density
    int indexOfRemovedEntity = entityToIndex[entity]!;
    int indexOfLastElement = size - 1;
    T temp = items[indexOfRemovedEntity] as T;
    items[indexOfRemovedEntity] = items[indexOfLastElement];

    // Update map to point to moved spot
    int entityOfLastElement = indexToEntity[indexOfLastElement]!;
    entityToIndex[entityOfLastElement] = indexOfRemovedEntity;
    indexToEntity[indexOfRemovedEntity] = entityOfLastElement;

    entityToIndex.remove(entity);
    indexToEntity.remove(indexOfLastElement);

    items.removeAt(indexOfLastElement);

    return temp;
  }

  /// Get data by Entity ID.
  T get(int entity) => items[entityToIndex[entity]!];

  toString() => '\n    ComponentArray<$T>(size: $size)\n  ';

  void fromJson(Map<String, dynamic> json, Function f) {
    items = <T>[for (var i in json['items']) f(i)];
    indexToEntity = HashMap.from({
      for (var e in json['indexToEntity'].entries) int.parse(e.key): e.value
    });
    entityToIndex = HashMap.from({
      for (var e in json['entityToIndex'].entries) int.parse(e.key): e.value
    });
  }

  Map<String, dynamic> toJson() => {
        'items': items,
        'entityToIndex': {
          for (var e in entityToIndex.entries) e.key.toString(): e.value
        },
        'indexToEntity': {
          for (var e in indexToEntity.entries) e.key.toString(): e.value
        }
      };
}
