from typing import Dict, Optional, TypeVar, Generic

T = TypeVar('T')

class BaseRegistry(Generic[T]):
    """Generic registry for components keyed by ``namespace:name``."""

    def __init__(self) -> None:
        self._items: Dict[str, T] = {}

    def register(self, item: T) -> None:
        """Register a new item."""
        self._items[item.full_name] = item

    def get(self, name: str, namespace: Optional[str] = None) -> T:
        """Retrieve an item by name and optional namespace."""
        # If name already contains a namespace, use it directly
        if ':' in name:
            if name not in self._items:
                raise KeyError(f"{name} not found")
            return self._items[name]

        full_name = f"{namespace or 'default'}:{name}"
        if full_name not in self._items:
            raise KeyError(f"{full_name} not found")
        return self._items[full_name]

    def list_items(self, namespace: Optional[str] = None) -> Dict[str, T]:
        """List registered items, optionally filtered by namespace."""
        if namespace:
            return {
                name.split(':', 1)[1]: item
                for name, item in self._items.items()
                if name.startswith(f"{namespace}:")
            }
        return self._items.copy()

    def list_namespaces(self) -> Dict[str, int]:
        """Return a mapping of namespaces to item counts."""
        namespaces: Dict[str, int] = {}
        for name in self._items.keys():
            ns = name.split(':', 1)[0]
            namespaces[ns] = namespaces.get(ns, 0) + 1
        return namespaces
