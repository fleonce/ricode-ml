from typing import Any, Mapping, MutableMapping


class LazyMapping(Mapping[str, Any]):
    def __init__(self, data: MutableMapping[str, Any]):
        self.data = data
        self.keys_to_format = set(data.keys())

    def __getitem__(self, key, /):
        value = self.data[key]
        if key in self.keys_to_format:
            self.data[key] = value = value.as_py()
            self.keys_to_format.remove(key)
        return value

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
