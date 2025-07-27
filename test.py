from typing import TypedDict

class ParentDict(TypedDict):
    key1: str
    key2: int

d = ParentDict(key1="hello", key2="bye")

print(d)

print(d["key1"])
print(d["key2"])