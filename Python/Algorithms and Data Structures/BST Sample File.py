import random
import ctypes
class Array:
    def __init__(self, size):
        assert size > 0, "Invalid Size"
        PyArray = ctypes.py_object * size
        self.array = PyArray()
        self.size = size
        self.clear(None)
    def __len__(self):
        return self.size
    def __setitem__(self, index, value):
        assert not index > len(self.array) or index < 0, "Index Out of Bounds"
        self.array[index] = value
    def __getitem__(self, index):
        assert not index < 0 or index > len(self.array), "Index Out of Bounds"
        return self.array[index]
    def clear(self, value):
        for i in range(len(self.array)):
            self.array[i] = value
    def __iter__(self):
        return _ArrayIterator(self.array)
class _ArrayIterator:
    def __init__(self, array):
        self.arrayRef = array
        self.curNd = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.curNd < len(self.arrayRef):
            item = self.arrayRef[self.curNd]
            self.curNd += 1
            return item
        else:
            raise StopIteration



class BSTNode:
    def __init__(self, key, value):
        self.value = value
        self.key = key
        self.left = None
        self.right = None
        
# __len, __iter, __contains, valueOf, _bstSearch, _bstMinnum, _bstMaxnum,
# add, _bstInsert, remove, _bstRemove, ascendTraverse, descendTraverse

class BSTMap:
    def __init__(self, key, value):
        self.root = BSTNode(key, value)
        self.size = 1
    def __len__(self):
        return self.size
    def __contains__(self, key):
        return self._bstSearch(self.root, key) is not None
    def valueOf(self, key):
        node = self._bstSearch(self.root, key)
        assert not node is None, "Invalid Map Key"
        return node.value
    def _bstSearch(self, subtree, target):
        if subtree is None:
            return None
        elif target < subtree.key:
            return self._bstSearch(subtree.left, target)
        elif target > subtree.key:
            return self._bstSearch(subtree.right, target)
        else:
            return subtree
    def _bstMinnum(self, subtree):
        if subtree is None:
            return None
        elif subtree.left is None:
            return subtree
        else:
            return self._bstMinnum(subtree.left)
    def _bstMaxnum(self, subtree):
        if subtree is None:
            return None
        elif subtree.right is None:
            return subtree
        else:
            return self._bstMaxnum(subtree.right)
    def add(self, key, value):
        node = self._bstSearch(self.root, key)
        if node is not None:
            node.value = value
            return False
        else:
            self.root = self._bstInsert(self.root, key, value)
            self.size += 1
            return True
    def _bstInsert(self, subtree, key, value):
        if subtree is None:
            subtree = BSTNode(key, value)
        elif key < subtree.key:
            subtree.left = self._bstInsert(subtree.left, key, value)
        elif key > subtree.key:
            subtree.right = self._bstInsert(subtree.right, key, value)
        return subtree
    def remove(self, key):
        assert key in self, "Invalid Map Key"
        self.root = self._bstRemove(self.root, key)
        self.size -= 1
    def _bstRemove(self, subtree, target):
        if subtree is None:
            return subtree
        elif target < subtree.key:
            subtree.left = self._bstRemove(subtree.left, target)
            return subtree
        elif target > subtree.key:
            subtree.right = self._bstRemove(subtree.right, target)
            return subtree
        else:
            if subtree.left is None and subtree.right is None:
                return None
            elif subtree.left is None or subtree.right is None:
                if subtree.left is not None:
                    return subtree.left
                else:
                    return subtree.right
            else:
                successor = self._bstMinnum(subtree.right)
                subtree.key = successor.key
                subtree.value = successor.value
                subtree.right = self._bstRemove(subtree.right, successor.key)
                return subtree
    def ascendingTraversal(self):
        return BSTArrayTraversal(self.root, self.size, True)
    def descendingTraversal(self):
        return BSTArrayTraversal(self.root, self.size, False)

class BSTArrayTraversal:
    def __init__(self, root, size, ascending):
        self.theKeys = Array(size)
        self.curItem = 0
        if ascending is True:
            self._ascending(root)
        else:
            self._descending(root)
        self.curItem = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.curItem < len(self.theKeys):
            key = self.theKeys[self.curItem]
            self.curItem += 1
            return key
        else:
            raise StopIteration
    def _ascending(self, subtree):
        if subtree is not None:
            self._ascending(subtree.left)
            self.theKeys[self.curItem] = subtree.key
            self.curItem += 1
            self._ascending(subtree.right)
    def _descending(self, subtree):
        if subtree is not None:
            self._descending(subtree.right)
            self.theKeys[self.curItem] = subtree.key
            self.curItem += 1
            self._descending(subtree.left)


# Testing

keys = [random.randint(0,100) for _ in range(10)]
values = ['a', 'b','c','d','e','f','g','h','i','j']

example = BSTMap(keys[0], values[0])
example.__len__()
example.__contains__(keys[0])
example.valueOf(keys[0])

node = example._bstSearch(example.root, keys[0])
print('Search Example -> Key: %d  Value: %s' % (node.key, node.value))
minNode = example._bstMinnum(example.root)
print('MinNode Example -> Key: %d  Value: %s' % (minNode.key, minNode.value))
maxNode = example._bstMaxnum(example.root)
print('MaxNode Example -> Key: %d  Value: %s' % (maxNode.key, maxNode.value))

example.add(keys[1], values[1])
example.__len__()
example.__contains__(keys[1])
example.valueOf(keys[0])

for i in range(2, len(values)):
    example.add(keys[i], values[i])
example.__len__()
print(example._bstMinnum(example.root).key)
print(example._bstMaxnum(example.root).key)

list(example.ascendingTraversal())
list(example.descendingTraversal())

example.remove(keys[0])
example.__len__()

[example.remove(keys[i]) for i in range(1,len(keys))]
example.__len__()
