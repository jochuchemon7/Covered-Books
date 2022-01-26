# Binary Search Tree
import random
import ctypes
# Array ADT
class Array:
    def __init__(self, size):
        assert not size < 0, 'Array size must be bigger than 0'
        self.size = size
        PyArray = ctypes.py_object * size
        self.array = PyArray()
        self.clear(None)

    def __len__(self):
        return self.size
    def __setitem__(self, index, value):
        assert not index < 0 or index > len(self.array), "Index is out of bounds!"
        self.array[index] = value
    def __getitem__(self, index):
        assert not index < 0 or index > len(self.array), "Index is out of bounds!"
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

# Stack ADT
class Stack:
    def __init__(self):
        self.stack = list()
    def __len__(self):
        return len(self.stack)
    def isEmpty(self):
        return len(self.stack) == 0
    def peek(self):
        assert not self.isEmpty(), "Stack must have at least one value"
        return self.stack[-1]
    def push(self, value):
        self.stack.append(value)
    def pop(self):
        assert not self.isEmpty(), "Cannot Pop from Empty Stack"
        return self.stack.pop()


# BST Nodes
class _BSTMapNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None

# BSTMap First Implementation
class BSTMap:
    def __init__(self):
        self.root = None
        self.size = 0
    def __len__(self):
        return self.size
    def __iter__(self):
        return _BSTMapIterator(self.root)
    def __contains__(self, key):
        return self._bstSearch(self.root, key) is not None
    def valueOf(self, key):
        node = self._bstSearch(self.root, key)
        assert node is not None, "Invalid map key"
        return node.value
    def _bstSearch(self, subtree, target):
        if subtree is None:
            return None
        elif target < subtree.key:
            return self._bstSearch(subtree.left)
        elif target > subtree.key:
            return self._bstSearch(subtree.right)
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
            subtree = _BSTMapNode(key, value)
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
                if subtree.left is None:
                    return subtree.right
                else:
                    return subtree.left
            else:
                successor = self._bstMinnum(subtree.right)
                subtree.key = successor.key
                subtree.value = successor.value
                subtree.right = self._bstRemove(subtree.right, successor.key)
                return subtree

# Iterator Class using Array ADT
class _BSTMapIterator:
    def __init__(self, root, size):
        self.theKeys = Array(size)
        self.curItem = 0
        self._bstTraversal(root)
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
    def _bstTraversal(self, subtree):
        if subtree is not None:
            self._bstTraversal(subtree.left)
            self.theKeys[self.curItem] = subtree.key
            self.curItem += 1
            self._bstTraversal(subtree.right)

# Iterator Class using Stack ADT
class _BSTMapIteratorStack:
    def __init__(self, root):
        self.theStack = Stack()
        self._traverseToMinNode(root)
    def __iter__(self):
        return self
    def __next__(self):
        if self._theSack.isEmpty():
            raise StopIteration
        else:
            node = self.theStack.pop()
            key = node.key
            if node.right is not None:
                self._traverseToMinNode(node.right)
    def _traverseToMinNode(self, subtree):
        if subtree is not None:
            self.theStack.push(subtree)
            self._traverseToMinNode(subtree.left)


# Testing First BST Implementation

example = BSTMap()
example.__len__()
keys = [random.randint(0,100) for _ in range(10)]
values = ['a', 'b','c','d','e','f','g','h','i','j']


for i in range(len(keys)):
    example.add(keys[i], values[i])


# Binary Search Trees Second Implementation (Works)

class BSTNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None

# __len, __iter, __contains, valueOf, _bstSearch, _bstMinnum, _bstMaxnum, add, _bstInsert, remove, _bstRemove,

class BST:
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
    def _bstSearch(self, subtree, key):
        if subtree is None:
            return None
        elif key < subtree.key:
            return self._bstSearch(subtree.left, key)
        elif key > subtree.key:
            return self._bstSearch(subtree.right, key)
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
        assert key in self, 'Invalid Map Key'
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
    def accendingTraversal(self):
        return _BSTArrayTraversal(self.root, self.size, True)
    def deccendingTraversal(self):
        return _BSTArrayTraversal(self.root, self.size, False)

class _BSTArrayTraversal:
    def __init__(self, root, size, ascending):
        self.theKeys = Array(size)
        self.curItem = 0
        if ascending is True:
            self._ascendingTraversal(root)
        else:
            self._descendingTraversal(root)
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
    def _ascendingTraversal(self, subtree):
        if subtree is not None:
            self._ascendingTraversal(subtree.left)
            self.theKeys[self.curItem] = subtree.key
            self.curItem += 1
            self._ascendingTraversal(subtree.right)
    def _descendingTraversal(self, subtree):
        if subtree is not None:
            self._descendingTraversal(subtree.right)
            self.theKeys[self.curItem] = subtree.key
            self.curItem += 1
            self._descendingTraversal(subtree.left)

# Previous Ascending and Descending
"""
class _BSTArrayAccendingTraversal:
    def __init__(self, root, size):
        self.theKeys = Array(size)
        self.curItem = 0
        self._bstTraversal(root)
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
    def _bstTraversal(self, subtree):
        if subtree is not None:
            self._bstTraversal(subtree.left)
            self.theKeys[self.curItem] = subtree.key
            self.curItem += 1
            self._bstTraversal(subtree.right)

class _BSTArrayDeccendingTraversal:
    def __init__(self, root, size):
        self.theKeys = Array(size)
        self.curItem = 0
        self._bstTraversal(root)
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
    def _bstTraversal(self, subtree):
        if subtree is not None:
            self._bstTraversal(subtree.right)
            self.theKeys[self.curItem] = subtree.key
            self.curItem += 1
            self._bstTraversal(subtree.left)
"""

# __len, __iter, __contains, valueOf, _bstSearch, _bstMinnum, _bstMaxnum, add, _bstInsert, remove, _bstRemove,

keys = [random.randint(0,100) for _ in range(10)]
values = ['a', 'b','c','d','e','f','g','h','i','j']

example = BST(keys[0], values[0])
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

list(example.accendingTraversal())
list(example.deccendingTraversal())

example.remove(keys[0])
example.__len__()

[example.remove(keys[i]) for i in range(1,len(keys))]
example.__len__()


# AVL Trees


LEFT_HIGH = 1
EQUAL_HIGH = 0
RIGHT_HIGH = -1

# _AVLMapNode
class _AVLMapNode:
    def __init__(self, key, value):
        self.value = value
        self.key = key
        self.bfactor = EQUAL_HIGH
        self.right = None
        self.left = None

# AVLMap
class AVLMap:
    def __init__(self):
        self.root = None
        self.size = 0
    def __len__(self):
        return self.size
    def __contains__(self, key):
        return self._bstSearch(self.root,  key) is not None
    def valueOf(self, key):
        node = self._bstSearch(self.root, key)
        assert not node is None, "Invalid Map Key"
        return node.value
    def add(self, key, value):
        node = self._bstSearch(self.root, key)
        if node is not None:
            node.value = value
            return False
        else:
            (self.root, temp) = self.avlInsert(self.root, key, value)
            self.size += 1
            return True
    def remove(self, key):
        assert key in self, "Invalid Map Key"
        (self.root, temp) = self.avlRevemo(self.root, key)
        self.size -= 1
    def __iter__(self):
        return _BSTMapIterator(self.root)
    def _avlRotateRight(self, pivot):
        C = pivot.left
        pivot.left = C.right
        C.right = pivot
        return C
    def _avlRotateLeft(self, pivot):
        C = pivot.right
        pivot.right = C.left
        C.left = pivot
        return C
    def _avlLeftBalance(self, pivot):
        C = pivot.left
        if C.bfactor == LEFT_HIGH:
            pivot.bfactor = EQUAL_HIGH
            C.bfactor = EQUAL_HIGH
            pivot = self._avlRotateRight(pivot)
            return pivot
        else:
            if G.bfactor == LEFT_HIGH:
                pivot.bfactor = RIGHT_HIGH
                C.bfactor = EQUAL_HIGH
            elif G.bfactor == EQUAL_HIGH:
                pivot.bfactor = EQUAL_HIGH
                C.bfactor = EQUAL_HIGH
            else: # G.bfactor == RIGHT_HIGH
                pivot.bfactor = EQUAL_HIGH
                C.bfactor = LEFT_HIGH
            G.bfactor = EQUAL_HIGH

            pivot.left = self._avlRotateLeft(L)
            pivot = self._avlRotateRight(pivot)
            return pivot
    def _avlRightBalance(self, pivot):
        C = pivot.right
        if C.bfactor == RIGHT_HIGH:
            pivot.bfactor = EQUAL_HIGH
            C.bfactor = EQUAL_HIGH
            pivot = self._avlRotateLeft(pivot)
            return pivot
        else:
            if G.bfactor == RIGHT_HIGH:
                pivot.bfactor = LEFT_HIGH
                C.bfactor = EQUAL_HIGH
            elif G.bfactor == EQUAL_HIGH:
                pivot.bfactor = EQUAL_HIGH
                C.bfactor = EQUAL_HIGH
            else: # G.bfactor == LEFT_HIGH
                pivot.bfactor = EQUAL_HIGH
                C.bfactor = RIGHT_HIGH
            B.bfactor = EQUAL_HIGH

            pivot.left  = self._avlRotateRight(R)
            pivot = self._avlRotateLeft(pivot)
            return pivot
    def _avlInsert(self, subtree, key, newitem):
        if subtree is None:
            subtree = _AVLMapNode(key, newitem)
            taller = True
        elif key == subtree.key:
            return (subtree, False)
        elif key < subtree.key:
            (subtree, taller) = self._avlInsert(subtree.left, key, newitem)
            if taller:
                if subtree.bfactor == LEFT_HIGH:
                    subtree.right = self._avlLeftBalance(subtree)
                    taller = False
                elif subtree.bfactor == EQUAL_HIGH:
                    subtree.bfactor = LEFT_HIGH
                    taller = True
                else:
                    subtree.bfactor = EQUAL_HIGH
                    taller = False
        else:
            (subtree, taller) = self._avlInsert(subtree.right, key, newitem)
            if taller:
                if subtree.bfactor == LEFT_HIGH:
                    subtree.bfactor = EQUAL_HIGH
                    taller = False
                elif subtree.bfactor == EQUAL_HIGH:
                    subtree.bfactor = RIGHT_HIGH
                    taller = True
                else:
                    subtree.right = self._avlRighBalance(subtree)
                    taller = False
        return (subtree, taller)


# 2-3 Trees

# _23TreeNode
class _23TreeNode(object):
    def __init__(self, key, value):
        self.key1 = key
        self.key2 = None
        self.data1 = value
        self.data2 = None
        self.left = None
        self.middle = None
        self.right = None
    def isAleaf(self):
        return self.left is None and self.middle is None and self.right is None
    def isFull(self):
        return self.key2 is not None
    def hasKey(self, target):
        if (target == self.key1) or (self.key2 is not None and target == self.key2):
            return True
        else:
            return False
    def getData(self, target):
        if target == self.key1:
            return self.data1
        elif self.key2 is not None and target == self.key2:
            return self.data2
        else:
            return None
    def getBranch(self, target):
        if target < self.key1:
            return self.left
        elif self.key2 is None:
            return self.middle
        elif target < self.key2:
            return self.middle
        else:
            return self.right

# Tree23Map (Incomplete)