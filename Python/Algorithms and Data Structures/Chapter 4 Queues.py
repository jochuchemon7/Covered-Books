# Basic Queue ADT class

class Queue:
    def __init__(self):
        self._qList = list()
    def isEmpty(self):
        return len(self._qList) == 0
    def __len__(self):
        return len(self._qList)
    def enqueue(self, item):
        self._qList.append(item)
    def dequeue(self):
        assert not self.isEmpty(), "Cannot dequeue from an empty queue"
        return  self._qList.pop(0)


# Array ADT class (needed for circular queue)
import ctypes

class Array:
    def __init__(self, size):
        assert size > 0, 'Size must be greater than 0'
        self._size = size
        PyArrayType = ctypes.py_object * size
        self._elements = PyArrayType()
        self.clear(None)
    def __len__(self):
        return self._size
    def __getitem__(self, index):
        assert index >= 0 and index < self._size, 'Index out of bounds'
        return self._elements[index]
    def __setitem__(self, index, value):
        assert index >= 0 and index < self._size, 'Index out of bounds'
        self._elements[index] = value
    def clear(self, value):
        for i in range(self._size):
            self._elements[i] = value
    def __iter__(self):
        return _ArrayIter(self._elements)

class _ArrayIter:
    def __init__(self, elements):
        self._arrayRef = elements
        self._curNd = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self._curNd < len(self._arrayRef):
            entry = self._arrayRef[self._curNd]
            self._curNd += 1
            return entry
        else:
            raise StopIteration


# __init, isEmpty, isFUll, __len, enqueue, dequeue

# Circular Queue
class Circular_Queue:
    def __init__(self, maxSize):
        self._count = 0
        self._front = 0
        self._back = maxSize - 1
        self._qArray = Array(maxSize)
    def isEmpty(self):
        return self._count == 0
    def isFull(self):
        return self._count == len(self._qArray)
    def __len__(self):
        return self._count
    def enqueue(self, value):
        assert not self.isFull(), 'You cannot add a value to a full Queue'
        maxSize = len(self._qArray)
        self._back = (self._back + 1) % maxSize
        self._qArray.__setitem__(self._back, value)
        self._count += 1
    def dequeue(self):
        assert not self.isEmpty(), 'Cannot remove from empty queue'
        item  = self._qArray[self._front]
        maxSize = len(self._qArray)
        self._front = (self._front + 1) % maxSize
        self._count -= 1
        return item

# Linked Listed Queue

class _QueueNode(object):
    def __init__(self, item):
        self.item = item
        self.next = None

class LinkListed_Queue:
    def __init__(self):
        self._qhead = None
        self._qtail = None
        self._count = 0
    def isEmpty(self):
        return self._qhead is None
    def __len__(self):
        return self._count
    def enqueue(self, value):
        node = _QueueNode(value)
        if self.isEmpty():
            self._qhead = node
        else:
            self._qtail.next = node
    def dequeue(self):
        assert not self.isEmpty(), 'Queue is empty!'
        node = self._qhead
        if self._qhead is self._qtail:
            self._qtail = None
        self._qhead = self._qhead.next
        self._count -= 1
        return node.item


# Unbounded Priority Queue

class _PriorityQEntry(object):
    def __init__(self, value, priority):
        self.value = value
        self.priority = priority

class PriorityQueue:
    def __init__(self):
        self._qList = list()
    def isEmpty(self):
        return len(self._qList) == 0
    def __len__(self):
        return len(self._qList)
    def enqueue(self, value, priority):
        newEntry = _PriorityQEntry(value, priority)
        self._qList.append(newEntry)
    def dequeue(self):
        assert not self.isEmpty(), 'Queue is Empty!'
        highest = self._qList[0].priority
        index = 0
        for i in range(len(self._qList)):
            if self._qList[i].priority < highest:
                highest = self._qList[i].priority
                index = i
        entry = self._qList.pop(index)
        return entry.value


# Bounded Priority Queue
class BPriorityQueue:
    def __init__(self, numLevels):
        self._qSize = 0
        self._qLevels = Array(numLevels)
        for i in range(numLevels):
            self._qLevels[i] = Queue()
    def isEmpty(self):
        return self._qSize == 0
    def __len__(self):
        return self._qSize
    def enqueue(self, value, priority):
        assert priority >= 0 and priority < len(self._qLevels), 'Invalid Priority level'
        self._qLevels[priority].enqueue(value)
    def dequeue(self):
        assert not self.isEmpty(), 'Queue is Empty!'
        i = 0
        p = len(self._qLevels)
        while i < p and not self._qLevels[i].isEmpty():
            i += 1
        return self._qLevels[i].dequeue()