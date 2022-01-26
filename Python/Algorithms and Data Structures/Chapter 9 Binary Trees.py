import random


class _BinTreeNode:
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None

def preOrderTraversal(subTree):
    if subTree is not None:
        print(subTree.data)
        preOrderTraversal(subTree.left)
        preOrderTraversal(subTree.right)

def inOrderTraversal(subTree):
    if subTree is not None:
        inOrderTraversal(subTree.left)
        print(subTree.data)
        inOrderTraversal(subTree.right)

def postOrderTraversal(subTree):
    if subTree is not None:
        postOrderTraversal(subTree.left)
        postOrderTraversal(subTree.right)
        print(subTree.data)

# BreadthFirstTraversal

# Queue ADT
class Queue:
    def __init__(self):
        self.queue = list()
    def __len__(self):
        return len(self)
    def isEmpty(self):
        return len(self) == 0
    def enqueue(self, value):
        self.queue.append(value)
    def dequeue(self):
        return self.queue.pop(0)
    def view(self):
        i = 0
        sol = []
        while i < len(self.queue):
            sol.append(self.queue[i])
            i += 1
        print(sol)
        return

def breadthFirstTraversal(subTree):
    q = Queue()
    q.enqueue(subTree)

    while not q.isEmpty():
        node = q.dequeue()
        print(node.data)

        if node.left is not None:
            q.enqueue(node.left)
        if node.right is not None:
            q.enqueue(node.right)


# Binary Tree ADT (First Implementation)

# __init, insert, preTraversal, inTraversal, postTraversal, breadthFirstTraversal

class binaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def insert(self, root, value):
        if root is None:
            root = binaryTreeNode(value)
            return root
        tree = []
        tree.append(root)
        while len(tree):
            root = tree[0]
            tree.pop(0)
            if not root.left:
                root.left = binaryTreeNode(value)
                break
            else:
                tree.append(root.left)
            if not root.right:
                root.right = binaryTreeNode(value)
                break
            else:
                tree.append(root.right)
        return tree
    def preOrderTraversal(self):
        elements = [self.data]
        if self.left:
            elements += self.left.preOrderTraversal()
        if self.right:
            elements += self.right.preOrderTraversal()
        return elements
    def inOrderTraversal(self):
        elements = []
        if self.left:
            elements += self.left.inOrderTraversal()
        elements.append(self.data)
        if self.right:
            elements += self.right.inOrderTraversal()
        return elements
    def postOrderTraversal(self):
        elements = []
        if self.left:
            elements += self.left.postOrderTraversal()
        if self.right:
            elements += self.right.postOrderTraversal()
        elements.append(self.data)
        return elements
    def breathFirstTraversal(self, subtree):
        tree = []
        tree.append(subtree)
        elements = []
        while len(tree):
            node = tree.pop(0)
            elements.append(node.data)

            if node.left is not None:
                tree.append(node.left)
            if node.right is not None:
                tree.append(node.right)
        print(elements)


values = [15, 10, 4, 12, 35, 90, 25, 30]
treeExample = binaryTreeNode(values[0])
for i in range(1, len(values)):
    treeExample.insert(treeExample, values[i])
treeExample.preOrderTraversal()
treeExample.inOrderTraversal()
treeExample.postOrderTraversal()

treeExample.insert(treeExample, 31)
treeExample.postOrderTraversal()

treeExample.breathFirstTraversal(treeExample)


# Binary Search Tree ADT (First Implementation)

# __init, insert, preTraversal, inTraversal, postTraversal, breadthFirstTraversal

class binarySearchTree:
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None
    def insert(self, root, value):
        if root is None:
            root = binarySearchTree(value)
            return root
        if value < root.data:
            root.left = self.insert(root.left, value)
        else:
            root.right = self.insert(root.right, value)
        return root
    def preOrderTraversal(self):
        elements = [self.data]
        if self.left:
            elements += self.left.preOrderTraversal()
        if self.right:
            elements += self.right.preOrderTraversal()
        return elements
    def inOrderTraversal(self):
        elements = []
        if self.left:
            elements += self.left.inOrderTraversal()
        elements.append(self.data)
        if self.right:
            elements += self.right.inOrderTraversal()
        return elements
    def postOrderTraversal(self):
        elements = []
        if self.left:
            elements += self.left.postOrderTraversal()
        if self.right:
            elements += self.right.postOrderTraversal()
        return elements
    def breathFirstTraversal(self, subtree):
        tree = []
        tree.append(subtree)
        elements = []
        while len(tree):
            node = tree.pop(0)
            elements.append(node.data)

            if node.left is not None:
                tree.append(node.left)
            if node.right is not None:
                tree.append(node.right)
        print(elements)
    def preOrderNoRecursion(self):
        elements = []
        stack = [self]
        while stack:
            current = stack.pop()
            elements.append(current.data)
            if current.right:
                stack.append(current.right)
            if current.left:
                stack.append(current.left)
        return elements
    def inOrderNoRecursion(self):
        elements = []
        stack = []
        current = self
        while current or stack:
            if current:
                stack.append(current)
                current = current.left
            else:
                current = stack.pop()
                elements.append(current.data)
                current = current.right
        return elements
    def postOrderNoRecursion(self):
        elements = []
        stack = [self]
        out = []
        while stack:
            current = stack.pop()
            out.append(current.data)
            if current.left:
                stack.append(current.left)
            if current.right:
                stack.append(current.right)
        while out:
            elements.append(out.pop())
        return elements




newValues = [15, 10, 4, 12, 35, 90, 25, 30]
treeSearchExample = binarySearchTree(newValues[0])
for i in range(1, len(newValues)):
    treeSearchExample.insert(treeSearchExample, newValues[i])
treeSearchExample.preOrderTraversal()
treeSearchExample.inOrderTraversal()
treeSearchExample.postOrderTraversal()
treeSearchExample.breathFirstTraversal()

# Binary Tree Works
"""


# Second BT Implementation

class binaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
    def insert(self, root, value):
        if root is None:
            root = binaryTreeNode(value)
            return root
        if value < root.data:
            root.left = self.insert(root.left, value)
        else:
            root.right = self.insert(root.right, value)
        return root
    def normalInsert(self, root, value):
        if root is None:
            root = binaryTreeNode(value)
            return root
        q = []
        q.append(root)
        while len(q):
            root = q[0]
            q.pop(0)
            if not root.left:
                root.left = binaryTreeNode(value)
                break
            else:
                q.append(root.left)
            if not root.right:
                root.right = binaryTreeNode(value)
                break
            else:
                q.append(root.right)
        return q
    def preOrderTraversal(self):
        elements = [self.data]
        if self.left:
            elements += self.left.preOrderTraversal()
        if self.right:
            elements += self.right.preOrderTraversal()
        return elements
    def inOrderTraversal(self):
        elements = []
        if self.leff:
            elements += self.left.inOrderTraversal()
        elements.append(self.data)
        if self.right:
            elements += self.right.inOrderTraversal()
        return elements

    def postOrderTraversal(self):
        elements = []
        if self.left:
            elements += self.left.postOrderTraversal()
        if self.right:
            elements += self.right.postOrderTraversal()
        elements.append(self.data)
        return elements

tree_elements = [15, 10, 4, 12, 35, 90, 25, 30]
newRoot = binaryTreeNode(tree_elements[0])
#for i in range(1, len(tree_elements)):
#    newRoot.insert(newRoot, tree_elements[i])

for i in range(1, len(tree_elements)):
    newRoot.normalInsert(newRoot, tree_elements[i])

newRoot.preOrderTraversal()
newRoot.inOrderTraversal(newRoot)
newRoot.postOrderTraversal(newRoot)

# Website Implementation
class BinarySearchTreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    # Adding nodes otherwise called insertion
    def insert(self, data):
        # Binary Search Tree cannot have duplicate values condition
        if data == self.data:
            return
        if data < self.data:
            # Check if there is a value in the left node,then
            # call the method recursively
            if self.left:
                self.left.insert(data)
            else:
                self.left = BinarySearchTreeNode(data)
        else:
            # Check if there is a value in the right node,then
            # call the method recursively
            if self.right:
                self.right.insert(data)
            else:
                self.right = BinarySearchTreeNode(data)

    def inorder_traversal(self):
        # Need to return the elements in an order Left,Right,Root
        # Create a list for elements to store retrieved values
        elements = []
        # visit Left Tree first
        if self.left:
            elements += self.left.inorder_traversal()
        # visit root node
        elements.append(self.data)
        # visit right Tree
        if self.right:
            elements += self.right.inorder_traversal()

        return elements

    def pre_order_traversal(self):
        # Need to return the elements in an order Root, Left, Right
        # Create a list of elements to store the retrieved data
        elements = [self.data]
        if self.left:
            elements += self.left.pre_order_traversal()
        if self.right:
            elements += self.right.pre_order_traversal()
        return elements

    def postorder_traversal(self):
        # Need to return the elements in an order Left,Right,Root
        elements = []
        if self.left:
            elements += self.left.postorder_traversal()
        if self.right:
            elements += self.right.postorder_traversal()
        elements.append(self.data)
        return elements

tree_elements = [15, 10, 4, 12, 35, 90, 25, 30]
root = BinarySearchTreeNode(tree_elements[0])
for i in range(1, len(tree_elements)):
    root.insert(tree_elements[i])
root.pre_order_traversal()
root.inorder_traversal()
root.postorder_traversal()

"""

# Expression Tree

class ExpressionTree:
    def __init__(self, expStr):
        self._expTree = None
        self._buildTree(expStr)
    def evaluate(self, varMap):
        return self._evalTree(self._expTree, varMap)
    def __str__(self):
        return self._buildString(self._expTree)
    def _buildString(self, treeNode):
        if treeNode.left is None and treeNode.right is None:
            return str(treeNode.data)
        else:
            expStr = '('
            expStr += self._buildString(treeNode.left)
            expStr += str(treeNode.data)
            expStr += self._buildString(treeNode.right)
            expStr += ')'
            return expStr
    def _evalTree(self, subtree, varDict):
        if subtree.left is None and subtree.right is None:
            if subtree.data >= '0' and subtree.data <= '9':
                return int(subtree.data)
            else:
                assert subtree.data in varDict, 'Invalid Variable'
                return varDict[subtree.data]
        else:
            lvalue = self._evalTree(subtree.left, varDict)
            rvalue = self._evalTree(subtree.right, varDict)
            return self.computeOP(lvalue, subtree.data, rvalue)
    def computeOP(self, left, op, right):
        return 1
    def _buildTree(self, expStr):
        expQ = Queue()
        for token in expStr:
            expQ.enqueue(token)
        self._expTree = _ExpTreeNode(None)
        self._recBuildTree(self._expTree, expQ)
    def _recBuildTree(self, curNode, expQ):
        token = expQ.dequeue()
        if token == '(':
            curNode.left = _ExpTreeNode(None)
            self._recBuildTree(curNode, expQ)
            curNode.data = expQ.dequeue()

            curNode.right = _ExpTreeNode(None)
            self._recBuildTree(curNode, expQ)
            expQ.dequeue()
        else:
            curNode.data = token

class _ExpTreeNode: # Storage Node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None



# Heaps (Max-Heap)

# Array ADT

# __init, __len, __getitem, __setitem, clear, __iter
import ctypes

class Array:
    def __init__(self, size):
        assert size > 0, "Size must be bigger than 0"
        self._size = size
        PyArray = ctypes.py_object * size
        self.array = PyArray()
        self.clear(None)
    def __len__(self):
        return self._size
    def __getitem__(self, index):
        assert not index > self._size or index < 0, "Aray out of bounds"
        return self.array[index]
    def __setitem__(self, index, value):
        assert not index > self._size or index < 0, "Array out of bounds"
        self.array[index] = value
    def clear(self, value):
        for i in range(self._size):
            self.array[i] = value
    def __iter__(self):
        return _ArrayIterator(self.array)
class _ArrayIterator:
    def __init__(self, array):
        self.ArrayRef = array
        self._curNode = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self._curNode < len(self.ArrayRef):
            entry = self.ArrayRef[self._curNode]
            self._curNode += 1
            return entry
        else:
            raise StopIteration

# __init, __len, capacity, add, extract, _siftUp, _siftDown,

"""
Given index i
PARENT: (i-1) // 2
LEFT: 2 * i + 1
RIGHT: 2 * i + 2
"""

class MaxHeap:
    def __init__(self, size):
        self.elements = Array(size)
        self.count = 0
    def __len__(self):
        return self.count
    def capacity(self):
        return len(self.elements)
    def add(self, value):
        assert self.count < self.capacity(), 'Cannot add to full heap'
        self.elements[self.count] = value
        self.count += 1
        self._siftUp(self.count-1)
    def extract(self):
        assert not self.count < 0, 'Cannot Extract from an empty heap'
        value = self.elements[0]
        self.count -= 1
        self.elements[0] = self.elements[self.count]
        self._siftDown(0)
        return value
    def _siftUp(self, ndx):
        if ndx > 0:
            parent = ndx // 2
            if self.elements[ndx] > self.elements[parent]:
                temp = self.elements[ndx]
                self.elements[ndx] = self.elements[parent]
                self.elements[parent] = temp
                self._siftUp(parent)
    def _siftDown(self, ndx):
        left = 2 * ndx + 1
        right = 2 * ndx + 2

        largest = ndx
        if left < self.count and self.elements[left] >= self.elements[largest]:
            largest = left
        elif right < self.count and self.elements[right] >= self.elements[largest]:
            largest = right
        if largest != ndx:
            self.elements[ndx], self.elements[largest] = self.elements[largest], self.elements[ndx]
            self._siftDown(largest)

def simpleHeapSort(theSeq):
    n = len(theSeq)
    heap = MaxHeap(n)

    for item in theSeq:
        heap.add(item)

    for i in range(n-1,0,-1):
        theSeq[i] = heap.extract()

    return theSeq


example = MaxHeap(12)
example.__len__()
example.capacity()
values = [random.randint(0,25) for _ in range(12)]
for i in range(len(values)):
    example.add(values[i])
example.__len__()
values
list(example.elements)
extracted = []
for i in range(example.capacity()):
    extracted.append(example.extract())
extracted


values
simpleHeapSort(values)


# Max Heap Second Implementation

# __init, __len, capacity, add, extract, _siftUp, _siftDown,

"""
Given index i
PARENT: (i-1) // 2
LEFT: 2 * i + 1
RIGHT: 2 * i + 2
"""

class maxHeap:
    def __init__(self, size):
        self.heap = Array(size)
        self.counter = 0
    def __len__(self):
        return self.counter
    def capacity(self):
        return len(self.heap)
    def add(self, value):
        assert not self.counter >= len(self.heap), "Cannot Add To A Full Heap Tree"
        self.heap[self.counter] = value
        self.counter += 1
        self.shiftUp(self.counter-1)
    def shiftUp(self, index):
        if index > 0:
            parent = (index-1) // 2
            if self.heap[index] > self.heap[parent]:
                self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                self.shiftUp(parent)
    def extract(self):
        assert not self.counter < 0, "Cannot Extract From Empty Heap"
        value = self.heap[0]
        self.counter -= 1
        self.heap[0] = self.heap[self.counter]
        self.heap[self.counter] = None
        self.shiftDown(0)
        return value
    def shiftDown(self, parent):
        left = 2 * parent + 1
        right = 2 * parent + 2

        largest = parent
        if left < self.counter and self.heap[largest] < self.heap[left]:
            largest = left
        elif right < self.counter and self.heap[largest] < self.heap[right]:
            largest = right
        if largest != parent:
            self.heap[parent], self.heap[largest] = self.heap[largest], self.heap[parent]
            self.shiftDown(largest)
    def peek(self):
        return self.heap[0]

values = [10,9,7,5,6,2]
example = maxHeap(len(values))
example.__len__()
example.capacity()
for i in range(len(values)):
    example.add(values[i])
example.peek()
sol = []
list(example.heap)
sol.append(example.extract())
example.peek()
while example.__len__() > 0:
    sol.append(example.extract())
    sol
    example.peek()
    list(example.heap)


# Heap Sort function

def heap_sort(collection):
    heap = maxHeap(len(collection))
    for i in range(len(collection)):
        heap.add(collection[i])
    sorted_arr = []
    while len(heap) > 0:
        sorted_arr.append(heap.extract())
    return sorted_arr


heap_sort(values)

# Third Own Implementation

# __init, __len, add, shiftUp, extract, shiftDown, peek

class max_Heap:
    def __init__(self, values = None):
        self.heap = []
        if values is not None:
            for val in values:
                self.add(val)

    def __len__(self):
        return len(self.heap)
    def peek(self):
        return self.heap[0]
    def add(self, value):
        self.heap.append(value)
        self.shiftUp(len(self)-1)
    def shiftUp(self, index):
        parent = (index-1) // 2
        if parent < 0:
            return
        if self.heap[parent] < self.heap[index]:
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            self.shiftUp(parent)
    def extract(self):
        self.heap[0], self.heap[len(self)-1] = self.heap[len(self)-1], self.heap[0]
        value = self.heap.pop()
        self.shiftDown(0)
        return value
    def shiftDown(self, parent):
        child = 2 * parent + 1
        if child >= len(self.heap):
            return
        if child + 1 < len(self.heap) and self.heap[child] < self.heap[child+1]:
            child += 1
        if self.heap[child] > self.heap[parent]:
            self.heap[child], self.heap[parent] = self.heap[parent], self.heap[child]
            self.shiftDown(child)
def heapSort(list):
    sol = []
    heap = max_Heap(list)
    while heap:
        sol.append(heap.extract())
    return sol


newVals = [10, 9, 7, 5, 6, 2]
sample = max_Heap()
sample.__len__()
for i in range(len(newVals)):
    sample.add(newVals[i])
sample.__len__()
sample.peek()

newsol = []
sample.heap
for i in range(len(newVals)):
    newsol.append(sample.extract())
newsol

heapSort(newVals)

# Max Heap Third Implementation (Internet)

"""

class newMaxHeap:
    def __init__(self, collection=None):
        self._heap = []

        if collection is not None:
            for el in collection:
                self.push(el)

    def push(self, value):
        self._heap.append(value)
        _sift_up(self._heap, len(self) - 1)

    def pop(self):
        _swap(self._heap, len(self) - 1, 0)
        el = self._heap.pop()
        _sift_down(self._heap, 0)
        return el

    def __len__(self):
        return len(self._heap)

    def print(self, idx=1, indent=0):
        print("\t" * indent, f"{self._heap[idx - 1]}")
        left, right = 2 * idx, 2 * idx + 1
        if left <= len(self):
            self.print(left, indent=indent + 1)
        if right <= len(self):
            self.print(right, indent=indent + 1)

def _swap(L, i, j):
    L[i], L[j] = L[j], L[i]

def _sift_up(heap, idx):
    parent_idx = (idx - 1) // 2
    # If we've hit the root node, there's nothing left to do
    if parent_idx < 0:
        return

    # If the current node is larger than the parent node, swap them
    if heap[idx] > heap[parent_idx]:
        _swap(heap, idx, parent_idx)
        _sift_up(heap, parent_idx)

def _sift_down(heap, idx):
    child_idx = 2 * idx + 1
    # If we've hit the end of the heap, there's nothing left to do
    if child_idx >= len(heap):
        return

    # If the node has a both children, swap with the larger one
    if child_idx + 1 < len(heap) and heap[child_idx] < heap[child_idx + 1]:
        child_idx += 1

    # If the child node is smaller than the current node, swap them
    if heap[child_idx] > heap[idx]:
        _swap(heap, child_idx, idx)
        _sift_down(heap, child_idx)

def heap_sort(collection):
    heap = MaxHeap(collection)
    sorted_arr = []
    while len(heap) > 0:
        sorted_arr.append(heap.pop())
    return sorted_arr

sample = newMaxHeap()
values = [10, 9, 7, 5, 6, 2]
for i in range(len(values)):
    sample.push(values[i])
sample.__len__()
sample.print()

newSol = []
for i in range(len(values)):
    newSol.append(sample.pop())
newSol
"""