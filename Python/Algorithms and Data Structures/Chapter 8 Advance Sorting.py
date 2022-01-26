
# First Merge Sort recursive implementation w/creating two list on every call
import random


def mergeSort(list):
    if len(list) < 1:
        return list
    else:
        mid = len(list) // 2

        leftHalf = mergeSort(list[0:mid])
        rightHalf = mergeSort(list[mid:])

        newList = mergeOrderedList(leftHalf, rightHalf)
        return newList

def mergeOrderedList(leftHalf, rigthHalf):
    return 1


# Array ADT implementation needed for merge sort
import ctypes
class Array:
    def __init__(self, size):
        assert not size < 0, 'Enter a larger size'
        self.size = size
        PyArrayType = ctypes.py_object * self.size
        self.array = PyArrayType()
        self.clear(0)
    def __len__(self):
        return self.size
    def __setitem__(self, index, value):
        assert not index > len(self.array), "Index Out Of Bounds!"
        self.array[index] = value
    def __getitem__(self, index):
        assert not index > len(self.array), "Index Out Of Bounds!"
        return self.array[index]
    def clear(self, value):
        for i in range(len(self.array)):
            self.array[i] = value
    def __iter__(self):
        return _ArrayIterator(self.array)
class _ArrayIterator:
    def __init__(self, array):
        self._arrayRef = array
        self._curElement = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self._curElement < len(self._arrayRef):
            entry = self._arrayRef[self._curElement]
            self._curElement += 1
            return entry
        else:
            raise StopIteration


# Second Merger Sort recursive implementation  ( Don't work :( )


def recMergeSort(list, first, last, temp):
    if first == last:
        return
    else:
        mid = (first + last) // 2

        recMergeSort(list, first, mid, temp)
        recMergeSort(list, mid+1, last, temp)

        mergeVirtualSeq(list, first, mid+1, last+1, temp)

def mergeVirtualSeq(list, left, right, end, temp):
    a = left
    b = right

    m = 0  # initialize Index for temp

    while a < right and b < end:
        if list[a] < list[b]:
            temp[m] = list[a]
            a += 1
        else:
            temp[m] = list[b]
            b += 1
        m += 1
    while a < right:
        temp[m] = list[a]
        a += 1
        m += 1

    while b < end:
        temp[m] = list[b]
        b += 1
        m += 1

    for i in range(end - left):
        list[i+left] = temp[i]

def mergeSort(list):
    n = len(list)
    tempArray = Array(n)
    recMergeSort(tempArray, 0, n-1, tempArray)


exampleList = [random.randint(0,150) for _ in range(10)]
exampleList
mergeSort(exampleList)


# Quick Sort

def quickSort(list):
    n = len(list)
    recQuickSort(list, 0, n-1)

def recQuickSort(list, first, last):
    if first >= last:
        return
    else:
        pivot = list[first]
        pos = partitionSeq(list, first, last)

        recQuickSort(list, first, pos-1)
        recQuickSort(list, pos+1, last)

def partitionSeq(list, first, last):
    pivot = list[first]

    left = first + 1
    right = last
    while left <= right:
        while left < right and list[left] < pivot:
            left += 1
        while right >= left and list[right] >= pivot:
            right -= 1

        if left < right:
            temp = list[left]
            list[left] = list[right]
            list[right] = temp
            # list[left], list[right] = list[right], list[left]
    if right != first:
        list[first] = list[right]
        list[right] = pivot

    return right


# First Radix Sorting Implementation

# Queue CLass ADT needed for sort implementation
class Queue:
    def __init__(self):
        self.queue = list()
    def isEmpty(self):
        return len(self.queue) == 0
    def __len(self):
        return len(self.queue)
    def enqueue(self,value):
        self.queue.append(value)
    def dequeue(self):
        assert not self.isEmpty(), "Queue is Empty"
        return self.queue.pop()
def radixSort(list, numDigits):
    binArray = Array(10)
    for i in range(10):
        binArray[i] = Queue()
    column = 1

    for i in range(numDigits):
        for key in list:
            digit = (key // column) % 10
            binArray[digit].enqueue(key)
        i = 0
        for bin in binArray:
            while not bin.isEmpty():
                list[i] = bin.dequeue()
                i += 1
        column *= 10
    return list

sampleList = [random.randint(0,25) for _ in range(14)]
radixSort(sampleList, 10)


# Second Radix Sorting Implementation

def RadixSort(list, binSize):
    sol = []
    power = 0
    while list:
        bins = [[] for _ in range(binSize)]
        for x in list:
            bins[x // binSize ** power % binSize].append(x)
        list = []
        for bin in bins:
            for x in bin:
                if x < binSize ** (power+1):
                    sol.append(x)
                else:
                    list.append(x)
        power += 1
    return sol

sampleList = [random.randint(0,25) for _ in range(15)]
RadixSort(sampleList, 10)


# LinkList Sorting

# (Book's Code)
def llmergeSort(list):
    if list is None:
        return None

    rightList = _splitLinkList(list)
    leftList = list

    leftList = llmergeSort(leftList)
    rightList = llmergeSort(rightList)

    list = _mergeLinkList(leftList, rightList)
    return list
def _splitLinkList(subList):
    midPoint = subList
    curNode = midPoint.next

    while curNode is not None:
        curNode = curNode.next
        if curNode is not None:
            midPoint = midPoint.next
            curNode = curNode.next
    rightList = midPoint.next
    midPoint.next = None
    return rightList
def _mergeLinkList(subListA, subListB):
    newList = Node(None)
    newTail = newList

    while subListA is not None and subListB is not None:
        if subListA.data <= subListB.data:
            newTail.next = subListA
            subListA = subListA.next
        elif subListB.data <= subListA.data:
            newTail.next = subListB
            subListB = subListB.next
        newTail = newTail.next
        newTail.next = None
    if subListA is not None:
        newTail.next  = subListA
    elif subListB is not None:
        newTail.next = subListB
    return newList.next


def mergeSort(list):
    if len(list) > 1:
        middle = len(list) // 2
        L = list[:middle]
        R = list[middle:]
        mergeSort(L)
        mergeSort(R)
        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                list[k] = L[i]
                i += 1
            else:
                list[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            list[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            list[k] = R[j]
            j += 1
            k += 1



class Node:
    def __init__(self, value):
        self.data = value
        self.next = None
class LinkList:
    def __init__(self):
        self.head = None
        self.size = 0
    def __len__(self):
        return self.size
    def isEmpty(self):
        return self.size == 0
    def insertOnHead(self, value):
        newNode = Node(value)
        newNode.next = self.head
        self.head = newNode
        self.size += 1
        return
    def insertOnEnd(self, value):
        newNode = Node(value)
        curNode = self.head
        while curNode.next is not None:
            curNode = curNode.next
        newNode.next = curNode.next
        curNode.next = newNode
        self.size += 1
        return
    def insertAfterNode(self, prevValue, newValue):
        if self.head.data == prevValue:
            newNode = Node(newValue)
            newNode.next = self.head.next
            self.head.next = newNode
            self.size += 1
        else:
            curNode = self.head
            newNode = Node(newValue)
            while curNode is not None and curNode.data != prevValue:
                curNode = curNode.next
            newNode.next = curNode.next
            curNode.next = newNode
            self.size += 1
            return
    def search(self, target):
        curNode = self.head
        while curNode is not None and curNode.data != target:
            curNode = curNode.next
        if curNode.data == target:
            print('Value Node Found!')
        else:
            print('Value Node Not Found!')
        return
    def remove(self, target):
        assert not self.head is None, 'Cannot Remove From Empty Link List'
        curNode = self.head
        prevNode = None
        while curNode is not None and curNode.data != target:
            prevNode = curNode
            curNode = curNode.next
        assert curNode is not None, "Value Node Not Found!"
        self.size -= 1
        if curNode is self.head:
            self.head.next = curNode.next
        else:
            prevNode.next = curNode.next
        return curNode.data
    def traverse(self):
        sol = []
        curNode = self.head
        while curNode is not None:
            sol.append(curNode.data)
            curNode = curNode.next
        print(' -> '.join(list(map(str,sol))))
        return

    def mergeSort(self):
        list = []
        curNode = self.head
        while curNode is not None:
            list.append(curNode.data)
            curNode = curNode.next
        self.sortList(list)
        curNode = self.head
        while curNode is not None:
            curNode.data = list.pop(0)
            curNode = curNode.next
        self.traverse()

    def sortList(self, list):
        if len(list) > 1:
            mid = len(list) // 2
            L = list[:mid]
            R = list[mid:]
            self.sortList(L)
            self.sortList(R)
            i = j = k = 0

            while i < len(L) and j < len(R):
                if L[i] < R[j]:
                    list[k] = L[i]
                    i += 1
                else:
                    list[k] = R[j]
                    j += 1
                k += 1
            while i < len(L):
                list[k] = L[i]
                i += 1
                k += 1
            while j < len(R):
                list[k] = R[j]
                j += 1
                k += 1


example = LinkList()
example.__len__()
example.isEmpty()
[example.insertOnHead(random.randint(0,1500)) for _ in range(1000)]
example.traverse()
example.mergeSort()
example.traverse()