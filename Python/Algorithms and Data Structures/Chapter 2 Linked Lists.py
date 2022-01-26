import numpy as np
import random as rand
### Link Lists

class ListNode:
    def __init__(self, data):
        self.data = data
        self.next = None

def traversal(head):
    curlNode = head
    while curlNode is not None:
        print(curlNode.data)
        curlNode = curlNode.next
    return

def unOrderSearch(head, target):
    curlNode = head
    while curlNode is not None:
        if curlNode.data == target:
            print("found")
            return
        else:
            curlNode = curlNode.next
    print("not founded")
    return

def unorderedSearch(head, target):
    curlNode = head
    while curlNode is not None and curlNode.data != target:
        curlNode = curlNode.next
    return curlNode is not None


# Adding a node to the head of the list
a = ListNode(9)
b = ListNode(10)
c = ListNode(11)

a.next = b
b.next = c
head = a

newNode = ListNode(8)
newNode.next = head
head = newNode


def removeNode(head, target):
    curlNode = head
    predNode = None
    while curlNode.next is not None:
        predNode = curlNode
        curlNode = curlNode.next
        if curlNode.data == target:
            predNode.next = curlNode.next
            curlNode.next = None
            return


def deleteNode(head, target):
    curlNode = head
    prevNode = None
    while curlNode.next is not None and curlNode.data != target:
        prevNode = curlNode
        curlNode = curlNode.next
    if curlNode.next is not None:
        if curlNode is head:
            head = curlNode.next
        else:
            prevNode.next = curlNode.next
    return head


# _init, _len, _contains, add, remove, _iter


class Bag:
    def __init__(self):
        self._size = 0
        self._head = None

    def __len__(self):
        return self._size

    def __contains__(self, target):
        curlNode = self._head
        while curlNode.next is not None and curlNode.item != target:
            curlNode = curlNode.next
        return curlNode.next is not None


    def add(self, item):
        newNode = _BagListNode(item)
        newNode.next = self._head
        self._head = newNode
        self._size += 1

    def remove(self, item):
        prevNode = None
        curlNode = self._head
        while curlNode.next is not None and curlNode.data != item:
            prevNode = curlNode
            curlNode = curlNode.next
        assert curlNode is not None, "The Item should be in the bag"
        self._size -= 1
        if curlNode is head:
            self._head = curlNode
        else:
            prevNode.next = curlNode.next
        return curlNode.item


class _BagListNode(object):
    def __init__(self, item):
        self.next = None
        self.item = item


class BagIterator:
    def __init__(self, listHead):
        self._curlNode = listHead
    def __iter__(self):
        return self

    def __next__(self):
        if self._curlNode.next is None:
            raise StopIteration
        else:
            item = self._curlNode.data
            self._curlNode = self._curlNode.next
            return item


# Appending using tail
traversal(head)
newNode = ListNode(12)
tail = c
if head is None:
    head = newNode
else:
    tail.next = newNode
tail = newNode

# Add node at the head
def addOnHead(head,value):
    newNode = ListNode(value)
    newNode.next = head
    head = newNode
    return head

# Get tail
def getTail(head):
    curNode = head
    while curNode.next is not None:
        curNode = curNode.next
    return curNode


# Removing a node using tail
def removeNode(head, tail, target):
    prevNode = None
    curNode = head
    while curNode.next is not None and curNode.data != target:
        prevNode = curNode
        curNode = curNode.next
    if curNode.next is not None:
        if curNode is head:
            head = curNode.next
        else:
            prevNode.next = curNode.next
        if curNode is tail:
            tail = prevNode
    tail = getTail(head)
    return head, tail


# Search a sorted Link List
def sortedSearch(head, target):
    curNode = head
    counter = 0
    while curNode.next is not None and curNode.data <= target:
        if curNode.data == target:
            print("found at:", counter)
            return
        else:
            counter += 1
            curNode = curNode.next
    print("not found")
    return


def insertValueOnSorted(head, value):
    prevNode = None
    curNode = head
    while curNode is not None and value > curNode.data:
        prevNode = curNode
        curNode = curNode.next
    newNode = ListNode(value)
    newNode.next = curNode
    if curNode is head:
        print("Hello")
        head = newNode
    else:
        prevNode.next = newNode
    tail = getTail(head)
    return head, tail

# Polynomial ADT



class PolyTermNode( object ):
    def __init__(self, degree, coefficient):
        self._degree = degree
        self._coefficient = coefficient
        self._next = None

class Polynomial:

    def __init__(self, degree = None, coefficient = None):
        if degree is None:
            self._polyHead = None
        else:
            self._polyHead = PolyTermNode(degree, coefficient)
        self._polyTail = self._polyHead

    def degree(self):
        if self._polyHead is None:
            return -1
        else:
            return self._polyHead.degree

    def __getitem__(self, degree):
        assert self.degree() >= 0, "Operation not permitted on an empty polynomial"
        curNode = self._polyHead
        while curNode.next is not None  and curNode.degree >= degree:
            curNode = curNode.next
        if curNode is None or curNode.degree != degree:
            return 0.0
        else:
            return curNode.coefficient

    def evaluate(self, scalar):
        curNode = self._polyHead
        total = 0
        while curNode.next is not None:
            total += curNode.coefficient * pow(scalar, curNode.degree)
            curNode = curNode.next
        return total

    def _appendTerm(self, degree, coefficient):
        if coefficient != 0:
            newTerm = PolyTermNode(degree, coefficient)
            if self._polyHead is None:
                self._polyHead = newTerm
            else:
                self._polyHead.next = newTerm
            self._polyTail = newTerm

    def simpleAdd(self, rhsPoly):
        newPoly = Polynomial()
        if self.degree() > rhsPoly.degree():
            maxDegree = self.degree()
        else:
            maxDegree = rhsPoly.degree()
        i = maxDegree
        while i >= 0:
            value = self[i] + rhsPoly[i]
            self._appendTerm(i, value)
            i += 1
        return newPoly

    def __add__(self, rhsPoly):
        assert self.degree() >= 0 and rhsPoly.degree() >= 0, "Only polynomials with degree greater than 0 "
        newPoly = Polynomial()
        nodeA = self._termList
        nodeB = rhsPoly._termList

        while nodeA is not None and nodeB is not None:
            if nodeA.degree > nodeB.degree:
                degree = nodeA.degree
                coefficient = nodeA.coefficient
                nodeA = nodeA.next
            elif nodeA.degree < nodeB.degree:
                degree = nodeB.degree
                coefficient = nodeB.coefficient
                nodeB = nodeB.next
            else:
                degree = nodeA.degree
                value = nodeA.coefficient + nodeB.coefficient
                nodeA = nodeA.next
                nodeB = nodeB.next
            self._appendTerm(degree, value)

        while nodeA is not None:
            self._appendTerm(nodeA.degree, nodeA.coefficient)
            nodeA = nodeA.next
        while nodeB is not None:
            self._appendTerm(nodeB.degree, nodeB.coefficient)
            nodeB = nodeB.next

        return newPoly

    def multiply(self, rhsPoly):
        assert self.degree() >= 0 and rhsPoly.degree() >= 0, "Please only polynomials that are bigger than 0"

        node = self._polyHead
        newPoly = rhsPoly._termMultiply(node)

        node = node.next
        while node is not None:
            tempPoly = rhsPoly._termMultiply(node)
            newPoly = newPoly.add(tempPoly)
            node = node.next
        return newPoly




    def _termMultiply(self, term):
        newPoly = Polynomial()

        curr = curr.next
        while curr is not None:
            newDegree = curr.degree + term.degree
            newCofficient = curr.coefficient  * term.coefficient

            newPoly._appendTerm(newDegree, newCofficient)

            curr = curr.next
        return  newPoly