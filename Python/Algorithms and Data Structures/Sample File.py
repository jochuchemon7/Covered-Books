# Singly Link List

# len, isempty, insertHead, insertEnd, insertAfter, search, remove, traverse

"""
class Node:
    def __init__(self, value):
        self.next = None
        self.data = value

class LinkList:
    def __init__(self):
        self.head = None
        self.size = 0
    def __len__(self):
        return self.size
    def isEmpty(self):
        return self.size == 0
    def insertHead(self, value):
        newNode = Node(value)
        newNode.next = self.head
        self.head = newNode
        self.size += 1
        return
    def insertEnd(self, value):
        curNode = self.head
        newNode = Node(value)
        while curNode.next is not None:
            curNode = curNode.next
        curNode.next = newNode
        newNode = curNode
        self.size += 1
        return
    def insertAfterNode(self,prevVal, newVal):
        if self.head.data == prevVal:
            newNode = Node(newVal)
            newNode.next = self.head  # newNode.next = self.head.next
            self.head = newNode  # self.head.next = newNode
        else:
            curNode = self.head
            while curNode.next is not None and curNode.data != prevVal:
                curNode = curNode.next
            if curNode is not None:
                newNode = Node(newVal)
                newNode.next = curNode.next
                curNode.next = newNode
        self.size += 1
        return
    def search(self, target):
        curNode = self.head
        while curNode.next is not None and curNode.data != target:
            curNode = curNode.next
        if curNode.data == target:
            print('Found! ', target)
            return
        else:
            print('Not Found!')
            return
    def remove(self, target):
        assert self.head.next is not None, 'Cannot remove from an empty Link List'
        curNode = self.head
        prevNode = None
        while curNode.next is not None and curNode.data != target:
            prevNode = curNode
            curNode = curNode.next
        assert curNode is not None, 'Not Found!'
        self.size -= 1
        if curNode is self.head:
            self.head = curNode.next
        else:
            prevNode.next = curNode.next
        return curNode.data
    def traverse(self):
        curNode = self.head
        sol = ""
        while curNode is not None:
            sol += str(curNode.data)
            curNode = curNode.next
        print(' -> '.join(sol))
        return

"""


class Node:
    def __init__(self, value):
        self.data = value
        self.head = None

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
    def insertEnd(self, value):
        curNode = self.head.next
        while curNode.next is not None:
            curNode = curNode.next
        newNode = Node(value)
        newNode.next = curNode.next
        curNode.next = newNode
        self.size += 1
        return
    def insertAfterNode(self, prevValue, newValue):
        newNode = Node(newValue)
        if self.head is None:
            self.insertOnHead(newValue)
        else:
            curNode = self.head
            while curNode.data != prevValue:
                curNode = curNode.next
            if curNode.data == prevValue:
                newNode.next = curNode.next
                curNode.next = newNode
        self.size += 1
        return
    def search(self, target):
        curNode = self.head
        while curNode.next is not None and curNode.data != target:
            curNode = curNode.next
        if curNode.data == target:
            print('Found! ', target)
            return
        else:
            print('Not Found!')
            return
    def remove(self, target):
        assert self.head is not None,  'Cannot remove from an empty link list'
        curNode = self.head
        prevNode = None
        while curNode is not None and curNode.data != target:
            prevNode = curNode
            curNode = curNode.next
        assert curNode is not None, 'Not Found'
        self.size -= 1
        if curNode is self.head:
            self.head = curNode.next
        else:
            prevNode.next = curNode.next
        return curNode.data
    def traverse(self):
        curNode = self.head
        sol = []
        while curNode is not None:
            sol.append(curNode.data)
            curNode = curNode.next
        print(' -> '.join(list(map(str,sol))))
        return

# """
example = LinkList()
example.isEmpty()
example.__len__()
for i in range(5):
    example.insertOnHead(i)
example.traverse()
example.remove(0)
example.traverse()
example.insertAfterNode(1,0)
example.traverse()

example.remove(0)
example.traverse()
example.insertEnd(0)
example.traverse()

example.insertAfterNode(0,-1)
example.traverse()
example.insertAfterNode(4,3.5)
example.traverse()

# """

# __init, __len, isEmpty, insertOnHead, insertOnEnd, insertAfterNode, search, remove, traverse

### Final Singly Link List

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
    def insertOnEnd(self,value):
        newNode = Node(value)
        curNode = self.head
        while curNode.next is not None:
            curNode = curNode.next
        newNode.next = curNode.next
        curNode.next = newNode
        self.size += 1
        return
    def insertAfterNode(self, preVal, newVal):
        if self.head.data == preVal:
            newNode = Node(newVal)
            newNode.next = self.head.next
            self.head.next = newNode
            self.size += 1
            return
        else:
            curNode = self.head
            while curNode is not None and curNode.data != preVal:
                curNode = curNode.next
            if curNode.data == preVal:
                newNode = Node(newVal)
                newNode.next = curNode.next
                curNode.next = newNode
                self.size += 1
                return
    def search(self, target):
        curNode = self.head
        while curNode is not None and curNode.data != target:
            curNode = curNode.next
        if curNode.data == target:
            print('Found Node!')
        else:
            print('Node Not Found!')
        return
    def remove(self, target):
        assert self.head is not None, 'Cannot remove from empty link list'
        curNode = self.head
        prevNode = None
        while curNode is not None and curNode.data != target:
            prevNode = curNode
            curNode = curNode.next
        assert curNode is not None, 'Node not Found!'
        self.size -= 1
        if curNode is self.head:
            self.head = self.head.next
        else:
            prevNode.next = curNode.next
        return curNode.data
    def traverse(self):
        curNode = self.head
        sol = []
        while curNode is not None:
            sol.append(curNode.data)
            curNode = curNode.next
        print(' -> '.join(list(map(str, sol))))
        return
    def reverseTraverse(self):
        curNode = self.head
        sol = []
        while curNode is not None:
            sol.append(curNode.data)
            curNode = curNode.next
        sol.reverse()
        print(' -> '.join(list(map(str, sol))))
        return

example = LinkList()
example.isEmpty()
example.__len__()
for i in range(5):
    example.insertOnHead(i)
example.traverse()
example.remove(0)
example.traverse()
example.insertAfterNode(1,0)
example.traverse()

example.remove(0)
example.traverse()
example.insertOnEnd(0)
example.traverse()

example.insertAfterNode(0,-1)
example.traverse()
example.insertAfterNode(4,3.5)
example.traverse()

example.reverseTraverse()