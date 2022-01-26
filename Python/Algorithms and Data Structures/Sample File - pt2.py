# __init, __len, isEmpty, insertOnHead, insertOnEnd, insertAfterNode, search, remove, traverse

class DNode:
    def __init__(self, value):
        self.data = value
        self.next = None
        self.prev = None

class DLinkList:
    def __init__(self, value):
        self.head = DNode(value)
        self.tail = self.head
        self.size = 1
    def __len__(self):
        return self.size
    def isEmpty(self):
        return self.size == 0
    def insertOnHead(self, value):
        newNode = DNode(value)
        newNode.next = self.head
        newNode.prev = self.head
        newNode.next.prev = newNode
        self.head = newNode
        self.size += 1
        return
    def insertOnTail(self, value):
        newNode = DNode(value)
        self.tail.next = newNode
        newNode.prev = self.tail
        self.tail = newNode
        self.size += 1
        return
    def insertAfterNode(self, prevValue, newValue):
        if self.head is None:
            self.head = DNode(newValue)
            self.tail = self.head
        if self.tail.data == prevValue:
            self.insertOnTail(newValue)
        elif self.head.data == prevValue:
            newNode = DNode(newValue)
            newNode.next = self.head.next
            newNode.prev = self.head
            newNode.next.prev = newNode
            self.head.next = newNode
        else:
            curNode = self.head.next
            while curNode is not None and curNode.data != prevValue:
                curNode = curNode.next
            newNode = DNode(newValue)
            newNode.next = curNode.next
            newNode.prev = curNode
            newNode.next.prev = newNode
            curNode.next = newNode
        self.size += 1
        return
    def search(self, target):
        assert self.head is not None, 'Empty Link List'
        if self.tail.data == target:
            print('Found Node')
        curNode = self.head
        while curNode is not None and curNode.data != target:
            curNode = curNode.next
        if curNode.data == target:
            print('Node Found')
        else:
            print('Node Not Found')
        return
    def remove(self, target):
        assert self.head is not None, 'Cannot remove from empty link list'
        if self.tail.data == target:
            self.tail = self.tail.prev
            self.size -= 1
            return
        curNode = self.head
        prevNode = None
        while curNode.data != target:
            prevNode = curNode
            curNode = curNode.next
        assert curNode is not None, 'Cannot find the node'
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
    def reverseTraverse(self):
        curNode = self.tail
        sol = ""
        while curNode is not None:
            sol += str(curNode.data)
            curNode = curNode.prev
        print( ' -> '.join(sol))
        return


example = DLinkList(0)
example.isEmpty()
example.__len__()
for i in range(1,5):
    example.insertOnHead(i)
example.traverse()

example.remove(0)
example.traverse()
example.insertAfterNode(1,0)
example.traverse()


example.remove(0)
example.traverse()
example.insertOnTail(0)
example.traverse()



# __init, __len, isEmpty, insertOnHead, insertOnTail, insertAfterNode, search, remove, traverse

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class DoublyList:
    def __init__(self, value):
        self.head = Node(value)
        self.tail = self.head
        self.size = 1
    def __len__(self):
        return self.size
    def isEmpty(self):
        return self.size == 0
    def insertOnTail(self, value):
        newNode = Node(value)
        newNode.prev = self.tail
        self.tail.next = newNode
        self.tail = newNode
        self.size += 1
        return
    def insertOnHead(self, value):
        newNode = Node(value)
        newNode.next = self.head
        self.head.prev = newNode
        self.head = newNode
        self.size += 1
        return
    def insertAfterNode(self, prevVal, newVal):
        if self.head is None:
            self.insertOnHead(newVal)
        elif self.tail.data == prevVal:
            self.insertOnTail(newVal)
        else:
            newNode = Node(newVal)
            curNode = self.head
            while curNode.data != prevVal:
                curNode = curNode.next
            newNode.next = curNode.next
            newNode.prev = curNode
            newNode.next.prev = newNode
            curNode.next = newNode
            self.size += 1
        return
    def search(self, target):
        if self.tail.data == target:
            print('Found!')
            return
        else:
            curNode = self.head
            while curNode is not None and curNode.data != target:
                curNode = curNode.next
            if curNode is not None:
                print('Found Node %d' % target)
            else:
                print('Node Not Found')
            return
    def remove(self, target):
        assert self.head is not None, 'Cannot remove from empty Link List'
        if self.tail.data == target:
            self.tail = self.tail.prev
            self.tail.next = None
            self.size -= 1
            return
        else:
            curNode = self.head
            prevNode = None
            while curNode.data != target:
                prevNode = curNode
                curNode = curNode.next
            assert curNode is not None, 'Node Not Found!'
            self.size -= 1
            item = curNode.data
            if curNode is self.head:
                self.head = self.head.next
                self.head.prev = None
            else:
                prevNode.next = curNode.next
                curNode.next.prev = prevNode
            return item
    def traverse(self):
        curNode = self.head
        sol = []
        while curNode is not None:
            sol.append(curNode.data)
            curNode = curNode.next
        print(' -> '.join(list(map(str,sol))))
        return
    def reveseTraverse(self):
        curNode = self.tail
        sol = []
        while curNode is not None:
            sol.append(curNode.data)
            curNode = curNode.prev
        print(' -> '.join(list(map(str,sol))))
        return

"""
example = DoublyList(2)
example.__len__()
example.isEmpty()
example.traverse()
example.reveseTraverse()

for i in range(1,-1,-1):
    example.insertOnTail(i)
example.reveseTraverse()
example.traverse()

for i in range(3,5,1):
    example.insertOnHead(i)
example.traverse()
example.reveseTraverse()

example.insertOnHead(6)
example.traverse()
example.reveseTraverse()

example.insertAfterNode(6,5)
example.traverse()
example.reveseTraverse()
"""

# __init, __len, isEmpty, insertOnHead, insertOnTail, insertAfterNode, search, remove, traverse, reverseTraverse

### Final Doubly Link List

class DNode:
    def __init__(self, value):
        self.next = None
        self.prev = None
        self.data = value

class DoublyList:
    def __init__(self, value):
        self.head = DNode(value)
        self.tail = self.head
        self.size = 1
    def __len__(self):
        return self.size
    def isEmpty(self):
        return self.size == 0
    def insertOnHead(self, value):
        newNode = DNode(value)
        newNode.next = self.head
        # newNode.prev = self.head.prev
        self.head.prev = newNode
        self.head = newNode
        self.size += 1
        return
    def insertOnTail(self, value):
        newNode = DNode(value)
        newNode.prev = self.tail
        self.tail.next = newNode
        self.tail = newNode
        self.size += 1
        return
    def insertAfterNode(self,preVal, newVal):
        if self.head is None:
            self.insertOnHead(newVal)
        elif self.head.data == preVal:
            newNode = DNode(newVal)
            newNode.next = self.head.next
            newNode.prev = self.head
            newNode.next.prev = newNode
            self.head.next = newNode
            self.size += 1
        elif self.tail.data == preVal:
            self.insertOnTail(newVal)
        else:
            newNode = DNode(newVal)
            curNode = self.head
            while curNode.data != preVal:
                curNode = curNode.next
            newNode.next = curNode.next
            newNode.prev = curNode
            newNode.next.prev = newNode
            curNode.next = newNode
            self.size += 1
        return
    def search(self, target):
        curNode = self.head
        while curNode is not None or curNode.data != target:
            curNode = curNode.next
        if curNode.data == target:
            print('Node Found!')
        else:
            print('Node Not Found!')
        return
    def remove(self, target):
        assert self.head is not None, 'Cannot remove from an empty link list'
        curNode = self.head
        prevNode = None
        while curNode is not None and curNode.data != target:
            prevNode = curNode
            curNode = curNode.next
        assert curNode is not None, 'Node not found'
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
        print(' -> '.join(list(map(str,  sol))))
        return
    def reverseTraverse(self):
        curNode = self.tail
        sol = []
        while curNode is not None:
            sol.append(curNode.data)
            curNode = curNode.prev
        print(' -> '.join(list(map(str,sol))))
        return


"""
example = DoublyList(2)
example.__len__()
example.isEmpty()
example.traverse()
example.reverseTraverse()

for i in range(1,-1,-1):
    example.insertOnTail(i)
example.reverseTraverse()
example.traverse()

for i in range(3,5,1):
    example.insertOnHead(i)
example.traverse()
example.reverseTraverse()

example.insertOnHead(6)
example.traverse()
example.reverseTraverse()

example.insertAfterNode(6,5)
example.traverse()
example.reverseTraverse()
"""


# __init, __len, isEmpty, insertOnHead, insertOnTail, insertAfterNode, search, remove, traverse, reverseTraverse

