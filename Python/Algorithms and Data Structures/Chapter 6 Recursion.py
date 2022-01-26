def printRev(n):
    if n > 0 :
        print('RevNum: ', n)
        printRev(n-1)


def printInc(n):
    if n > 0:
        printInc(n-1)
        print('IncNum', n)


def factorial(n):
    assert n >= 0, 'Factorial not defined for negative values'
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


printInc(5)
printRev(5)
factorial(5)


def foo(n):
    if n % 2 != 0:
        return 0
    else:
        return n + foo(n-1)
def bar(n):
    if n > 0:
        print(n)
        bar(n-1)

def main():
    foo(3)
    bar(2)


main()


def fib(n):
    assert n >= 0, 'Enter positive numbers'
    if n < 2:
        return n
    else:
        return fib(n-1) + fib(n-2)


def fastFib(n, memoization):
    if n < 0:
        return 'Enter Positive Numbers'
    elif n == 0 or n ==1:
        return n
    elif n not in memoization:
        memoization[n] = fastFib(n-1, memoization) + fastFib(n-2, memoization)
    return memoization[n]

sol = {}
fastFib(15, sol)  # Using Memoization
fib(10)  # Exponential time fib



### Using a stack to reverse traverse in a singly link list
class Stack:
    def __init__(self):
        self._theItems = list()
    def isEmpty(self):
        return len(self._theItems) == 0
    def __len__(self):
        return len(self._theItems)
    def peek(self):
        assert not self.isEmpty(), 'The Stack must have at least one value'
        return self._theItems[-1]
    def pop(self):
        assert not self.isEmpty(), 'Cannot pop an empty stack'
        #temp = self._theItems[-1]
        #self._theItems = self._theItems[0:len(self._theItems) - 2]
        #return temp
        return self._theItems.pop()
    def push(self, item):
        self._theItems.append(item)
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
class LinkList:
    def __init__(self):
        self.head = None
        self.size = 0
    def __len__(self):
        return self.size
    def isEmtpy(self):
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
            prevNode =curNode
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

example = LinkList()
for i in range(10):
    example.insertOnHead(i)
example.traverse()

def printReverseLinkList(head):  # Reverse Traversal in Singly Link List using a Stack
    s = Stack()
    curNode = head
    while curNode is not None:
        s.push(curNode.data)
        curNode = curNode.next
    sol = []
    while not s.isEmpty():
        sol.append(s.pop())
    print(' -> '.join(list(map(str, sol))))

printReverseLinkList(example.head)


def printReverseList(head):  # Reverse Traversal using recursion
    if head is not None:
        printReverseList(head.next)
        print(head.data)
printReverseList(example.head)



def recursiveBinarySearch(seq, target, first, last):  # Binary Search using recursion
    if first > last:
        False
    else:
        mid = (first + last) // 2
        if seq[mid] == target:
            return True
        elif seq[mid] < target:
            return recursiveBinarySearch(seq, target, mid + 1, last)
        else:
            return recursiveBinarySearch(seq, target, first, mid - 1)
seq = list(range(0,100,1))
recursiveBinarySearch(seq, 22, 0, len(seq) -1)


### Hanoi Towers Puzzle with recursion

def move(n, src, dest, temp):
    if n >= 1:
        move(n-1, src, temp, dest)
        print(" Move %d -> %d " % (src, dest))
        move(n-1, temp, dest, src)
move(3,1,3,2)



### Exponential Operator

def exp1(x, n):  # Exponential operator naive
    y = 1
    for i in range(n):
        y *= x
    return y
exp1(2,8)


def expRecusive(x,n):  # Exponential operator with recursion
    if n == 0:
        return 1
    result = expRecusive(x*x, n // 2)
    if n % 2 == 0:
        return result
    else:
        return x * result
expRecusive(2,8)




