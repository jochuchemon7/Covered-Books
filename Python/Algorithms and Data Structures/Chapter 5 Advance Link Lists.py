# Doubly Link Lists

class DListNode:
    def __init__(self, value):
        self.data = value
        self.next = None
        self.prev = None

def reverseTraversal(tail):
    curNode = tail
    while curNode.next is not None:
        print(curNode.data)
        curNode = curNode.prev

def doublySearch(target):
    if self._head is None:
        return False
    elif self._prove is None:
        self._prove = self._head

    if target < self._prove.data:
        while self._prove is not None and target <= sef._prove.data:
            if target == self._prove.data:
                return True
            else:
                self._prove = self._prove.prev

    else:
        while self._prove is not None and target >= self._prove.data:
            if self._prove == target:
                return True
            else:
                self._prove = self._prove.next
    return False

def doublyInsert(value):
    newNode = DListNode(value)
    if self._head is None:
        self._head = newNode
        newNode = self._head
    elif value < self._head.data:
        newNode.next = self._head
        self._head.prev = newNode
        self._head = newNode
    elif value > self._tail.data:
        newNode.next = self._tail
        self._tail.next = newNode
        self._tail = newNode
    else:
        node = self._head
        while node.next is not None and node.data < value:
            node = node.next
        newNode.next = node
        newNode.tail = node.prev
        node.prev.next = newNode
        node.prev = newNode


# Doubly Link List (Real Implementation)

class _DLinkNode(object):
    def __init__(self, value):
        self.data = value
        self.next = None
        self.prev = None

class DLinkList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.prove = None

    def traversal(self):
        curNode = self.head
        while curNode is not None:
            print(curNode.data)
            curNode = curNode.next
        return

    def reverseTraversal(self):
        curNode = self.tail
        while curNode is not None:
            print(curNode.data)
            curNode = curNode.prev
        return

    def search(self, target):
        if self.head is None:
            print('Not found!')
            return
        elif self.prove is None:
            self.prove = self.head

        if target < self.prove.data:
            while self.prove is not None and target <= self.prove.data:
                if target == self.prove.data:
                    print('Found it!')
                    return
                else:
                    self.prove = self.prove.prev
        else:
            while self.prove is not None and target >= self.prove.data:
                if target == self.prove.data:
                    print('Found it!')
                    return
                else:
                    self.prove = self.prove.next
        print('Not Found!')
        return

    def insert(self, value):
        newNode = _DLinkNode(value)
        if self.head is None:
            self.head = newNode
            self.tail = self.head
            return
        elif newNode.data < self.head.data:
            newNode.next = self.head
            self.head.prev = newNode
            self.head = newNode
            return
        elif newNode.data > self.tail.data:
            newNode.prev = self.tail
            self.tail.next = newNode
            self.tail = newNode
            return
        else:
            node = self.head
            while node is not None and node.data < newNode.data:
                node = node.next
            newNode.next = node
            newNode.prev = node.prev
            node.prev.next = newNode
            node.prev = newNode
            return

    def remove(self, target):
        print('Begining')
        assert self.head is not None or self.tail is not None, 'Cannot remove node from empty Link List'

        if self.prove is None:
            self.prove = self.head

        if target < self.prove.data:
            print('Hello')
            while self.prove is not None and target <= self.prove.data:
                print('Part1')
                if target == self.prove.data:
                    temp = self.prove
                    temp.next.prev = None
                    self.prove = temp.next
                    temp.next = None
                    return temp.data
                else:
                    self.prove = self.prove.prev
        else:
            while self.prove is not None and target >= self.prove.data:
                print('Parta')
                if target == self.prove.data:
                    temp = self.prove
                    print('A-1')
                    # temp.prev.next = None
                    self.prove = temp.prev
                    temp.prev = None
                    return temp.data
                else:
                    print('A-2')
                    self.prove = self.prove.next
        return






class Node:
    def __init__(self, data):
        self.data = data  # adding an element to the node
        self.next = None  # Initally this node will not be linked with any other node
        self.prev = None  # It will not be linked in either direction


# Creating a doubly linked list class
class DoublyLinkedList:
    def __init__(self):
        self.head = None  # Initally there are no elements in the list
        self.tail = None

    def push_front(self, new_data):  # Adding an element before the first element
        new_node = Node(new_data)  # creating a new node with the desired value
        new_node.next = self.head  # newly created node's next pointer will refer to the old head

        if self.head != None:  # Checks whether list is empty or not
            self.head.prev = new_node  # old head's previous pointer will refer to newly created node
            self.head = new_node  # new node becomes the new head
            new_node.prev = None

        else:  # If the list is empty, make new node both head and tail
            self.head = new_node
            self.tail = new_node
            new_node.prev = None  # There's only one element so both pointers refer to null

    def push_back(self, new_data):  # Adding an element after the last element
        new_node = Node(new_data)
        new_node.prev = self.tail

        if self.tail == None:  # checks whether the list is empty, if so make both head and tail as new node
            self.head = new_node
            self.tail = new_node
            new_node.next = None  # the first element's previous pointer has to refer to null

        else:  # If list is not empty, change pointers accordingly
            self.tail.next = new_node
            new_node.next = None
            self.tail = new_node  # Make new node the new tail

    def peek_front(self):  # returns first element
        if self.head == None:  # checks whether list is empty or not
            print("List is empty")
        else:
            return self.head.data

    def peek_back(self):  # returns last element
        if self.tail == None:  # checks whether list is empty or not
            print("List is empty")
        else:
            return self.tail.data

    def pop_front(self):  # removes and returns the first element
        if self.head == None:
            print("List is empty")

        else:
            temp = self.head
            temp.next.prev = None  # remove previous pointer referring to old head
            self.head = temp.next  # make second element the new head
            temp.next = None  # remove next pointer referring to new head
            return temp.data

    def pop_back(self):  # removes and returns the last element
        if self.tail == None:
            print("List is empty")

        else:
            temp = self.tail
            temp.prev.next = None  # removes next pointer referring to old tail
            self.tail = temp.prev  # make second to last element the new tail
            temp.prev = None  # remove previous pointer referring to new tail
            return temp.data

    def insert_after(self, temp_node, new_data):  # Inserting a new node after a given node
        if temp_node == None:
            print("Given node is empty")

        if temp_node != None:
            new_node = Node(new_data)
            new_node.next = temp_node.next
            temp_node.next = new_node
            new_node.prev = temp_node
            if new_node.next != None:
                new_node.next.prev = new_node

            if temp_node == self.tail:  # checks whether new node is being added to the last element
                self.tail = new_node  # makes new node the new tail

    def insert_before(self, temp_node, new_data):  # Inserting a new node before a given node
        if temp_node == None:
            print("Given node is empty")

        if temp_node != None:
            new_node = Node(new_data)
            new_node.prev = temp_node.prev
            temp_node.prev = new_node
            new_node.next = temp_node
            if new_node.prev != None:
                new_node.prev.next = new_node

            if temp_node == self.head:  # checks whether new node is being added before the first element
                self.head = new_node  # makes new node the new head