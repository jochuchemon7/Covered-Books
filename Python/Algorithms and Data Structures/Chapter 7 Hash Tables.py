765 % 13
903 % 13  # -> 6
579 % 13


226 % 13
96 % 13


226 % 7
142 % 7
765 % 7


# Separate Chaining  == Open Hashing
# When Keys are store within elements of the table == Closed Hashing

# Types of hashing functions -> [division, truncation, folding, hashing strings]


### Close Hashing Table

# Creating the Array ADT

import ctypes
class Array:
    def __init__(self, size):
        assert size > 0, "Enter a size greater than 0"
        self._size = size
        PyArrayType = ctypes.py_object * size
        self._elements = PyArrayType()
        self.clear(None)
    def __len__(self):
        return self._size
    def __getitem__(self, index):
        assert index > 0 and index < len(self), "Index Out of Bounds"
        return self._elements[index]
    def __setitem__(self, index, value):
        assert index >= 0 and index < len(self), 'Array index out of bounds'
        self._elements[index] = value
    def clear(self, value):
        for i in range(len(self)):
            self._elements[i] = value
    def __iter__(self):
        return _ArrayIterator(self._elements)
class _ArrayIterator:
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
            StopIteration

# Hash Map

class _MapEntry:
    def __init__(self, key, value):
        self.key = key
        self.value = value

# __init, __len, __contains, add, valueOf, remove, __iter, _findSlot, _rehash, _hash1, _has2

class HashMap:
    UNUSED = None
    EMPTY = _MapEntry(None, None)

    def __init__(self):
        self._table = Array(7)
        self._count = 0
        self._maxCount = len(self._table) - len(self._table) // 3
    def __len__(self):
        return self._count
    def __contains__(self, key):
        slot = self._findSlot(key, False)
        return slot is not None
    def add(self, key, value):
        if key in self:
            slot = self._findSlot(key, False)
            self._table[slot].value = value
            return False
        else:
            slot = self._findSlot(key, False)
            self._table[slot] = _MapEntry(key, value)
            self._count += 1
            if self._count == self._maxCount:
                self._rehash()
            return True
    def valueOf(self, key):
        slot = self._findSlot(key, False)
        assert slot is not None, "Invalid Map Key"
        return self._table[slot].value
    def _findSlot(self, key, forInsert):
        slot = self._hash1(key)
        step = self._hash2(key)

        while self._table[slot] is not self.UNUSED:
            if forInsert and (self._table[slot] is self.UNUSED or self._table[slot] is self.EMPTY):
                return slot
            elif not forInsert and (self._table[slot] is not self.UNUSED and self._table[slot].key == key):
                return slot
            else:
                slot = (slot + step) % len(self._table)
    def _rehash(self):
        originalTable = self._table
        newSize = len(self._table) * 2 + 1
        self._table = Array(newSize)

        self._count = 0
        self._maxCount = newSize - newSize // 3

        for entry in originalTable:
            if entry is not self.UNUSED and entry is not self.EMPTY:
                slot = self._findSlot(key, False)
                self._table[slot] = entry
                self._count += 1
    def _hash1(self, key):
        return abs(hash(key)) % len(self._table)
    def _hash2(self, key):
        return 1 + abs(hash(key)) % (len(self._table) - 2)



# HashTable #1 Implementation
class HashTable:

    def __init__(self):
        self.size = 256
        self.hashmap = [[] for _ in range(0, self.size)]
        # print(self.hashmap)

    def hash_func(self, key):
        hashed_key = hash(key) % self.size
        return hashed_key

    def set(self, key, value):
        hash_key = self.hash_func(key)
        key_exists = False
        slot = self.hashmap[hash_key]
        for i, kv in enumerate(slot):
            k, v = kv
            if key == k:
                key_exists = True
                break

        if key_exists:
            slot[i] = ((key, value))
        else:
            slot.append((key, value))

    def get(self, key):
        hash_key = self.hash_func(key)
        slot = self.hashmap[hash_key]
        for kv in slot:
            k, v = kv
            if key == k:
                return v
            else:
                raise KeyError('Key does not exist.')

    def __setitem__(self, key, value):
        return self.set(key, value)

    def __getitem__(self, key):
        return self.get(key)


H = HashTable()
H.set('key1','value1')
H.set('key2','value2')
H.set('key3','value3')
H.set(10,'value10')
H.set(20, 'value20')
H['NEWWWWWWWWW'] = 'newwwwwwwww'

print(H['key1'])
print(H[10])
print(H[20])
print(H.hashmap)



# HashTable #2 Implementation
class HashTable:
    def __init__(self):
        self.size = 11
        self.slots = [None] * self.size
        self.data = [None] * self.size

    def put(self,key,data):
      hashvalue = self.hashfunction(key,len(self.slots))

      if self.slots[hashvalue] == None:
        self.slots[hashvalue] = key
        self.data[hashvalue] = data
      else:
        if self.slots[hashvalue] == key:
          self.data[hashvalue] = data  #replace
        else:
          nextslot = self.rehash(hashvalue,len(self.slots))
          while self.slots[nextslot] != None and \
                          self.slots[nextslot] != key:
            nextslot = self.rehash(nextslot,len(self.slots))

          if self.slots[nextslot] == None:
            self.slots[nextslot]=key
            self.data[nextslot]=data
          else:
            self.data[nextslot] = data #replace

    def hashfunction(self,key,size):
         return key%size

    def rehash(self,oldhash,size):
        return (oldhash+1)%size

    def get(self,key):
      startslot = self.hashfunction(key,len(self.slots))

      data = None
      stop = False
      found = False
      position = startslot
      while self.slots[position] != None and  \
                           not found and not stop:
         if self.slots[position] == key:
           found = True
           data = self.data[position]
         else:
           position=self.rehash(position,len(self.slots))
           if position == startslot:
               stop = True
      return data

    def __getitem__(self,key):
        return self.get(key)

    def __setitem__(self,key,data):
        self.put(key,data)

H=HashTable()
H[54]="cat"
H[26]="dog"
H[93]="lion"
H[17]="tiger"
H[77]="bird"
H[31]="cow"
H[44]="goat"
H[55]="pig"
H[20]="chicken"
print(H.slots)
print(H.data)

print(H[20])

print(H[17])
H[20]='duck'
print(H[20])
print(H[99])




# HashTable #3 Implementation  (Own Implementation Semi finished)

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None
    def __iter__(self):
        print('Key: %s, Value: %s' % (str(self.key), str(self.value)))

# __init, __len, isEmpty, insert, search, remove, hashFunction


class HashTable:
    def __init__(self, size):
        self.table = [Node(None, None) for _ in range(size)]
        self.size = size
    def __len__(self):
        return self.size
    def isEmpty(self):
        return self.size == 0
    def hashFunction(self, key):
        hashedKey = 0
        for i in str(key):
            hashedKey += ord(i)
        hashedKey %= self.size
        return hashedKey
    def insert(self, key, value):
        hashedKey = self.hashFunction(key)
        assert not len(self.table) - 1 < hashedKey, 'Hashed Key Out of Bounds!'
        if self.table[hashedKey].key == None:
            self.table[hashedKey].key = key
            self.table[hashedKey].value = value
        elif self.table[hashedKey].key != None:
            newNode = Node(key, value)
            self.table[hashedKey].next = newNode
        return
    def search(self, key):
        hashedKey = self.hashFunction(key)
        assert not len(self.table) - 1 < hashedKey, 'Hashed Key Out of Bounds!'
        if self.table[hashedKey].key == None:
            print('No key found!')
        elif self.table[hashedKey].key  != None:
            found = False
            curNode = self.table[hashedKey]
            while not found:
                if curNode.key == key:
                    curNode.__iter__()
                    found = True
                elif curNode.next == None:
                    print('No Key/Value Pair Found!')
                    found = False
                else:
                    curNode = curNode.next
        return
    def remove(self, key):
        hashedKey = self.hashFunction(key)
        assert not len(self.table) - 1 < hashedKey, "Hashed Key Out Of Bounds!"
        if self.table[hashedKey].key == None:
            print('No Key Found!')
        elif self.table[hashedKey].key != None:
            curNode = self.table[hashedKey]
            prevNode = None
            while curNode.next is not None and curNode.key != key:
                prevNode = curNode
                curNode = curNode.next
            assert curNode is not None, 'Key Not Found in Link List'
            if curNode is self.table[hashedKey]:
                self.table[hashedKey].next = curNode.next
            else:
                prevNode.next = curNode.next
            return curNode.__iter__()

hashTable = HashTable(8)
keys = list(range(1100,1111,1))
values = ['one', 'two', 'three', 'four', 'five','six', 'seven', 'eigth', 'nine', 'ten']
for i in range(len(keys) - 4):
    hashTable.insert(keys[i], values[i])

