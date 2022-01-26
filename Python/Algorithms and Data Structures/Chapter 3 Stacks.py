# Stack implementation using a list
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


# Stack implementation using a singly link list

class _StackNode:
    def __init__(self, item, link):
        self.item = item
        self.next = link

class Stack:
    def __init__(self):
        self._top = None
        self._size = 0
    def isEmpty(self):
        return self._size == 0
    def __len__(self):
        return self._size
    def peek(self):
        assert not self.isEmpty(), 'Cannot peek an empty Stack'
        return self._top.item
    def pop(self):
        assert not self.isEmpty(), 'Cannot pop an empty Stack'
        node = self._top
        self._top = self._top.next
        self._size -= 1
        return node.item
    def push(self, item):
        self._top = _StackNode(item, self._top)
        self._size += 1


# Example of infix to postfix stack

def infix_to_postfix(infix):
    priority = {'(': 1, '+': 2, '-': 2, '*': 3, '/': 3}
    operators = set(['+','-','*','/','(', ')',"^"])
    postfix = []
    stack = Stack()
    if len(infix.split()) == 1:
        infix = list(map(str, infix))
    else:
        infix = infix.split()
    for token in infix:
        if token not in operators:
            postfix.append(token)
        elif token == '(':
            stack.push(token)
        elif token == ')':
            topToken = stack.pop()
            while topToken != '(':
                postfix.append(topToken)
                topToken = stack.pop()
        else:
            while (not stack.isEmpty()) and (priority[stack.peek()] >= priority[token]):
                postfix.append(stack.pop())
            stack.push(token)
    while not stack.isEmpty():
        postfix.append(stack.pop())
    return " ".join(postfix)

print(infix_to_postfix("A * B + C * D"))
print(infix_to_postfix("( A + B ) * C - ( D - E ) * ( F + G )"))
print(infix_to_postfix("A + B * C"))

def postfixEvaluation(postfix):
    stack = Stack()
    operators = list(['+','-','*','/'])
    postfix = postfix.split()
    for token in postfix:
        if token not in operators:
            stack.push(token)
        elif token in operators:
            first = float(stack.pop())
            second = float(stack.pop())
            if token == '+':
                result = first + second
            elif token == '-':
                result = first - second
            elif token == '*':
                result = first * second
            elif token == '/':
                result = first / second
            stack.push(result)
    return stack.pop()


sample1 = "1 * 2 + 3 * 4"
sample2 = "( 1 + 2 ) * 3 - ( 4 - 5 ) * ( 6 + 7 )"
sample3 = "8+5+1+5*8"

test1 = infix_to_postfix(sample1)
test2 = infix_to_postfix(sample2)
test3 = infix_to_postfix(sample3)

postfixEvaluation(test1)
postfixEvaluation(test2)
postfixEvaluation(test3)


# Backtracking with Maze Example


# Creation of an array and 2D array classes
import ctypes

class Array:
    def __init__(self, size):
        assert size > 0, 'Array size must be greater than 0'
        self._size = size
        PyArrayType = ctypes.py_object * size  # Array using the ctype module
        self._elements = PyArrayType()
        self.clear(None)  # Initialize all the elements
    def __len__(self):
        return self._size
    def __getitem__(self, index):
        assert index >= 0 and index < len(self), 'Array Subscription out of range'
        return self._elements[index]
    def __setitem__(self, index, value):
        assert index >= 0 and index < len(self), 'Array Subscription out of range'
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
            raise StopIteration

class Array2D:
    def __init__(self, numRows, numCols):
        self._theRows = Array(numRows)
        self._nrow = numRows
        self._ncol = numCols
        for i in range(numRows):
            self._theRows[i] = Array(numCols)
    def numRows(self):
        return self._nrow  # len(self._theRows)
    def numCols(self):
        return self._ncol  # (self._theRows[0])
    def clear(self, value):
        for row in range(self.numRows()):
            row.clear(value)
    def __getitem__(self, ndTuple):
        assert len(ndTuple) == 2, 'Invalid Number of array subscripts'
        row = ndTuple[0]
        col = ndTuple[1]
        assert row >= 0 and row < self.numRows() and col >= 0 and col < self.numCols(), 'Array Subscripts out of bound'
        the1dArray = self._theRows[row]
        return the1dArray[col]
    def __setitem__(self, ndTuple, value):
        assert len(ndTuple) == 2, 'Invalid Number of Array Subscripts'
        row = ndTuple[0]
        col = ndTuple[1]
        assert row >= 0 and row < self.numRows() and col >= 0 and col < self.numCols(), 'Array Subscripts out of bound'
        the1dArray = self._theRows[row]
        the1dArray[col] = value

# Example of 2dArray
example = Array2D(3,3)
seq = 1
for row in range(example.numRows()):
    for col in range(example.numCols()):
        example.__setitem__((row,col),(seq))
        seq += 1
for row in range(example.numRows()):
    for col in range(example.numCols()):
        print(example.__getitem__((row,col)), end = '')
    print('')

# Creation of the Maze
class Maze:
    MAZE_WALL = '*'
    PATH_TOKEN = 'X'
    TRIED_TOKEN = 'O'

    def __init__(self, nrow, ncol):
        self._mazeCells = Array2D(nrow, ncol)
        self._startCell = None
        self._exitCell = None
    def numRow(self):
        return self._mazeCells.numRows()
    def numCol(self):
        return self._mazeCells.numCols()
    def setWall(self, row, col):
        assert row >= 0 and row < self.numRow() and col >= 0 and col < self.numCol(), 'Cell index out of range'
        self._mazeCells.set(row, col, self.MAZE_WALL)
    def setStart(self, row, col):
        assert row >= 0 and row < self.numRow() and col >= 0 and col < self.numCol(), 'Cell index out of range'
        self._startCell = _CellPosition(row, col)
    def setExit(self, row, col):
        assert row >= 0 and row < self.numRow() and col >= 0 and col < self.numCol(), 'Cell index out of range'
        self._exitCell = _CellPosition(row, col)
    def findPath(self):
        return True
    def reset(self):
        return True
    def draw(self):
        return True
    def _validMove(self, row, col):
        return row >= 0 and row < self.numRow() and col >= 0 and col < self.numCol() and self._mazeCells[row, col] is None
    def _exitFound(self, row, col):
        return row == self._exitCell.row and col == self._exitCell.col
    def _markTried(self, row, col):
        self._mazeCells.set(row, col, self.TRIED_TOKEN)
    def _markPath(self, row, col):
        self._mazeCells.set(row,col, self.PATH_TOKEN)

class _CellPosition:
    def __init__(self, row, col):
        self.row = row
        self.col = col