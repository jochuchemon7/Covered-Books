import random
import numpy as np

def linearSearch(values, target):
    for value in values:
        if value == target:
            print('Found!')
            return
    print("Not Found")

def binarySearch(values, target):
    high = len(values) - 1
    low = 0
    while low <= high:
        middle = (high + low) // 2
        if values[middle] == target:
            print('Found Value: ', values[middle], " at: ", middle)
            return
        elif values[middle] < target:
            low = middle + 1
        else:
            high = middle - 1

# --- Sort ---

def bubbleSort(values):
    for i in range(len(values)):
        for j in range(0, len(values) - i - 1):
            if values[j] > values[j+1]:
                temp = values[j]
                values[j] = values[j+1]
                values[j+1] = temp
    return

def selectionSort(values):
    for i in range(len(values)):
        minIndex = i
        for j in range(i+1, len(values)):
            if values[minIndex] > values[j]:
                minIndex = j
        values[minIndex], values[j] = values[j], values[minIndex]

def insertionSort(values):
    for i in range(1, len(values)):
        key = values[i]
        j = i - 1
        while j >= 0 and key < values[j]:
            values[j+1] = values[j]
            j -= 1
        values[j+1] = key
    return

def mergeSort(values):
    if len(values) > 1:
        mid = len(values) // 2
        L = values[:mid]
        R = values[mid:]
        mergeSort(L)
        mergeSort(R)
        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                values[k] = L[i]
                i += 1
            else:
                values[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            values[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            values[k] = R[j]
            j += 1
            k += 1


def partition(start, end, values):
    pivot_index = start
    pivot = values[pivot_index]
    while start < end:
        while start < len(values) and values[start] <= pivot:
            start += 1
        while values[end] > pivot:
            end -= 1
        if (start < end):
            values[start], values[end] = values[end], values[start]
    values[end], values[pivot_index] = values[pivot_index], values[end]
    return end

def quickSort(start, end, values):
    if (start < end):
        p = partition(start, end, values)
        quickSort(start, p - 1, values)
        quickSort(p + 1, end, values)

# bubble, selection, insertion, merge, quick

def newBubbleSort(values):
    for i in range(len(values)):
        for j in range(0, len(values) -i -1):
            if values[j] > values[j+1]:
                values[j], values[j+1] = values[j+1], values[j]
    return

def newSelectionSort(values):
    for i in range(len(values)):
        minIndex = i
        for j in range(i, len(values)):
            if values[minIndex] > values[j]:
                minIndex = j
        values[minIndex], values[i] = values[i], values[minIndex]
    return

def newInsertionSort(values):
    for i in range(len(values)):
        key = values[i]
        j = i - 1
        while j >= 0 and values[j] > key:
            values[j+1] = values[j]
            j -= 1
        values[j+1] = key
    return

def newMergeSort(values):
    if len(values) > 1:
        mid = len(values) // 2
        L = values[:mid]
        R = values[mid:]
        newMergeSort(L)
        newMergeSort(R)
        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                values[k] = L[i]
                i += 1
            else:
                values[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            values[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            values[k] = R[j]
            j += 1
            k += 1

def newPartition(start, end, values):
    pivot_index = start
    pivot = values[pivot_index]
    while start < end:
        while start < len(values) and values[start] <= pivot:
            start += 1
        while values[end] > pivot:
            end -= 1
        if start < end:
            values[start], values[end] = values[end], values[start]
    values[pivot_index], values[end] = values[end], values[pivot_index]
    return end

def newQuickSort(start, end, values):
    if start < end:
        p = newPartition(start, end, values)
        newQuickSort(start, p - 1, values)
        newQuickSort(p + 1, end, values)


# Data Creation
values = list(np.random.randint(0, 2000, 1000))
target = values[random.randint(0, 1000)]
sortedValues = sorted(values)

# Functions
linearSearch(values, target)
binarySearch(sortedValues, target)

bubbleSort(values); print(values)
selectionSort(values); print(values)
insertionSort(values); print(values)
mergeSort(values); print(values)
quickSort(0, len(values)-1, values); print(values)
