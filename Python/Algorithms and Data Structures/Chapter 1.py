import random as rand
import numpy as np


#### Search

def linearSearch(values, target):
    length = len(values)
    for i in range(0,length):
        if values[i] == target:
            print("Found")
            return
    print("NOt FOund")
    return

def linearSearchMin(values):
    min = values[0]
    for i in range(0,len(values)):
        if values[i] < min:
            min = values[i]
    return min



def find_sum_of_two(A, val):
  found_values = set()
  for a in A:
    if a - val in found_values:
      print(a)
      return True
    found_values.add(a)
  return False;

v = [5,7,1,2,8,4,3]
test = [3,20,1,2,7]


def binarySearch(values, target):
    high = len(values) - 1
    low = 0
    while low <= high:
        mid = (low + high) // 2
        if(values[mid] == target):
            print("found")
            return
        elif(values[mid] > target):
            high = mid - 1
        else:
            low = mid + 1
    print("not found")
    return


#### Sort

def bubbleSort(values):
    length = len(values)
    for i in range(length):
        for j in range(0,len(values) - i - 1):
            if values[j] > values[j+1]:
                temp = values[j]
                values[j] = values[j+1]
                values[j+1] = temp
                # values[j], values[j+1] = values[j+1], values[j] # Or this!
    return

def selectionSort(values):
    length = len(values)
    for i in range(len(values)):
        minIndex = i
        for j in range(i+1,length):
            if values[j] < values[minIndex]:
                minIndex = j
        values[i], values[minIndex] = values[minIndex], values[i]


def insertionSort(values):
    for i in range(1, len(values)):
        key = values[i]
        j = i - 1
        while j >= 0 and key < values[j]:
            values[j+1] = values[j]
            j -= 1
        values[j+1] = key


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


def partition(start, end, array):
    pivot_index = start
    pivot = array[pivot_index]
    while start < end:
        while start < len(array) and array[start] <= pivot:
            start += 1
        while array[end] > pivot:
            end -= 1
        if (start < end):
            array[start], array[end] = array[end], array[start]
    array[end], array[pivot_index] = array[pivot_index], array[end]
    return end


# The main function that implements QuickSort
def quick_sort(start, end, array):
    if (start < end):
        p = partition(start, end, array)
        quick_sort(start, p - 1, array)
        quick_sort(p + 1, end, array)





# Driver code to test above
arr = [64, 34, 25, 12, 22, 11, 90]

bubbleSort(arr)


def heapify(values, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and values[largest] < values[l]:
        largest = l
    if r < n and values[largest] < values[r]:
        largest = r
    if largest != i:
        values[i], values[largest] = values[largest], values[i]
        heapify(values, n, largest)


def heapSort(values):
    n = len(values)
    for i in range(n // 2 - 1, -1, -1):
        heapify(values, n, i)
    for i in range(n - 1, 0, -1):
        values[i], values[0] = values[0], values[i]
        heapify(values, i, 0)


def mergeSortedLists(listA, listB):
    newList = list()
    a = 0
    b = 0

    while a < len(listA) and b < len(listB):
        if listA[a] < listB[b]:
            newList.append(listA[a])
            a += 1
        else:
            newList.append(listB[b])
            b += 1

    while a < len(listA):
        newList.append(listA[a])
        a += 1

    while b < len(listB):
        newList.append(listB[b])
        b += 1

    print(newList)

