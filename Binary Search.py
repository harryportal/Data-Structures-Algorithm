import math
from typing import List


# Binary Search
def search(self, nums: List[int], target: int) -> int:
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            high = mid - 1
        else:
            low = mid + 1
    return -1


# Search a 2D Matrix
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    # first we figure out the row to search
    rows, cols = len(matrix), len(matrix[0])
    low, high = 0, rows - 1
    while low <= high:
        row = (low + high) // 2
        if target > matrix[row][-1]:
            low = row + 1
        elif target < matrix[row][0]:
            high = row - 1
        else:
            break

    # now we've found the row
    row = (low + high) // 2
    low, high = 0, cols - 1
    while low <= high:
        mid = (low + high) // 2
        if target > matrix[row][mid]:
            low = mid + 1
        elif target < matrix[row][mid]:
            high = mid - 1
        else:
            return True
    return False


# Koko eating bananas
def minEatingSpeed(self, piles: List[int], h: int) -> int:
    result = 0
    low, high = 1, max(piles)
    while low <= high:
        speed = (low + high) // 2
        hours_spent = sum([math.ceil(x / speed) for x in piles])
        if hours_spent <= h:
            result = speed
            high = speed - 1
        else:
            low = speed + 1
    return result


# Search in a Rotated Sorted Array
def _search(self, nums: List[int], target: int) -> int:
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        # check for left portion
        if nums[mid] >= nums[low]:
            if target > nums[mid] or target < nums[low]:
                low = mid + 1
            else:
                high = mid - 1
        else:
            if target < nums[mid] or target > nums[high]:
                high = mid - 1
            else:
                low = mid + 1
    return -1


# Find Minimum in a rotated sorted Array
def findMin(self, nums: List[int]) -> int:
    minimum = nums[0]
    low, high = 0, len(nums) - 1
    while low <= high:
        # sorted already
        if nums[low] < nums[high]:
            return min(minimum, nums[low])
        mid = (low + high) // 2
        minimum = min(minimum, nums[mid])
        if nums[mid] >= nums[low]:
            low = mid + 1
        else:
            high = mid - 1
    return minimum


# Time Based Key Value Store
class TimeMap:

    def __init__(self):
        self.timeMap = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.timeMap:
            self.timeMap[key] = []
        self.timeMap[key].append([value, timestamp])

    def get(self, key: str, timestamp: int) -> str:
        result = ""
        if key not in self.timeMap:
            return result
        key = self.timeMap[key]
        low, high = 0, len(key) - 1
        while low <= high:
            mid = (low + high) // 2
            if key[mid][1] == timestamp:
                return key[mid][0]
            if key[mid][1] < timestamp:
                result = key[mid][0]
                low = mid + 1
            else:
                high = mid - 1
        return result
