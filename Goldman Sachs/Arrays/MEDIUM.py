from typing import List

# 1 Set Matrix Zeroes
def setZeroes(self, matrix: List[List[int]]) -> None:
    """ The simple approach here is to use a hashset to store row and columns numbers that need to be set to zero
    However, we can reduce the space to 0(1) by using the matrix itself as the set, we go through the matrix and for
    every zero value we see, we make the first value in the row and column zero
    However there is a tiny edge case because the if first value of the first column and first row is set to zero because
    a value in the first zero, our algorithm makes the first value in the first row zero(and this also happens to be the
    first value of the first column) and this will make everything in the first column also zero even though it's not
    supposed to be. To solve this we use an additional variable to check if the first column needs to be set to zero

    Ps: you should actually mention this hashset approach first and then talk about this as a better alternative
    """
    rows, cols = len(matrix), len(matrix[0])
    is_col = False

    for row in range(rows):
        # if any of the first values in the row is zero, then the first column has to be set to all zeroes later
        if matrix[row][0] == 0:
            is_col = True
        # if an element is zero, we set the first value in it row and column to zero
        for col in range(1, col):
            if matrix[row][col] == 0:
                matrix[row][0] = 0
                matrix[0][col] = 0

    # set values to zeroes
    for row in range(1, rows):
        for col in range(1, cols):
            if not matrix[0][col] or not matrix[row][0]:
                matrix[row][col] = 0

    # check first row
    if matrix[0][0] == 0:
        for col in range(cols):
            matrix[0][col] = 0

    # check first column
    if is_col:
        for row in range(rows):
            matrix[rows][0] = 0



# 2 Find the Index of the First Occurrence in a String
def strStr(self, haystack: str, needle: str) -> int:
    # let's do brute force for now
    # we can further optimize the brute force greatly but i'll come back to that when i understand string matching
    for i in range(len(haystack) + 1 - len(needle)):
        if haystack[i] == needle[0] and haystack[i:i + len(needle)] == needle:
            return i
    return -1

# 3 Product of Array Except Self -- Arrays
def productExceptSelf(self, nums: List[int]) -> List[int]:
    # The brute force here will be to compute the product just before the current number and use another for loop
    # to compute the product of the rest....We can reduce the runtime from O(n^2) to 0(n) by computing the product
    # before each number and after each number seperately using a porefix and postfix pointer
    result = [1] * len(nums)
    prefix = 1
    for i in range(len(nums)):
        result[i] = prefix
        prefix *= nums[i]

    postfix = 1
    for i in range(len(nums) - 1, -1, -1):
        result[i] *= postfix
        postfix *= nums[i]

    return result


# 4 Find Winner of circular game -- Arrays
def findTheWinner(self, n: int, k: int) -> int:
    # build players
    players, index = [player + 1 for player in range(n)], 0
    while players > 1:
        index = (index + k - 1) % len(players)  # using mod allows us to avoid overflow
        players.pop(index)
    return players[0]


# 5 Jump Game
def jump(self, nums: List[int]) -> int:
    # Neetcode has a clever explanation of the algorithm used here
    result = 0
    left = right = 0
    while right < len(nums) - 1:  # tells us when we've gotten to the end of the list
        farthest = 0
        for i in range(left, right + 1):
            farthest = max(farthest, i + nums[i])
        left = right + 1
        right = farthest
        result += 1
    return result


# 6 Dot Product of Two Sparse Vectors
class SparseVector:
    def __init__(self, nums: List[int]):
        self.array = nums

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        """This approach below is actually a straight forward approach but your interviewer could raise discussions
        around other solutiions and possible tradeoffs...Don't know much but we could use an hashmap to keep track of
        the non zero vectors so we don't have to do unneseccary multiplications all the time..However using hashmap
        increases the space complexity from O(1) to O(n) and when we have several sparse vectors filled with non zero
        vector, the time taken to compute the hashing also affect our solution"""
        result = 0
        for vectorA, vectorB in zip(self.array, vec.array):
            if vectorA and vectorB:
                result += vectorA * vectorB
        return result


# 240. Search a 2D Matrix II
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    """ The obvious thing to do here is search the enitre matrix for our target. However the sorted property of the ma
    trix should hint us that we can narrow down the search to specific column or row as we go through the matrix
    So basically we start from either the top right or the bottom left of the matrix, if the current value is less than
    the the value at the current position of the matrix, we move to the left by decreasing our column number by 1.
    If the value is more than the current value position, we move to the next row for that same column number.. Hopef
    ully this is explanatory enough"""
    if not matrix: return False

    row, col = 0, len(matrix[0]) - 1
    while row < len(matrix) and col >= 0:
        value = matrix[row][col]
        if value == target:
            return True

        elif value < target:
            col -= 1

        else: # value more than the target so we move to the next rows
            row += 1
    return False


# Next Permutation
def nextPermutation(self, nums: List[int]) -> None:
    # Brute force approach would be generate every possible permutations of the list, sort them and return the
    # one next to the current list....
    n = len(nums)
    i = n - 2
    # look for the rightmost pair that's in ascending order, if we don't see any then the list is already sorted in
    # descending order which means it has no next greater permutation
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    # if we find an adjacent pair, we look for the smallest element towards the right of the left element
    # in our pair and swap them
    if i >= 0:
        j = n - 1 # we start checking from the list
        while j >= 0 and nums[i] >= nums[j]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]

    # sort the right side of the left element in our pair in ascending order to make sure it's in the lowest order
    left, right = i + 1, n - 1
    while left < right:
        nums[right], nums[left] = nums[left], nums[right]
        right -= 1
        left += 1


# Find all the good Indices
def goodIndices(self, nums: List[int], k: int) -> List[int]:
    """ For this problem, we use two arrays.
    - One to keep track of the number of descending values towards the left of each index and one to determine
    the number of ascending values towards the right of the index. E.g leftArray[i] = 2 means at index i, there are 2
    values towards the left and they are in descending order..You get the gist now"""
    # an helper function to do the counting, we call it on the array and reverse of the array
    def helper(nums):
        counter = 0
        arr = [0] * len(nums)
        for index, value in enumerate(nums):
            arr[index] = counter
            if index > 0 and nums[index - 1] >= value:
                counter += 1
            else:
                counter = 1  # we reset the counter
        return arr

    leftCount = helper(nums)
    rightCount = helper(nums[::-1])[::-1]
    return [i for i in range(len(nums)) if leftCount[i] >= k and rightCount[i] >= k]


# Spiral Matrix
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    # we set up boundaries
    # Neetcode has a pretty good explanation on the algorithm used here https://www.youtube.com/watch?v=BJnMZNwUk1M
    left, right = 0, len(matrix[0]) - 1  # column boundaries
    top, bottom = 0, len(matrix) - 1  # row boundaries
    result = []
    while top <= bottom and left <= right:
        # top row
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        # right column
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        # check if boundaries are intact
        if not (top <= bottom and left <= right):
            break

        # bottom row
        for i in range(right, left - 1, - 1):
            result.append(matrix[bottom][i])
        bottom -= 1
        # left column
        for i in range(bottom, top - 1, -1):
            result.append(matrix[i][left])
        left += 1
    return result
