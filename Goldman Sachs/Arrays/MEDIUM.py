from typing import List

# 77 Set Matrix Zeroes
def setZeroes(self, matrix: List[List[int]]) -> None:
    """ The simple approach here is to use a hashset to store row and columns numbers that need to be set to zero
    However, we can reduce the space to 0(1) by using the matrix itself as the set, we go through the matrix and for
    every zero value we see, we make the first value in the row and column zero
    However there is a tiny edge case because the if first value of the first column and first row is set to zero because
    a value in the first zero, our algorithm makes the first value in the first row zero(and this also happens to be the
    first value of the first column) and this will make everything in the first column also zero even though it's not
    supposed to be. To solve this we use an additional variable to check if the first column needs to be set to zero

    Ps: you should actually mention this hashset approach first and only talk about this if the interviewer wants a
    better approach(don't over engineer from the start)
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



# 56 Find the Index of the First Occurrence in a String
def strStr(self, haystack: str, needle: str) -> int:
    # let's do brute force for now
    # we can further optimize the brute force greatly but i'll come back to that when i understand string matching
    for i in range(len(haystack) + 1 - len(needle)):
        if haystack[i] == needle[0] and haystack[i:i + len(needle)] == needle:
            return i
    return -1

# 45 Product of Array Except Self -- Arrays
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


# 37 Find Winner of circular game -- Arrays
def findTheWinner(self, n: int, k: int) -> int:
    # build players
    players, index = [player + 1 for player in range(n)], 0
    while players > 1:
        index = (index + k - 1) % len(players)  # using mod allows us to avoid overflow
        players.pop(index)
    return players[0]


# 36 Jump Game -- BFS
def jump(self, nums: List[int]) -> int:
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


# 63 Dot Product of Two Sparse Vectors
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
            result += vectorA * vectorB
        return result
