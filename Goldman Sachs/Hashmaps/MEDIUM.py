import random
from _ast import List


# 47  Insert Delete GetRandom O(1) -- Hashmaps
class RandomizedSet:
    """
    For inserting and deleting in constant time, we need an hashmap. To get random numbers in 0(1), we need a list or
    array to hold the values in the hashmap. So to ensure the array is updated as items are deleted from the
    hashmap, we ensure that the hashmap maps the values to their index.
    """

    def __init__(self):
        self.list = []
        self.hashmap = {}

    def insert(self, val: int) -> bool:
        if val not in self.hashmap:
            self.hashmap[val] = len(self.list)
            self.list.append(val)
            return True
        return False

    def remove(self, val: int) -> bool:
        if val in self.hashmap:
            index = self.hashmap[val]  # get the index of the value to be removed
            last = self.list[-1]
            self.list[index] = last  # replace the number at that index with the last value in the list
            self.hashmap[last] = index  # update the index of the last value in the hashmap
            self.list.pop()  # pop the last value
            del self.hashmap[val]
            return True
        return False

    def getRandom(self) -> int:
        # clarify with your interviewer if you're allowed to use an imported module for getting random numbers
        return random.choice(self.list)



# 53 Valid Sodoku -- Hashmap
def isValidSudoku(self, board: List[List[str]]) -> bool:
    # we're basically checking for repitions here so mapping each row number to a set of unique integers is intuitive
    rows = {row: set() for row in range(9)}
    cols = {col: set() for col in range(9)}
    squares = {(row, col): set() for row in range(3) for col in range(3)}

    for row in range(9):
        for col in range(9):
            value = board[row][col]
            if value == ".":
                continue
            if value in rows[row] or value in cols[col] or value in squares[(row // 3, col // 3)]:
                return False
            rows[row].add(value)
            cols[col].add(value)
            squares[(row // 3, col // 3)].add(value)
    return True



# 65 Find all Anagrams in a String
def findAnagrams(self, s: str, p: str) -> List[int]:
    if len(p) > len(s): return []  # edge case
    sCount = {}
    pCount = {}  # initialise the sliding window
    for i in range(len(p)):
        sCount[s[i]] = sCount.get(s[i], 0) + 1
        pCount[p[i]] = pCount.get(p[i], 0) + 1
    result = [0] if sCount == pCount else []
    left = 0
    for r in range(len(p), len(s)):
        sCount[s[r]] = sCount.get(s[r], 0) + 1
        sCount[s[left]] -= 1
        if sCount[s[left]] == 0:
            del sCount[s[left]]
        left += 1
        if sCount == pCount:
            result.append(left)
    return result


# 61 Longest Consecutive Sequence -- Hashset
def longestConsecutive(self, nums: List[int]) -> int:
    hashset = set(nums)
    longest = 0
    for num in nums:
        if num - 1 not in hashset:  # then the number is the start of a sequence
            length = 1
            while num + length in hashset:
                length += 1
            longest = max(longest, length)
    return longest


# 29 K diff pairs in an array --  Hashmaps
def findPairs(self, nums: List[int], k: int) -> int:
    hashmap = {}
    for i in nums:
        hashmap[i] = 1 + hashmap.get(i, 0)
    counter = 0
    if k == 0:
        for i in hashmap.values():
            if i > 1: counter += 1
        return counter
    for i in hashmap:
        if i + k in hashmap:
            counter += 1
    return counter


# 10 String Compression -- Hashmap
def compress(self, chars: List[str]) -> int:
    # For a weird reason the expected output is different from what was stated in the question description
    counter = {}  # get the letters and their frequency
    for i in chars:
        counter[i] = 1 + counter.get(i, 0)
    count = len(counter)  # number of characters
    for i in counter.values():  # get the length of the each frequency digits(if it's not 1)
        if i != 1:
            count += len(str(i))
    return count  # won't work if you try it on leetcode sha


# 1 Group Anagrams -- Hashmap
def groupAnagrams(self, strs: List[str]):
    # there are two ways to know if two strings are anagram
    # the first one is the sorted version of the two will be the same
    # the second approach is if we have an array that corresponds to
    # the count of each letter in the string, the two strings will have the same
    # i will use the second approach since it has a better runtime(O(NK)) compared to (O(NKlogK) for sorting
    group = {}

    def getCount(string):
        counter = [0] * 26
        for i in string:
            counter[ord(i) - ord('a')] += 1  # clarify with your interviewer if the string contains only lower case
            # letters
        return counter

    for string in strs:
        counts = getCount(string)
        group[tuple(counts)] = [string] + group.get(tuple(counts), [])  # using tuple since an array is not hashable

    return group.values()


# 5 Pairs of Songs With Total Durations Divisible by 60  *
def numPairsDivisibleBy60(self, time: List[int]) -> int:
    # This is a very tricky problem, you can first talk about the brute force approach before bringing this up
    remainders = {i: 0 for i in range(60)}
    pairsCount = 0
    for t in time:
        remainder = t % 60
        if remainder == 0:
            pairsCount += remainders[remainder]
        else:
            pairsCount += remainders[60 - remainder]
        remainders[t % 60] += 1
    return pairsCount


# 325. Maximum Size Subarray Sum Equals k
def maxSubArrayLen(self, nums: List[int], k: int) -> int:
    """we use a prefix Sum approach
    we store the cummulative Sum up to eaxh index in an hashmap, whenever the cummulative sum equals k, we check if
    the lenght is greater than our current maximum lenght.. we also do a check to see if the Sum - k already exists
    in the prefix Sum hashmap"""
    prefixSum = {}
    currSum, maxLen = 0, 0
    for i in range(len(nums)):
        currSum += nums[i]
        if currSum == k:
            maxLen = max(maxLen, i + 1)
        if currSum - k in prefixSum:
            maxLen = max(i - prefixSum[currSum], maxLen)
        if currSum not in prefixSum:
            prefixSum[currSum] = i
    return maxLen


# Snapshot Array
class SnapshotArray:
    """
    - we use an integer to represent the current snap id
    set - We use hashmap to map the index to to the value and current snap id {index: [(val, snap_id)]}
    snap - simply increase the snap id by 1 and return snapid - 1
    get - check if index exist in hashmap, if it does we perform a linear search or better still a binary search
    since the list will be sorted based on the snap id. Also there's a little edge case here(This might change in an
    actual interview so make sure you ask clarifying questions). The edge case is just in case we don't see the exact
    exact snap id, we return the val that has the snap id closest to it."""
    def __init__(self, length: int):
        self.map = {}
        self.snap_id = 0

    def set(self, index: int, val: int) -> None:
        if index not in self.map:
            self.map[index] = []
        self.map[index].append((val, self.snap_id))

    def snap(self) -> int:
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index: int, snap_id: int) -> int:
        if index in self.map:
            values = self.map[index]
            # binary search
            left, right = 0, len(values) - 1
            ans = -1
            while left <= right:
                mid = (left + right) // 2
                if values[mid][1] <= snap_id:
                    ans = mid
                    left = mid + 1
                else:
                    right = mid - 1
            if ans == -1: return 0 # this will happen if the current snap id is greater than all the snapids for the index
            return values[ans][0]
        return 0


