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
    for i in counter.keys():  # get the length of the each frequency digits(if it's not 1)
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

