
# 1 Check Pangram
from typing import List


def checkIfPangram(self, sentence: str) -> bool:
    # create an hashset to hold unique characters, keep adding the characters to the hashset as
    # you go through the list, if the length of the set gets to 26 at any point return True
    hashset = set()
    for char in sentence:
        hashset.add(char)
        if len(hashset) == 26:
            return True
    return False

# 6 First Unique Character in a String
def firstUniqChar(self, s: str) -> int:
    hashmap = {}  # map each character to their occurence
    for i in s:
        hashmap[i] = hashmap.get(i, 0) + 1
    for index, value in enumerate(s):  # return the first value that occurs once
        if hashmap[value] == 1:
            return index
    return -1

# 9 Two Sum
def twoSum(self, nums: List[int], target: int) -> List[int]:
    hashmap = {}  # map the numbers in the list to their index
    for index, value in enumerate(nums):
        remainder = target - value
        if remainder in hashmap:
            return [index, hashmap[remainder]]
        hashmap[value] = index
    return []

# 15 Valid Anagram
def isAnagram(self, s: str, t: str) -> bool:
    # This is a simple hashmap problem, the code is pretty explanatory
    if len(s) != len(t):  # Quick Check
        return False
    counter = {}
    for char in s:
        counter[char] = 1 + counter.get(counter, 0)
    length = len(counter)  # get the size of unique characters
    for char in t:
        if char not in counter or counter[char] < 0:
            return False
        counter[char] -= 1
        if counter[char] == 0:
            length -= 1
    return length == 0

# 16 Design Hashmap
class MyHashMap:
    def __init__(self):
        self.hashmap = {}

    def put(self, key: int, value: int) -> None:
        self.hashmap[key] = value

    def get(self, key: int) -> int:
        return self.hashmap[key] if key in self.hashmap else -1

    def remove(self, key: int) -> None:
        if key in self.hashmap: del self.hashmap[key]

# 42 Palindrome Permutation
def canPermutePalindrome(self, s: str) -> bool:
    # Palindrome reads the same from left to right...we have two types of palindrome: odd palindrome where all letters
    # occurs an even number of times expect for the one in the middle which occurs an odd number of times. Even
    # palindrome: where all the letters occurs an even number of times. So basically to know if we can a string can be
    # permuted into a palindrome we need to make sure at most one of the letters occurs in an odd amount of times
    hashmap = {}
    for char in s: hashmap[char] = 1 + hashmap.get(char, 0)
    odd_count = 0
    for count in hashmap.values():
        if count % 2 and odd_count == 1: return False
        if count % 2: odd_count += 1
    return True

# 39 Shortest Word Distance
def shortestDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
    # we basically use two pointers
    n = shortest = len(wordsDict)
    pointerA = pointerB = None
    for index, word in enumerate(wordsDict):
        if word == word1:
            pointerA = index
        elif word == word2:
            pointerB = index
        if pointerB and pointerA:
            shortest = min(shortest, abs(pointerA - pointerB))
    return shortest


# 36 . Keep Multiplying Found Values by Two
def findFinalValue(self, nums: List[int], original: int) -> int:
    hashset = set(nums)  # clarify with your interviewer if the values will be unique, if they won't we use an hashmap
    while original in hashset:
        original *= 2
    return original




