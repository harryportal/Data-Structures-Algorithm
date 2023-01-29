import math
import random
from typing import List


# 1 Contains Duplicate - Easy
def containsDuplicate(self, nums: List[int]) -> bool:
    hashset = set()
    for i in nums:
        if i in hashset:
            return True
        hashset.add(i)
    return False


# 2 Is Anagram - Easy
def isAnagram(self, s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    hashmap = {}
    for char in s:
        hashmap[char] = hashmap.get(char, 0) + 1
    length = len(hashmap)  # lenght of unique characters in string s
    for char in t:
        if char not in hashmap or hashmap[char] < 0:
            return False
        hashmap[char] -= 1
        if hashmap[char] == 0:
            length -= 1
    return length == 0


# 3 Two Sum -- Easy
def twoSum(self, nums: List[int], target: int) -> List[int]:
    hashmap = {}  # map elements to their index
    for index, value in enumerate(nums):
        remainder = target - value
        if remainder in hashmap:
            return [hashmap[remainder], index]
        hashmap[value] = index
    return [-1, -1]  # depends on the default return


# 4 Top k Frequent - Medium -- analyze the complexity with Shade later
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    hashmap = {}  # map elements to their count
    for i in nums:
        hashmap[i] = hashmap.get(i, 0) + 1
    frequency: List[List[int]] = [[] for i in range(len(nums) + 1)]  # added the annonation for pycharm lol
    # a list where the index is the count and the values is a list of numbers with that count
    for number, count in hashmap.items():
        frequency[count].append(number)
    topk = []
    for i in range(len(frequency) - 1, 0, -1):
        for j in frequency[i]:
            topk.append(j)
            if len(topk) == k:
                return topk


# 5 Product Except Self
def productExceptSelf(self, nums: List[int]) -> List[int]:
    result = [1 for i in range(len(nums))]
    # prefix operation
    prefix = 1
    for i in range(len(nums)):
        result[i] = prefix
        prefix *= nums[i]
    # postfix operation
    postfix = 1
    for i in range(len(nums) - 1, -1, -1):
        result[i] *= postfix
        postfix *= nums[i]
    return result


# 6 Is Valid Sodoku -- Medium
def isValidSudoku(self, board: List[List[str]]) -> bool:
    rows = {row: set() for row in range(9)}
    cols = {col: set() for col in range(9)}
    square_grid = {(row, col): set() for row in range(3) for col in range(3)}
    for row in range(9):
        for col in range(9):
            value = board[row][col]
            if value == ".":
                continue
            if value in rows[row] or value in cols[col] or value in square_grid[(row // 3, col // 3)]:
                return False
            rows[row].add(value)
            cols[col].add(value)
            square_grid[(row // 3, col // 3)].add(value)
    return True


# 7 Encode and Decode Strings -- Medium  -- analayze complexity with Shade
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
          The logic here is to add the lenght of the string and a symbol before each string
        """
        encode = ""
        for word in strs:
            encode += str(len(word)) + "*" + word
        return encode

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        Don't know the best type the logic out but take we are going to be very particular about index here
        first we get the lenght , skip the symbol, grab the string since we know it's lenght,
        shift our index and repeat the above steps
        """
        decode = []
        i = 0
        while i < len(s):
            # get the lenght of the string to be captured
            j = i
            while s[j] != "*":
                j += 1
            length = int(s[i:j])
            string = s[j + 1: j + 1 + length]
            decode.append(string)
            i = j + 1 + length
        return decode


# 8 longest consecutive sequence
def longestConsecutive(self, nums: List[int]) -> int:
    # the logic here is to know when we are the beginning of a sequence
    # to do that we just check if a number 1 less than the current number exists
    hashset = set(nums)
    longest = 0
    for i in nums:
        if i - 1 not in hashset:
            lenght = 1
            while i + lenght in hashset:
                lenght += 1
            longest = max(lenght, longest)
    return longest


# 9 Inser Delte Get Random 0(1)
class RandomizedSet:
    def __init__(self):
        self.array = []  # needed cause we need to get random numbers
        self.hashmap = {}  # map elements to thier index in the array

    def insert(self, val: int) -> bool:
        if val in self.hashmap:
            return False
        self.hashmap[val] = len(self.array)
        self.array.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.hashmap:
            return False
        index = self.hashmap[val]
        last_element = self.array[-1]
        self.array[index] = last_element
        self.hashmap[last_element] = index
        del self.hashmap[val]
        self.array.pop()
        return True

    def getRandom(self) -> int:
        return random.choice(self.array)


# 10 Word Pattern  - easy
def wordPattern(self, pattern: str, s: str) -> bool:
    list_word = s.split()  # convert the list of words to a list
    if len(pattern) != len(s.split()):
        return False
    patternmap = {}  # map the pattern to the word
    wordmap = {}  # map the word to the pattern
    for i in range(len(pattern)):
        word, char = list_word[i], pattern[i]
        if (char in patternmap and patternmap[char] != word) or \
                (word in wordmap and wordmap[word] != char):
            return False
        patternmap[char] = word
        wordmap[word] = char
    return True


# 11 Majority Element - easy
def majorityElement(self, nums):
    # the most intuitive approach here is to use an hashmap to count the frequencies
    # of the numbers and return the number that occurs (n/2) times plus there is only
    # number that can have that number of occurence
    # Using Bayor-Moore's Algorithm -- 0(n) and 0(1) -space
    count = 0
    candidate = None
    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    return candidate


# 12 Unique Email Adresss
def numUniqueEmails(self, emails: List[str]) -> int:
    unique_emails = set()

    def sanitize(email) -> str:
        # seperate the domain fron the localname
        local_name, domain_name = email.split("@")
        exact = ""
        for char in local_name:
            if char == "+":
                break
            if char != ".":
                exact += char
        return f"{exact}@{domain_name}"

    for i in emails:
        email = sanitize(i)
        unique_emails.add(email)

    # or you can make the code shorter by using python's list method
    # i'ld prefer the first one since it's creates more discussion with the interviewer
    unique_emails = set()
    for email in emails:
        local_name, domain_name = email.split('@')
        local_name = local_name.split('+')[0]
        local_name = local_name.replace('.', '')
        email = local_name + '@' + domain_name
        unique_emails.add(email)
    # return len(unique_emails)
    return len(unique_emails)


# 13 Detect Capital  - Easy
def detectCapitalUse(self, word: str) -> bool:
    start_upper = word[0].isupper()
    if len(word) == 1: return True

    if word[1].isupper() and not start_upper:
        return False

    for i in range(2, len(word)):
        if not start_upper and word[i].isupper():
            return False
        if word[i].isupper() and word[i - 1].islower():
            return False
        if word[i].islower() and word[i - 1].isupper():
            return False

    return True


# 14 Delete Problems to Make Sorted
def minDeletionSize(self, strs: List[str]) -> int:
    deleted = 0
    """the number of col is the same as the length of any word in the list
    the number of rows is the length of the list
    for each col, you go through every row until you find a wrong sorting """
    for col in range(len(strs[0])):
        for row in range(1, len(strs)):
            if strs[row][col] < strs[row - 1][col]:
                deleted += 1
                break
    return deleted


# 15 Minimum Round to complete Tasks  -- Medium
def minimumRounds(self, tasks: List[int]) -> int:
    hashmap = {}
    for task in tasks:  # store the occurence of the numbers in a hashmap
        hashmap[task] = 1 + hashmap.get(task, 0)
    rounds = 0
    for num in tasks:
        value = hashmap[num]
        # if the occurence is 1 we want to return -1 since we must complete 2 or 3 tasks for each round
        if value == 1: return -1
        if value == 0: continue  # skipping cos this means we are done with the task
        # the optimal way is to subtract 3 if the number is odd or is divisible by 3
        hashmap[num] += -3 if value % 2 or not value % 3 else -2
        round += 1
    return rounds


# a slightly optimised version of the previous question
def minimumRounds2(self, tasks: List[int]) -> int:
    hashmap = {}
    for task in tasks:  # store the occurence of the numbers in a hashmap
        hashmap[task] = 1 + hashmap.get(task, 0)
    rounds = 0
    frequency = hashmap.values()  # contains the frequency of the numbers
    for num in frequency:
        if num == 1: return -1
        rounds += (num + 2) // 3  # calculates the max completable task based on the frequency
    return rounds


# 16 - Max Points on a Line
def maxPoints(self, points: List[List[int]]) -> int:
    length = len(points)
    if length <= 2: return length  # any two points are typically on the same line

    def slope(pointA, pointB):
        if pointA[0] == pointB[0]:  # infinte gradient if they are on the same vertical line
            return math.inf
        gradient = (pointB[1] - pointA[1]) / (pointB[0] - pointA[0])
        return gradient

    maxPoints = 0
    for i in range(length):
        hashmap = {}
        for j in range(i + 1, length):
            gradient = slope(points[i], points[j])
            hashmap[gradient] = 1 + hashmap.get(gradient, 0)
        if hashmap:
            maxPoints = max(maxPoints, max(hashmap.values()) + 1)  # the + 1 is for the starting point
    return maxPoints


# 17 Pascals Triangle
def generate(self, numRows: int) -> List[List[int]]:
    result = [[1]]
    for i in range(numRows - 1):
        temp = [0] + result[-1] + [0]
        level = []
        for i in range(len(temp) - 1):
            level.append(temp[i] + temp[i + 1])
        result.append(level)
    return result


def minFlipsMonoIncr(self, s: str) -> int:
    """The Algorithm here is a bit tricky so i'll try to explain well to my future self
    We basically wanna keep the zeroes to the left and the ones to the right
    Whenever we get to a '1', we increase the number of ones we've seen so far
    Once we get to a '0', we can either flip the 0 to 1, or flip all the '1's we have seen before the zero
    Pretty simple but really hard to come up with
    """
    flips, countOnes = 0, 0
    for char in s:
        if char == "1":
            countOnes += 1
        else:
            flips = min(flips + 1, countOnes)
    return flips


def maxSubarraySumCircular(self, nums: List[int]) -> int:
    # using kandene's algorithm in a weird way
    # hey Future Dammy, just check neetcode video on it if you've forgotten the logic
    currSum, currMin, maxSum, maxMin = 0, 0, nums[0], nums[0]
    for num in nums:
        currSum = max(currSum, currSum + num)
        currMin = min(currMin, currMin + num)
        maxSum = max(currSum, maxSum)
        maxMin = min(currMin, maxMin)
    return max(maxSum, sum(nums) - maxMin) if maxSum >= 0 else maxSum


# Subarray Sums Divisible by K
def subarraysDivByK(self, nums: List[int], k: int) -> int:
    # we use the prefix sum idea here to but in a sligtly different way
    # Probably need to come back to this, don't think i get the logic properly
    prefixSum = {0: 1}
    currSum = ans = 0
    for num in nums:
        currSum += num
        key = currSum % k
        if key in prefixSum:
            ans += prefixSum[key]
            prefixSum[key] += 1
        else:
            prefixSum[key] = 1
    return ans
