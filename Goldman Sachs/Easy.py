import math
from typing import List, Optional
from heapq import heappush
import heapq
from Linkedlist import ListNode


# 1 Check if the Sentence Is Pangram
def checkIfPangram(self, sentence: str) -> bool:
    # create an hashset to hold unique characters, keep adding the characters to the hashset as
    # you go through the list, if the length of the set gets to 26 at any point return True
    hashset = set()
    for char in sentence:
        hashset.add(char)
        if len(hashset) == 26:
            return True
    return False


# 2 High Five
def highFive(self, items: List[List[int]]) -> List[List[int]]:
    # we use a priority queue approach since we sort of need a top k(k = 5 here)
    # we use an hashmap to map the student ids to their scores(we build this as a
    # max heap)
    # For every of the id in the hashmap, we compute get the top 5 by popping from the maxHeap
    # compute the average and append the id,average to the result
    # return the result in sorted order
    # runtime is O(Nlogn)
    hashmap = {}

    for id, score in items:
        if id not in hashmap:
            hashmap[id] = []
        heappush(hashmap[id], -score)  # using negative here since python only offers min heap

    result = []

    # compute the average for each id
    for id in hashmap:
        scores = hashmap[id]
        sum_five = 0
        for _ in range(5):
            sum_five += abs(heapq.heappop(scores))
        result.append([id, sum_five // 5])

    return sorted(result)


# 3 Pascals Triangle
def generate(self, numRows: int) -> List[List[int]]:
    # runtime - O(n^2), space - in the worst casem our temp variable would hold n numbers, O(n)
    result = [[1]]
    for i in range(numRows - 1):
        temp = [0] + result[-1] + [0]
        row = []
        for j in range(len(temp) - 1):
            row.append(temp[j] + temp[j + 1])
        result.append(row)
    return result


# 4 Judge Robots
def judgeCircle(self, moves: str) -> bool:
    x = y = 0  # to represent the initial position of the robots
    for move in moves:
        if move == "U":
            y += 1
        elif move == "D":
            y -= 1
        elif move == "R":
            x += 1
        else:
            x -= 1
    return x == y == 0


# 5 Best Time to Buy and Sell Stock
def maxProfit(self, prices: List[int]) -> int:
    buy_price = prices[0]
    maxProfit = 0
    for i in range(1, len(prices)):
        sell_price = prices[i]
        if sell_price < buy_price:
            buy_price = sell_price
        else:
            maxProfit = max(maxProfit, sell_price - buy_price)
    return maxProfit


# 6 First Unique Character in a String
def firstUniqChar(self, s: str) -> int:
    hashmap = {}  # map each character to their occurence
    for i in s:
        hashmap[i] = hashmap.get(i, 0) + 1
    for index, value in enumerate(s):  # return the first value that occurs once
        if hashmap[value] == 1:
            return index
    return -1


# 7 Rotate Strings
def rotateString(self, s: str, goal: str) -> bool:
    # the first thing you migth think of is to split the first string into array
    # and keep keep rotating within the range of its length, then for every rotation
    # we compare it with second string, we return False if there was no match

    # another approach is to double the first string, by doubling it gives every posible string
    # that would have been gotten by rotating it one by one
    # now we only need to check if the new string contains the second string
    return len(s) == len(goal) and (goal in s + s)

    # using the in operator to check here has a runtime of O(n^2), to reduce this to o(n) you can
    # a popular string matching algorithm called KMP(Kunnath...whatever)
    # the complexity is an overkill for a leetcode "Easy" question


# 8 Merge Sorted Arrays
def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    Do not return anything, modify nums1 in-place instead.
    """
    last_index = m + n - 1
    m, n = m - 1, n - 1
    while m >= 0 and n >= 0:
        if nums1[m] > nums2[n]:
            nums1[last_index] = nums1[m]
            m -= 1
        else:
            nums1[last_index] = nums2[n]
            n -= 1
        last_index -= 1
    # if there are still values in nums2
    while n >= 0:
        nums1[last_index] = nums2[n]
        n -= 1
        last_index -= 1


# 9 Two Sum
def twoSum(self, nums: List[int], target: int) -> List[int]:
    hashmap = {}
    for index, value in enumerate(nums):
        if target - value in hashmap:
            return [index, hashmap[target - value]]
        hashmap[value] = index
    return []


# 10 Count Binary Strings
def countBinarySubstrings(self, s: str) -> int:
    """
    Check this for detailed explanation
    https://leetcode.com/problems/count-binary-substrings/solutions/1172569/short-easy-w-explanation-comments-keeping-consecutive-0s-and-1s-count-beats-100/
    """
    ans, prev, curr = 0, 0, 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            curr += 1
        else:
            ans += min(prev, curr)
            prev, curr = curr, 1
    ans += min(prev, curr)
    return ans


# 11 Linked list Cycle
def hasCycle(self, head: Optional[ListNode]) -> bool:
    # This can be easily solved using hashset to store the nodes as they are visited and return True if a
    # duplicate node is found(not a duplicate value as the different nodes might have the same value)
    # However the space complexity is o(n) in worst case and this can be optimized to constant time by using
    # flood cycle's algorithm to detect cycle by using a slow and fast pointer to fo through the linked list
    # we move the slow pointer once and the fast pointer twice, if there is a cycle these two would eventually
    # catch up
    fast = slow = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
        if fast == slow:
            return True
    return False


# 12 Find Pivot Index
def pivotIndex(self, nums: List[int]) -> int:
    # the naive approach would be that as we go through the array,
    # For every index, we compute sum of the values to the right and also do the same for the values to the left
    # if the two sum is equal, we return the current index
    # However this is 0(n^2) since we are computing a sum(O(n)) in every iteration
    # to reduce the runtime to O(n), we can compute the left sum as we move through the array
    # to get the right sum, we subtract the left sum and current value from the total sum of the array
    leftSum, total = 0, sum(nums)
    for i in range(len(nums)):
        rightSum = total - leftSum - nums[i]
        if rightSum == leftSum:
            return i
        leftSum += nums[i]
    return -1


# 13 Pascal Triangle II
def getRow(self, rowIndex: int) -> List[int]:
    # so the first approach we can use here is to simple generate r number of rows for the pascal triangle
    # and then return the last row in our list as the answer -- rutime O(n^2)
    # but that can be optimized using some basic knowledge of maths and combinations
    # Here are few things to note
    # The kth row in a pascal triangle will have k + 1 numbers
    # to get a value at a row or column in a pascal triangle can be done using combination
    # value at nth row and kth column = nCk = n!/(n-k)!k! == n*(n-1)*...(n-k+1)/k!
    # e.g 5C4 == 5*4*3*2/1*2*3*4
    # so basically we can initiliase the numerator to the row number given and denominator to 1
    # and then basically decrease the numerator and increase the numerator as we try to figure the value at each column
    # of the rth row
    # use the link below incase all this still sounds like gibberish
    # https://leetcode.com/problems/pascals-triangle-ii/solutions/1203260/very-easy-o-n-time-0-ms-beats-100-simple-maths-all-languages/
    row = [1] * (rowIndex + 1)
    up, down = rowIndex, 1
    for i in range(1, rowIndex):
        row[i] = int(row[i - 1] * up / down)
        up -= 1
        down += 1
    return row


# 14 Fibonnaci Sequence
def fib(n: int) -> int:
    # The most common and inefficient approach is to do this recursively -- O(2^n)
    # if n <= 1:
    #     return n
    # return fib(n - 1) + fib(n - 2)

    # dynamic programming approach -- bottom up (reuse previously computed result if they reappear - O(n)
    fib = [0] * (n + 1)
    fib[1] = 1
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib[n]


# 15 Valid Anagram
def isAnagram(self, s: str, t: str) -> bool:
    # This is a simple hashmap problem, the code is pretty explanatory
    if len(s) != len(t):  # edge case
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


# 17 Counting Bit
def countBits(self, n: int) -> List[int]:
    # one approach is to have a helper function that converts the number to binary and count the number of  - O(nlogn)
    # I'll come back to optimize this using dynamic programming or bit manipulation
    def count(num: int):
        count = 0
        while num != 0:
            if num % 2 == 1:
                count += 1
            num //= 2
        return count

    result = []
    for i in range(n + 1):
        result.append(count(i))
    return result


def countBits(self, n: int) -> List[int]:
    # A dynamic programming approach to the question to reduce the runtime to O(n)
    # https://www.youtube.com/watch?v=RyBM56RIWrM Neetcode explains here well!
    # You can talk about this if your interviewer wants you to provide a better approach
    dp = [0] * (n + 1)
    offset = 1
    for i in range(1, n + 1):
        if offset * 2 == i:
            offset = i
        dp[i] = 1 + dp[i - offset]
    return dp


# 18 Sort Integers by The Number of 1 Bits
def sortByBits(self, arr: List[int]) -> List[int]:
    # Let's count the number of 1 Bit for each number and store in a list
    def count(n):
        # we can use the flip flop method: using the AND logic on a number n and n - 1 flips the least significant
        # 1 to zero or let's use the usual method of( right shifting and AND logic )since that's more intuitive and
        # can easily be explained to the interviewer
        count = 0
        for _ in range(32):
            if n & 1:
                count += 1
            n >>= 1
        return count

    bitsmap = []
    for i in arr:
        bitsmap.append([count(i), i])
    bitsmap.sort()  # clarify if you're allowed to use an inbuilt function since the question talks about sorting
    return [bits[1] for bits in bitsmap]


# 19 Climbing Stairs
def climbStairs(self, n: int) -> int:
    # If you read the problem you'll notice a recurrence relation that the number of ways to get to let's say the
    # the third step is the same as the sum of the number of ways to get to the 2nd step and the first step
    # dp[i] = dp[i-1] + dp[1-2].. Therefore we can solve this problem with either bottom up or top down dynamic pro
    # gramming approach

    # Bottom Up Approach
    if n <= 2: return n  # edge case
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

    # Top Down Approach with memoization
    hashmap = {}

    def dp(n):
        if n <= 2: return n
        if n not in hashmap:
            hashmap[n] = dp(n - 1) + dp(n - 2)
        return hashmap[n]

    return dp(n)


# 20 Minimum Value to Get Positive Step by Step Sum
def minStartValue(self, nums: List[int]) -> int:
    # we precompute the sum using 0 as a start value and get the minimum of the step by step sum
    # Our minimum start value should be able to make the current minimum step by step sum equal to exactly 1
    total, minStep = 0, 0
    for num in nums:
        total += num
        minStep = min(total, minStep)


# 21 Greatest Common Divisor of Two Strings
def gcdOfStrings(self, str1: str, str2: str) -> str:
    # we can approach this mathematically
    # if two numbers have a common divisor, this means that two numbers can be expressed as a multiple of that divisor
    # so looking at the below example
    # a = 4, b = 6..we see that ab (4 * 6)->[2*2*2*2*2] = ba (6 * 4)->[2*2*2*2*2]
    # Therefore two strings would have a common divisor if the concantentation of the two strings are equal in both ways
    # stringa + stringb = stringb + stringa
    # Once we know this, we just need to find the gcd of the two strings lenght and return the string up to that length
    if str1 + str2 != str2 + str1:
        return ""
    gcdLength = math.gcd(len(str1), len(str2))
    return str1[:gcdLength]

    return 1 - minStep
