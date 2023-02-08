import heapq
import math
import random
from typing import List, Optional
from functools import cmp_to_key
from Linkedlist import ListNode
from Trees import TreeNode


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


# 2 Best time to buy and sell stock II -- Greedy
def maxProfit(self, prices: List[int]) -> int:
    # we only buy when we know the price would rise the next day
    maxProfit = 0
    for index in range(len(prices) - 1):
        if prices[index] < prices[index + 1]:
            maxProfit += prices[index + 1] - prices[index]
    return maxProfit


# 3 Delete duplicates from an unsorted linked list  -- Linked List
def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
    # get the occurence of each node
    # create a new Linked list and only add nodes that occur once
    counter = {}  # {value:frequency}
    current = head
    while current:
        counter[current.val] = 1 + counter.get(current.val, 0)
        current = current.next

    newHead = ListNode(0, head)  # build a new linked list starting from the head and remove nodes that occur more than
    # once, adding the head as the next node of the linked list instead of it being the first node caters for the edge
    # case where the head's value occurs more than once
    current = newHead
    while current.next:
        if counter[current.next.val] > 1:
            current.next = current.next.next
        else:
            current = current.next
    return newHead.next


# 4 Palindromic Substring -- Two Pointers
def countSubstrings(self, s: str) -> int:
    # The most intuitive approach is to take each character and check it agains the remaining characters to
    # see if it forms a palindrome. However the runtime is O(n^3)[O(n^2) for nested for loop * O(n) for checking
    # if it forms a palindrome
    # we can optimize the runtime to O(n^2) by assuming each character in the string is a middle of another subtring
    # and then expands towards the left and right( we do this for odd and even lenght palindrome)

    def countpalindrome(left, right):  # an helper function
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count

    result = 0
    for index in range(len(s)):
        result += countpalindrome(index, index)  # odd palindromic length
        result += countpalindrome(index, index + 1)  # even palindromic length
    return result


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


# 6 Break a Palindrome -- Greedy
def breakPalindrome(self, palindrome: str) -> str:
    # we want to be greedy with our approach here
    # basically to break a palindrome in the least lexological order, we want to change the first letter we see to
    # 'a'(that's if it not 'a' already).. the edge case is if it happens that all the letter are a then we want to
    # change the last letter to b
    length = len(palindrome)
    if length == 1:
        return palindrome
    for i in range(length // 2):  # going through half of the string since it's a palindrome
        if palindrome[i] != "a":
            return palindrome[:i] + "a" + palindrome[i + 1:]
    return palindrome[:-1] + "b"


# 7 Length of longest substring with non - repeating characters -- Sliding Window
def lengthOfLongestSubstring(self, s: str) -> int:
    hashset = set()  # sliding window to check for duplicates
    start, maxLength = 0, 0
    for end in range(len(s)):
        while s[end] in hashset:
            hashset.remove(s[start])
            start += 1
        hashset.add(s[end])
        maxLength = max(maxLength, len(hashset))
    return maxLength


# 8 LRU Cache  -- Linked List
class Double:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = self.prev = None


class LRUCache:
    # This problem is actually straightforward
    # We only need a data structure that can keep track of items as they are added and can also push them to the
    # front whenever they are used, to get the least recently used we simply return the item at the end.
    # This is where double linked list come in because now we can model the cache as a linked list and specifically
    # a double linked list since they have 'previous' pointers to allow nodes to be push to the front wheneve they
    # are used and also allows insertion and removal of nodes possible without having a reference to the head
    # the head of the double will point to the most recently used while the tail points to the least

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = self.tail = Double(0, 0)
        # the tail and the head should point initially to each other
        self.head.prev, self.tail.next = self.tail, self.head

    # helper method for the Double linked list
    def insert(self, node):
        prev, current = self.head.prev, self.head
        prev.next = current.prev = node
        node.next, node.prev = current, prev

    def remove(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self.remove(node)
            self.insert(node)  # push the cache to the front since it has just been used
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        node = Double(key, value)
        self.insert(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            node = self.tail.next  # remove the least recenlty used cache
            self.remove(node)
            del self.cache[node.key]


# 9 Longest Palindromic Substring -- Two Pointers
def longestPalindrome(self, s: str) -> str:
    # the idea is to take each character in the string and expand around the centers
    # we would need to consider cases for odd even length
    result = ""
    maximum_length = 0

    def check(left, right):
        while left >= 0 and right < len(s) and s[right] == s[left]:
            length = right - left + 1
            if length > maximum_length:
                result = s[left:right + 1]
                maximum_length = result
            left -= 1
            right += 1

    for i in range(len(s)):
        # odd palindrome
        check(i, i)

        # even palindrome
        check(i, i + 1)
    return result


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
    return


# 11 Fraction to Recurring Decimal -- Maths and Geometry
def fractionToDecimal(self, numerator: int, denominator: int) -> str:
    """this is a actually very tricky problem but basically test our knowledge on how to code up mathematical
    division logic
    I'll try to break it down and explain each logic"""
    if numerator == 0: return "0"  # edge case

    # first take care of negative integers whether the numerator or denominator or both
    prefix = ""  # should be negative or empty(positive)
    if numerator < 0 and denominator > 0 or numerator > 0 and denominator < 0:
        prefix = "-"

    # make the numerators postive(this also covers the case where they are both negative
    numerator, denominator = abs(numerator), abs(denominator)

    # sort of simulate the division logic
    # One key thing to note here is that in a recurring decimal will occur whenever a remainder repeat itself
    decimals, remainders = [], []
    while True:
        decimal = numerator // denominator
        remainder = numerator % denominator
        decimals.append(str(decimal))
        remainders.append(remainder)

        numerator %= denominator
        numerator *= 10

        if numerator == 0:  # then there was no recurring decimal
            if len(decimals) == 1:
                return prefix + decimals[0]
            else:
                return prefix + decimals[0] + "." + str(decimals[1:])

        # if the remainder we just got has already occured before, then the fraction has a reccuring decimal
        if remainders.count(remainder) > 1:
            decimals.insert(remainders.index(remainder) + 1, "(")
            decimals.append(")")
            return prefix + decimals[0] + "." + str(decimals[1:])


# 12 Kth Largest element in an Array
def findKthLargest(self, nums: List[int], k: int) -> int:
    # the most intuitive approach is to sort the list first and return the element at the kth position
    # we can optimize the runtime from O(Nlogn) to O(Klogn) by using a max heap
    nums = [-i for i in nums]  # making every number to build the max heap since python supports min heap by default
    heapq.heapify(nums)
    kthlargest = 0
    for _ in range(k):
        kthlargest = -(heapq.heappop(nums))
    return kthlargest


# 13 Number of Islands
def numIslands(self, grid: List[List[str]]) -> int:
    # we use a depth first search to search recursively the neighbours(vertically and horizontally)
    # whenever we get to a position of the island(grid) that has a value of 1
    # we can also use a bfs here but i think the dfs code is more concise
    rows, cols = len(grid), len(grid[0])
    num_islands = 0

    def dfs(row, col):
        if not (row in range(rows) and col in range(cols)) or grid[row][col] != "1":
            return
        # mark the position as visited by changing the value (if the interveiwer permits the input grid modified)
        # or use a hashset to keep track of visited positions
        grid[row][col] = "*"
        dfs(row + 1, col)
        dfs(row - 1, col)
        dfs(row, col + 1)
        dfs(row, col - 1)

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == "1":
                dfs(row, col)
                num_islands += 1

    return num_islands


# 14 Three Sum
def threeSum(self, nums: List[int]) -> List[List[int]]:
    # the brute force approach would be to use three nested for loop to check every three pairs
    # that can be formed by each interger  - O(n^3)
    # we can reduce this runtime to 0(n^2), if we sort the array(remember the "Two Sum II" problem?)
    # pick each integer, and use the idea of (Two Sum II) question to see which two integers can add
    # up to the current integer to give us 0
    res = []
    nums.sort()  # we first sort the input array

    for i, j in enumerate(nums):
        # make sure that we're not starting with a number that has occured before cos if it has
        # it will definitely be in our result list
        if i > 0 and nums[i - 1] == j:
            continue

        # if it has not occured before, we now pick the number and find other two that will make the sum Zero
        # to do this we make use of two pointers, one will point to the index after the current index and the second
        # will point to the end of the list
        left, right = i + 1, len(nums) - 1
        while left < right:
            threesum = j + nums[left] + nums[right]
            if threesum > 0:
                # we shift the right pointer back by 1
                right -= 1
            elif threesum < 0:
                # you get the gist now lol
                left += 1
            else:  # we found a three numbers that sun up to zero
                res.append([j, nums[left], nums[right]])
                # we move our left pointer forward at this point and make sure
                # we don't move it to a position containing a value that has already been used
                left += 1
                while nums[left] == nums[left - 1] and left < right:
                    left += 1
        return res


# 15 Search a 2D Matrix
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    # the brute force(O(n^2)) approach is the most intuitive approach
    # However we can take advantage of the sorted property of the matrix and use
    # binary search to search through the matrix
    # first we figure out the row to search
    rows, cols = len(matrix), len(matrix[0])
    low, high = 0, rows - 1
    while low <= high:
        row = (low + high) // 2
        if target > matrix[row][-1]:  # we search the rows down if the target is more than the value in the current row
            low = row + 1
        elif target < matrix[row][0]:  # we search to the top if the target is less than the first number in the row
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


# 16 Largest Numbers ** check leetcode implementation on custom sorting without using the functools module
def largestNumber(self, nums: List[int]) -> str:
    # for each pairwise comparison during the sort, we compare the
    # numbers achieved by concatenating the pair in both orders
    nums = [str(i) for i in nums]  # convert all the integers to strings

    def compare(n1, n2):  # define a custom sorting function
        if n1 + n2 > n2 + n1:
            return -1
        else:
            return 1

    nums.sort(key=cmp_to_key(compare))  # clarify with your interviewer if you're allowed to use an imported module

    # take care of a minor edge case where all the numbers are 0
    # it should return just zero if that's the case
    return "".join(nums) if nums[0] != "0" else "0"


# 17 Meeting Rooms   -- Greedy
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    # basically for every meeting, we need to know if it starts a time less than the end time of another
    # meeting, if this happens we know the two meetings would overlap so we allocate a seperate room for the
    # current meeting
    # since it's obvious we want to make the best use of the number of rooms we allocate, we have to be
    # greedy with this approach
    # let's seperate the meetings into start time and end time and sort them seperately
    starting_times = sorted([time[0] for time in intervals])
    ending_times = sorted([time[1] for time in intervals])

    rooms, start, end = 0, 0, 0
    maxrooms = 0
    while start < len(intervals):
        if starting_times[start] < ending_times[end]:
            # the starting time of the current meeting overlaps with the ending time of another meeting
            rooms += 1
            start += 1
        else:
            rooms -= 1
            end += 1
        maxrooms = max(rooms, maxrooms)
    return maxrooms


# 19 Maximum SubArray -- Greedy
def maxSubArray(self, nums: List[int]) -> int:
    currSum = 0
    maxSum = 0
    for num in nums:
        if currSum < 0:  # if the sum of the previous subarray is negative, we reset it to zero
            currSum = 0  # cos we are really after getting the maximum subarray
        currSum += num
        maxSum = max(maxSum, currSum)
    return maxSum


# 20 Find Minimum in a rotated Sorted Array  **
def findMin(self, nums: List[int]) -> int:
    # we use a modified binary search algorithm since the array is still somehow rotated
    # https://www.youtube.com/watch?v=nIVW4P8b1VA ,Neetcode explains well here!
    left, right = 0, len(nums) - 1
    if len(nums) == 1: return nums[0]
    minimum = nums[0]
    while left <= right:
        if nums[left] <= nums[right]:  # the array is properly sorted at this point
            return min(minimum, nums[left])
        mid = (left + right) // 2
        minimum = min(minimum, nums[mid])
        if nums[mid] >= nums[left]:  # we search to the right when we're in the left sorted portion
            left = mid + 1
        else:  # we search to the left when we're in the left sorted portion of the array
            right = mid - 1


# 21 Minimum Size SubArray -- Sliding Window
def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    # sliding window approach
    window, start, minLength = 0, 0, len(nums)
    for end in range(len(nums)):
        window += nums[end]
        while window >= target:
            minLength = min(end - start + 1, minLength)
            window -= nums[start]
            start += 1
    return minLength if minLength != len(nums) else 0


# 22 Container with most Water  -- Two Pointers
def maxArea(self, height: List[int]) -> int:
    # the brute force would be to have every possible combination of the heights - O(n^2)
    # we can optimize it to linear time using two pointers approach,
    # we set our left and right pointers to the first and last height respectively and we only shift the
    # pointers that have a lower height
    left, right, maximumArea = 0, len(height) - 1, 0
    while left <= right:
        area = (right - left) * min(height[left], height[right])
        maximumArea = max(maximumArea, area)
        if height[left] <= height[right]:
            left += 1
        else:
            right -= 1
    return maximumArea


# continue
# 23 Word Search   --  Come back to this
def exist(self, board: List[List[str]], word: str) -> bool:
    # This can be solved with backtracking
    rows, cols = len(board), len(board[0])

    # visited = set()  once we use a character once during our backtracking, we can't use it again

    def dfs(row, col, index):
        if index == len(word):
            return True
        if col not in range(cols) or row not in range(rows) or board[row][col] != word[index] or \
                board[row][col] == ".":
            return False

        temp, board[row][col] = board[row][col], "."
        result = dfs(row + 1, col, index + 1) or \
                 dfs(row - 1, col, index + 1) or \
                 dfs(row, col + 1, index + 1) \
                 or dfs(row, col - 1, index + 1)
        board[row][col] = temp
        return result

    for row in range(rows):
        for col in range(cols):
            if dfs(row, col, 0): return True
    return False


# 24 Rotting Oranges  -- Graph(Bfs)
def orangesRotting(self, grid: List[List[int]]) -> int:
    # This problems becomes easier when you view it as a graph and imagine the rotting oranges are somehow the
    # roots of the graph
    # we'll use a bfs approach here(not dfs) since we can have multiple roots(rotting oranges)
    rows, cols = len(grid), len(grid[0])  # just like every grid problem
    fresh_oranges, time = 0, 0
    # get the number of fresh oranges and append the dimensions of the rotten oranges to the queue that will be used
    # for our bfs algorithm
    queue = []
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 2:
                queue.append([row, col])
            if grid[row][col] == 1:
                fresh_oranges += 1

    # time for our bfs logic
    directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]  # for checking the adjacent oranges
    while queue and fresh_oranges > 0:
        for i in range(len(queue)):
            row, col = queue.pop(0)
            for r, c in directions:
                if row + r not in range(rows) or col + c not in range(cols) or grid[row + r][col + c] != 1:
                    continue
                grid[row + r][col + c] = 2
                queue.append([row + r, col + c])  # add the dimension since the orange is also now rotten
                fresh_oranges -= 1
        time += 1
    return time if fresh_oranges == 0 else -1


# 25 Delete and Earn -- dynamic programming
def deleteAndEarn(self, nums: List[int]) -> int:
    # I'll explain the logic to solving the problem using Bottom Up Approach
    maxNumber = max(nums)
    points = {i: 0 for i in range(maxNumber + 1)}
    # First we map each number to get the points we can get from it and get the maximum of the numbers at the same time
    for num in nums:
        points[num] = num + points.get(num, 0)
    maxEarnings = [0] * (maxNumber + 1)
    maxEarnings[1] = points[1]
    # The maximum you can earn at any points will the max of you taking the current price n (which means you can't take
    # the previous one (n-1) or you ignore the current the current price which means you've taken (n-1)
    for i in range(2, maxNumber):
        maxEarnings[i] = max(maxEarnings[i - 1], maxEarnings[i - 2] + points[nums[i]])
    return maxEarnings[maxNumber]


# 26 Coin Change  **
def coinChange(self, coins: List[int], amount: int) -> int:
    # Neetcode explains well here! https://www.youtube.com/watch?v=H9bfqozjoqs
    dp = [math.inf] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if i - coin >= 0:
                dp[i] = min(dp[i], 1 + dp[i - coin])
    result = dp[amount]
    return result if result != math.inf else -1


# 27 House Robber -- Dynamic Programming-- check neetcode video on reducing the space to constant
def rob(self, nums: List[int]) -> int:
    # at every house we basically have two choices
    # 1. To rob the current house(which means we were not able to rob the previous house)
    # 2. Not rob the current house, the money we have so far is what we've up to the previous house
    # This is another dynamic programming because we are after an optimal value and the future decisions are affected
    # by earlier decisions
    n = len(nums)
    robbing = [0] * (n + 1)
    robbing[0] = nums[0]  # if you're at the first house you have a choice of robbing the house or not robbing at all
    robbing[1] = max(nums[0], nums[1])  # if you have just two houses left to rob, you rob the house that has more lol
    for i in range(2, n + 1):
        robbing[i] = max(robbing[i - 1], nums[i] + robbing[i - 2])
    return robbing[n]


# 28 Decode Ways  **
def numDecodings(self, s: str) -> int:
    # Basically at every index we can either decode it as a single digit or a double digits provided that it is valid
    # The current index becomes invalid if it's equal to zero "0" and the double digits is invalid if it's more than 26
    # This will be solved recursively(2^n) but will reduced to 0(n) using memoization since the recursive functions have
    # overallaping calls
    cache = {}

    def dfs(index):
        if index == len(s):  # we found a valid decodings
            return 1
        if s[index] == "0":  # we found an invalid decoding
            return 0
        if index == len(s) - 1: return 1
        if index not in cache:
            res = dfs(index + 1)  # as a single digit
            if int(s[index:index + 2]) <= 26: res += dfs(index + 2)
            cache[index] = res
        return cache[index]

    return dfs(0)


# 29 K diff pairs in an array  Hashmaps
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


# 30 String to Integer Atoi
def myAtoi(self, s: str) -> int:
    index, sign, n = 0, 1, len(s)
    result, maxInt, minInt = 0, pow(2, 31) - 1, -pow(2, 31)

    # let's ignore leading whitespaces
    while index < n and s[index] == " ":
        index += 1

    # let's check if it has a negative sign
    if index < n and s[index] == "-":
        sign = -1
        index += 1
    elif index < n and s[index] == "+":
        index += 1

    # keep getting the digits until a non digit is met
    while index < n and s[index].isdigit():
        # check for overflow and underflow
        value = int(s[index])
        if result > maxInt // 10 or (result == maxInt // 10 and value > maxInt % 10):
            return maxInt if sign == 1 else minInt

        # if the code get's here then there would be no overflow after adding the current value
        result = result * 10 + value
        index += 1
    return sign * result


# 31 Asteroid Collision
def asteroidCollision(self, asteroids: List[int]) -> List[int]:
    # we use a stack to keep track of the asteroids that have not collided
    # a collision only happens if the item at the asteroid is moving to the the right and the one about to be added
    # is moving to the left direction(this can be checked with the sign)
    ans = []
    for new in asteroids:
        while ans and new < 0 < ans[-1]:
            if ans[-1] < -new:
                ans.pop()
            elif ans[-1] > -new:
                new = 0
            elif ans[-1] == -new:
                ans.pop()
                new = 0
        if new:
            ans.append(new)
    return ans


# 32 Remove Adjacent Duplicate from a string  II
def removeDuplicates(self, s: str, k: int) -> str:
    stack = []
    for char in s:
        if stack and stack[-1][0] == char:
            stack[-1][1] += 1
        else:
            stack.append([char, 1])
        if stack[-1][1] == k:
            stack.pop()
    result = ""
    for char, count in stack:
        result += char * count
    return result


# 33 Generate Paranthesis
def generateParenthesis(self, n: int) -> List[str]:
    # This is a backtracking problem
    result = []

    def backtrack(opened, close, current):
        if opened == close == n:
            result.append(current)
            return

        if opened < n:
            backtrack(opened + 1, close, current + "(")

        if close < opened:
            backtrack(opened, close + 1, current + ")")

    backtrack(0, 0, "")
    return result


# 34 Unique Paths
def uniquePaths(self, m: int, n: int) -> int:
    # build the grid
    grid = [[1] * n for _ in range(m)]
    for row in range(1, m):
        for col in range(1, n):
            grid[row][col] = grid[row - 1][col] + grid[row][col - 1]

    return grid[m - 1][n - 1]


# 35  def minPathSum(self, grid: List[List[int]]) -> int:
def minPathSum(self, grid: List[List[int]]) -> int:
    # This is another dynamic programming problem that will be solved with bottom up approach
    rows, cols = len(grid), len(grid[0])
    dp = [[math.inf] * (cols + 1) for _ in range(rows + 1)]
    dp[rows - 1][cols] = 0  # come back to why i am doing rows - 1

    for row in range(rows - 1, -1, -1):
        for col in range(cols - 1, -1, -1):
            dp[row][col] = grid[row][col] + min(dp[row][col + 1], dp[row + 1][col])

    return dp[0][0]


# 36 Jump Game
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


# 37 Find Winner of circular game
def findTheWinner(self, n: int, k: int) -> int:
    # build players
    players, index = [player + 1 for player in range(n)], 0
    while players > 1:
        index = (index + k - 1) % len(players)  # using mod allows us to avoid overflow
        players.pop(index)
    return players[0]


# 38 Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts
def maxArea(self, h: int, w: int, horizontalCuts: List[int], verticalCuts: List[int]) -> int:
    # If you observe carefully, you would realize that the maximum area of area of cake has a width that is equal
    # to the maximum width gotten after applying only the vertical cuts and has a height equal to the maximum
    # height after applying only the horizontal cuts
    # we first sort the array to ensure the cuts are beside each other
    horizontalCuts.sort()
    verticalCuts.sort()

    # get the maximum height
    # consider the edges first
    max_height = max(horizontalCuts[0], h - horizontalCuts[-1])
    for i in range(1, len(horizontalCuts)):
        max_height = max(horizontalCuts[i] - horizontalCuts[i - 1], max_height)

    # get the maximum width
    # consider the edges first
    max_width = max(verticalCuts[0], w - verticalCuts[-1])
    for i in range(1, len(verticalCuts)):
        max_width = max(verticalCuts[i] - verticalCuts[i - 1], max_width)

    return (max_height * max_width) % (10 ** 9 + 7)


# 39 Minimum costs to connect Sticks -- Heaps
def connectSticks(self, sticks: List[int]) -> int:
    # we basically use a min heap to because for an optimal solution, we need to
    # join the two sticks with minimum values as we iterate through the array
    heapq.heapify(sticks)
    minCost = 0
    while sticks:
        cost = heapq.heappop(sticks) + heapq.heappop(sticks)
        minCost += cost
        heapq.heappush(cost, sticks)
    return minCost


# 40 Jump game -- Greedy Algorithm
def canJump(self, nums: List[int]) -> bool:
    goal = len(nums) - 1
    for i in range(len(nums) - 2, -1, -1):
        if i + nums[i] >= goal:
            goal = i

    return goal == 0


# 41 Robots bounded in a circle  -- Maths and Geometry
def isRobotBounded(self, instructions: str) -> bool:
    """The basic algorithm of this problem is that the robot will be bounded in a circle if either the position after
    going through the movements once is unchanged or if there is a change in direction after going through the movements
    To change the direction of a given cordinate by 90 degrees, we multiply the coordinate by rotation matrix...
    For clockwise, we multiply it by matrix [[0,1],[-1,0]] and for anti clockwise, we multiply it by [[0,-1],[1,0]"""

    dirX, dirY = 0, 1  # it initialy faces north
    x, y = 0, 0
    for movement in instructions:
        if movement == "G":
            x, y = x + dirX, y + dirY
        elif movement == "L":
            dirX, dirY = -1 * dirY, dirX
        else:
            dirX, dirY = dirY, -1 * dirX
    return (x, y) == (0, 0) or (dirX, dirY) != (0, 1)


# 42 Max Diff you can get from changing an Integer  -- Greedy Algorithm
def maxDiff(self, num: int) -> int:
    # We can approach this with greedy algorithm the max integer from changing the integer would be changing the
    # first digit in the number that is not's a 9 to a 9 the min integer would be changing either the first integer(
    # if it's not already a 1) or changing the next digits to zero if it's not already a zero.. we can't change the
    # first digit to zero since we don't want to have leading zeroes
    num = str(num)
    maxInt = num
    for x in num:
        if x != '9':
            maxInt = num.replace(x, '9')
            break

    minInt = num
    for x in num:
        if num[0] == x and x != '1':  # change the first integer to 1 if it's not already 1
            minInt = num.replace(x, '1')
            break
        if num[0] != x and x != '0':  # else change the next zero digit to a zero
            minInt = num.replace(x, '0')
            break
    return int(maxInt) - int(minInt)


# 43 Count Number of Teams -- Dynamic Programming
def numTeams(self, rating: List[int]) -> int:
    # This is a bit tricky..would have to come back to this
    n = len(rating)
    up = [0] * n
    down = [0] * n
    teams = 0
    for i in range(n):
        for j in range(i):
            if rating[j] > rating[i]:
                up[j] += 1
                teams += up[i]
            else:
                down[j] += 1
                teams += down[i]
    return teams


# 44 Delete Node in a linked list  -- Linked List
def deleteNode(self, node):
    nextNode = node.next  # store the next node temporarily
    node.val = nextNode.val
    node.next = nextNode.next
    nextNode = None  # delete the next Node from memory


# 45 Product of Array Except Self -- Arrays
def productExceptSelf(self, nums: List[int]) -> List[int]:
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


# 46 Binary Tree Right Side View -- Trees
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    # we basically do a bfs and return the node at the right most side of each level
    if not root: return []  # edge case
    queue, result = [root], []
    while queue:
        for i in range(len(queue)):  # go through each level
            node = queue.pop(0)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(node.val)
    return result


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


# 48 Min Stack -- Stack
class MinStack:
    """Create a main stack and another monotically decreasing stack for keeping track of
    minimum values as they are added to the main stack
    """

    def __init__(self):
        self.stack = []
        self.minStack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.minStack or self.minStack[-1] >= val:
            self.minStack.append(val)

    def pop(self) -> None:
        value = self.stack.pop()
        if value == self.minStack[-1]:
            self.minStack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minStack[-1]


# 49 Merge Intervals
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    """Two intervals overlap if the end time of the previous intervals is more than the start time of the current
    interval. Sorting the intervals by their start time make the intervals that can possibly be merged appear in a
    contigous run """
    intervals.sort(key=lambda x: x[0])
    merged_intervals = []
    for interval in intervals:
        # clarify with your interviewer if two intervals where the start time of one equals the end time can be consider
        # ed as overlapping
        if not merged_intervals or merged_intervals[-1][1] < interval[0]:
            merged_intervals.append(interval)
        else:
            # if they overlap, we pick the maximum end time
            merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
    return merged_intervals


# 50 Add Two Numbers  -- Linked List
def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummyNode = ListNode()  # create a new node to hold the sum
    dummy, carry = dummyNode, 0
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        value_sum = val1 + val2 + carry
        carry = value_sum // 10
        dummy.next = ListNode(value_sum % 10)
        dummy = dummy.next
        l1 = l1.next if l1 else 0
        l2 = l2.next if l2 else 0
    return dummyNode.next


# 51 Pow(x,n) -- Recursion
def myPow(self, x: float, n: int) -> float:
    # we can solve this recursively by breaking it down to sub problem
    if x == 0:  # edge case
        return 0

    def power(x, n):
        if n == 0:  # base case
            return 1
        value = power(x * x, n // 2)
        return value * x if n % 2 else value  # check for odd powers

    result = power(x, abs(n))
    return result if n >= 0 else 1 / result


# 52 Add Power II -- Linked list
def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    if not l1 and not l2:  # never forget to handle edge case
        return None

    # reverse the linked list and add
    def reverse(node):
        prev = None
        while node:
            temp = node.next
            node.next = prev
            prev = node
            node = temp
        return prev

    l1, l2 = reverse(l1), reverse(l2)
    carry, dummynode = 0, ListNode()
    pointer = dummynode
    while carry or l1 or l2:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        value_sum = val1 + val2 + carry
        carry = value_sum // 10
        pointer.next = ListNode(value_sum % 10)
        pointer = pointer.next
        l1 = l1.next if l1 else 0
        l2 = l2.next if l2 else 0
    return reverse(dummynode.next)


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


# 54 Decode String -- Stack
def decodeString(self, s: str) -> str:
    """ As we go through the string in one pass, we keep track of the current number until we get to an opening square
bracket(which also means that the previous string has been processed), so we append the currString * it'c count to the
stack thenreset the count and currentString
"""
    currString, currNum, stack = "", 0, []
    for i in s:
        if i == "[":  # the previous string (if any )has been processed
            stack.append(currNum)
            stack.append(currString)
            currString = ""
            currNum = 0
        elif i == "]":
            prevString = stack.pop()
            strCount = stack.pop()  # pop the number that was just before the opening sqaure brackets "["
            currString = prevString + (currString * strCount)
            # don't append to the stack just yet until you see another opening paranthesis
        elif i.isalpha():
            currString += i
        else:  # it's a digit
            currNum = currNum * 10 + int(i)
    return currString


# 55 Top K Frequent Elements -- Heaps and Hashmap
def topKFrequent(self, words: List[str], k: int) -> List[str]:
    """get the count of each words, build a max heap and pop k values out as our output
    .. You can first talk about sorting the list(containing the words and their frequency) before mentioning
    how heaps can helps optimize our approach"""
    hashmap = {}
    for word in words:
        hashmap[word] = 1 + hashmap.get(word, 0)
    word_count = [(-count, word) for word, count in hashmap.items()]  # using negative so i could max heap in python
    heapq.heapify(word_count)

    return [heapq.heappop(word_count)[1] for _ in range(k)]


# 56 Find the Index of the First Occurrence in a String
def strStr(self, haystack: str, needle: str) -> int:
    # let's do brute force for now
    # we can further optimize the brute force greatly but i'll come back to that when i understand string matching
    for i in range(len(haystack) + 1 - len(needle)):
        if haystack[i] == needle[0] and haystack[i:i + len(needle)] == needle:
            return i
    return -1


# 57 Insert Intervals -- Intervals
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    result = []
    for i in range(len(intervals)):
        # check the new interval end time against the current interval start time
        # if the end time is less than the start time, then they do not overlap
        # and this means it definitely won't overlap with the rest of the intervals
        if newInterval[1] < intervals[i][0]:
            result.append(newInterval)
            return result + intervals[i:]

        # if the start time of the new interval is more than the endtime of the current interval, then they also do
        # not overlap but there is a possibility of it overlapping with one of the remaining intervals

        elif newInterval[0] > intervals[i][1]:
            result.append(intervals[i])

        else:  # If the code get's here that means they overlap
            newInterval = [min(intervals[i][0], newInterval[0]), max(intervals[i][1], newInterval[1])]
    result.append(newInterval)  # an edge case where the new interval overlaps with the last interval in the list
    return result


# 58 Gas Stations -- Greedy Algorithm
def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
    total_tank, curr_tank = 0, 0
    starting_index = 0
    for i in range(len(gas)):
        remaining = gas[i] - cost[i]
        total_tank += remaining
        curr_tank += remaining
        if curr_tank < 0:
            starting_index = i + 1
            curr_tank = 0
    return starting_index if total_tank >= 0 else -1


# 59 Search in rotated Sorted Array -- Binary Search
def search(self, nums: List[int], target: int) -> int:
    """ There are a few things to note about rotated sorted arrays
    1. We still apply our binary search algorithm since they are still kinda sorted
    2. The array will have two sorted portion(left and right) and knowing the portion we are guides how we shift our low
    and high pointers
    3. Every value in the right portion are less than the values in the kleft portion
    """
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] >= nums[low]:  # we're in the left sorted portion
            if target < nums[low] or target > nums[mid]:
                low = mid + 1
            else:
                high = mid - 1
        else:  # we're in the right sorted portion
            if target < nums[mid] or target > nums[high]:
                high = mid - 1
            else:
                low = mid + 1
        return -1  # Target was not found


# 60 Two Sum II  -- Two Pointers
def twoSum(self, numbers: List[int], target: int) -> List[int]:
    low, high = 0, len(numbers) - 1
    while low < high:
        value_sum = numbers[low] + numbers[high]
        if value_sum > target:
            high -= 1
        elif value_sum < target:
            low += 1
        else:
            return [low + 1, high + 1]


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


# 62 Maximum Size SubArray Sum - Sliding Window  -- can't be solved with sliding window..coming back to this
# after i understand prefix Sum properly
def maxSubArrayLen(self, nums: List[int], k: int) -> int:
    window_sum, start = 0, 0
    longest = 0
    for end in range(len(nums)):
        while window_sum == k:
            window_sum -= nums[start]
            start += 1
        window_sum += nums[end]
        longest = max(longest, end - start + 1)
    return longest if longest != 0 else -1


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


# 64 Permutations -- Bactkrackig
def permute(self, nums: List[int]) -> List[List[int]]:
    result = []
    if len(nums) == 1:  # base case
        return [nums[:]]

    for _ in range(len(nums)):
        temp = nums.pop(0)
        permutations = self.permute(nums)

        for perm in permutations:
            perm.append(temp)
        result.extend(permutations)
        nums.append(temp)
    return result


# 65 Find all Anagrams in a String -- Sliding window and Hashmap
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


# 66 Snake and Ladders -- Bfs(shortest path)
def snakesAndLadders(self, board: List[List[int]]) -> int:
    lenght = len(board)
    board.reverse()  # reverse the board makes our calculation easier

    def getPosition(value):  # an helper function to get the row and column if the value is not -1
        row = (value - 1) // lenght
        col = (value - 1) % lenght
        if row % 2:  # extra logic since the rows are alternating
            col = lenght - col - 1
        return [row, col]

    queue = [[1, 0]]  # [value,moves]
    visited = set()  # so we don't visit a position twice
    while queue:
        value, moves = queue.pop(0)
        for i in range(1, 7):  # let's throw a die
            nextValue = value + i
            row, col = getPosition(nextValue)
            if board[row][col] != -1:  # then there is a snake or ladder
                nextValue = board[row][col]
            if nextValue == lenght ** 2:  # we've gotten to the end of the board
                return moves + 1
            if nextValue not in visited:
                queue.append([nextValue, moves + 1])
                visited.add(nextValue)
    return -1


# 67  3 Sum closest -- Two pointers, Sorting
def threeSumClosest(self, nums: List[int], target: int) -> int:
    """the problem would be easier to solve if you've solved the following problems
    Two Sum, Two Sum II and 3 Sum
    """
    nums.sort()
    closest = math.inf
    for i in range(len(nums) - 2):
        left, right = i + 1, len(nums) - 1
        while left < right:
            threeSum = nums[i] + nums[left] + nums[right]
            if abs(threeSum - target) < abs(target - closest):
                closest = threeSum
            elif threeSum >= target:
                right -= 1
            else:
                left += 1
    return closest


# 68 Find Peak Element -- Binary Search
def findPeakElement(self, nums: List[int]) -> int:
    """Solving this with binary search is quite tricky
    I recommend reading the offical solution for a good explanation on why binary search works here despite the
    array not been necceasarily sorted"""
    low, high = 0, len(nums) - 1
    while low < high:
        mid = (low + high) // 2
        if nums[mid] < nums[mid + 1]:
            low = mid + 1  # we eliminate half of the array
        else:
            right = mid
    return low


# 69 Insert into a Binary Tree
def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    """For a binary search tree, every node is more than the all the values in it left subtree and less than all the
    values in it right subtree"""
    node = root
    while node:
        # if the value is more than the current node value, we either insert it as the right child if it has none or
        # we continue searching from the right child
        if val > node.val:  # we check the right subtree
            if not node.right:
                node.right = TreeNode(val)
                return root
            else:
                node = node.right
        else:
            if not node.left:
                node.left = TreeNode(val)
                return root
            else:
                node = node.left


# 70 Binary Search Iterator
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        current = root
        while current:
            self.stack.append(current)
            current = current.left

    def next(self) -> int:
        result = self.stack.pop()
        current = result.right
        while current:
            self.stack.append(current)
            current = current.left
        return result.val

    def hasNext(self) -> bool:
        return len(self.stack)


# 71 3 Sum Smaller
def threeSumSmaller(self, nums: List[int], target: int) -> int:
    count = 0
    nums.sort()
    for i in range(len(nums)):
        j, k = i + 1, len(nums) - 1
        while j < k:
            valueSum = nums[i] + nums[j] + nums[k]
            if valueSum < target:
                count += k - j
                j += 1
            else:
                k -= 1
    return count


# 72 Maximum Product SubArray -- Dynamic Programming
def maxProduct(self, nums: List[int]) -> int:
    # maintain a max and min subarray as you iterate through the array
    currMin, curMax = 1, 1
    totalMax = max(nums)
    for num in nums:
        if num == 0:
            currMin, currMax = 1, 1
            continue
        temp = currMax
        currMax = max(currMax * num, currMin * num, num)
        currMin = min(temp * num, currMin * num, num)

        totalMax = max(totalMax, curMax)
    return totalMax


# 73 Minimum moves to make array elements equal  -- Sorting
def minMoves(self, nums: List[int]) -> int:
    nums.sort()
    moves = 0
    for i in range(len(nums) - 1, -1, -1):
        moves += nums[i] - nums[0]
    return moves


# 76 Number of Sub Array Products less than k -- Sliding window
def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
    if k <= 1: return 0
    product = 1
    start, count = 0, 0
    for end in range(len(nums)):
        product *= nums[end]
        while product >= k:
            product /= nums[start]
            start += 1
        count = end - start + 1
    return count


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


# 78 Letter Combinations of a Phone Number -- Backtracking
def letterCombinations(self, digits: str) -> List[str]:
    result = []
    letters = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl",
               "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    if not digits:
        return []

    def backtrack(index, currStr):
        if len(currStr) == len(digits):
            result.append(currStr)
            return

        for char in letters[digits[index]]:
            backtrack(index + 1, currStr + char)

    backtrack(0, "")
    return result


# 79 Sort the Jumbled Numbers -- Arrays, Strings
def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
    result = []
    for value in nums:
        string = str(value)
        newString = ""
        for char in string:
            newString += str(mapping[int(char)])
        result.append([int(string), int(newString)])
    result.sort(key=lambda x: x[1])
    return [i[0] for i in result]


# 80 Find K Closest Elements --  Binary Search
def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
    # we simply use binary search to search for the left most index closer to x and return the 'k' lenght subarray
    # starting from that left index
    low, high = 0, len(arr) - k
    while low < high:
        mid = (low + high) // 2
        if x - arr[mid] > arr[mid + k] - x:
            # we shift our left pointer forward
            left = mid + 1
        else:
            right = mid
    return arr[left:left + k]
