import heapq
import math
from typing import List
from functools import cmp_to_key
from Linkedlist import ListNode


# 1 Group Anagrams
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
            counter[ord(i) - ord('a')] += 1
        return counter

    for string in strs:
        counts = getCount(string)
        group[tuple(counts)] = [string] + group.get(tuple(counts), [])

    return group.values()


# 2 Best time to buy and sell stock II
def maxProfit(self, prices: List[int]) -> int:
    # we only buy when we know the price would rise the next day
    maxProfit = 0
    for index in range(len(prices) - 1):
        if prices[index] < prices[index + 1]:
            maxProfit += prices[index + 1] - prices[index]
    return maxProfit


# 3 Delete duplicates from an unsorted linked list
def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
    # get the occurence of each node
    # create a new Linked list and only add nodes that occur once
    counter = {}
    pointer = head
    current = pointer
    while current:
        counter[current.val] = 1 + counter.get(current.val, 0)
        current = current.next

    newHead = ListNode(0, head)  # adding the head as the next node caters for the edge case where even the
    # head occurs more than once
    current = newHead
    while current.next:
        if counter[current.next.val] > 1:
            current.next = current.next.next
        else:
            current = current.next
    return newHead.next


# 4 Palindromic Substring
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


# 6 Break a Palindrome
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


# 7 Length of longest substring with non - repeating characters
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


# 8 LRU Cache
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
    # are used
    # the head of the double will point to the most recently used while the tail points to the least

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = self.tail = Double(0, 0)
        self.head.prev, self.tail.next = self.tail, self.head  # the tail and the head should point to each other

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


# 9 Longest Palindromic Substring
def longestPalindrome(self, s: str) -> str:
    # the idea is to take each character in the string and expand around the centers
    # we would need to consider cases for odd even length
    result = ""
    maximum_length = 0
    for i in range(len(s)):
        # odd palindrome
        left, right = i, i
        while left >= 0 and right < len(s) and s[left] == s[right]:
            length = right - left + 1
            if length > maximum_length:
                result = s[left:right + 1]
                maximum_length = length
            left -= 1
            right += 1

        # even palindrome
        left, right = i, i + 1
        while left >= 0 and right < len(s) and s[left] == s[right]:
            length = right - left + 1
            if length > maximum_length:
                result = s[left:right + 1]
                maximum_length = length
            left -= 1
            right += 1
    return result


# 10 String Compression
def compress(self, chars: List[str]) -> int:
    # For a weird reason the expected output is different from what was stated in the question description
    counter = {}  # get the letters and their frequency
    for i in chars:
        counter[i] = 1 + counter.get(i, 0)
    char_count = len(counter)  # number of characters
    value_count = 0
    for i in counter.keys():  # get the length of the each frequency digits(if it's not 1)
        if i != 1:
            value_count += len(str(i))
    return value_count + char_count


# 11 Fraction to Recurring Decimal
def fractionToDecimal(self, numerator: int, denominator: int) -> str:
    # this is a very tricky problem
    # spent up to 15min trying to understand someones approach
    # i'll try to break it down and explain each logic
    if numerator == 0: return "0"  # edge case

    # first take care of negative integers whether the numerator or denominator or both
    prefix = ""  # should be negative or empty(positive)
    if numerator < 0 and denominator > 0 or numerator > 0 and denominator < 0:
        prefix = "-"

    # make the numerators postive(this also covers the case where they are both negative
    numerator, denominator = abs(numerator), abs(denominator)

    # sort of simulate the division logic
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
        # mark the position as visited by changing the value (if the interveiwer allows that)
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


# 16 Largest Numbers
def largestNumber(self, nums: List[int]) -> str:
    # this is actually a very unintuitive approach
    # for each pairwise comparison during the sort, we compare the
    # numbers achieved by concatenating the pair in both orders
    nums = list(map(str, nums))  # convert all the integers to strings

    def compare(n1, n2):  # define a custom sorting function
        if n1 + n2 > n2 + n1:
            return -1
        else:
            return 1

    nums.sort(key=cmp_to_key(compare))  # clarify with your interviewer if you're allowed to use an imported module

    # take care of a minor edge case where all the numbers are 0
    # it should return just zero if that's the case
    return "".join(nums) if nums[0] != "0" else "0"


# 17 Meeting Rooms
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
    # basically for every meeting, we need to know if it starts a time less than the end time of another
    # meeting, if this happens we know the two meetings would overlap so we allocate a seperate room for the
    # current meeting
    # since it's obvious we want to make the best use of the number of rooms we allocate, we have to be
    # greedy with this approach
    # let's seperate the meetings into start time and end time and sort them seperately
    starting_times = sorted([time[0] for time in intervals])
    ending_times = sorted([time[0] for time in intervals])

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


# 19 Maximum SubArray
def maxSubArray(self, nums: List[int]) -> int:
    currSum = 0
    maxSum = 0
    for num in nums:
        if currSum < 0:  # if the sum of the previous subarray is negative, we reset it to zero
            currSum = 0  # cos we are really after getting the maximum subarray
        currSum += num
        maxSum = max(maxSum, currSum)
    return maxSum


# 20 Find Minimum in a rotated Sorted Array
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


# 21 Minimum Size SubArray
def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    # sliding window approach
    window, start, minLength = 0, 0, math.inf
    for end in range(len(nums)):
        window += nums[end]
        while window >= target:
            minLength = min(end - start + 1, minLength)
            window -= nums[start]
            start += 1
    return minLength if minLength != math.inf else 0


# 22 Container with most Water
def maxArea(self, height: List[int]) -> int:
    # the brute force would be to have every possible combination of the heights - O(n^2)
    # we can optimize it to linear time using two pointers approach,
    # we set our left and right pointers to the first and last height respectively and we only shift the
    # pointers that have a lower height
    left, rigth, maximumArea = 0, len(height) - 1, 0
    while left <= rigth:
        area = (rigth - left) * min(height[left], height[rigth])
        maximumArea = max(maximumArea, area)
        if height[left] <= height[rigth]:
            left += 1
        else:
            rigth -= 1
    return maximumArea


# 23 Word Search  *
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


# 24 Rotting Oranges
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


# 25 Delete and Earn
def deleteAndEarn(self, nums: List[int]) -> int:
    # This is another dynamic programming problem
    # I'll explain the logic to solving the problem using Bottom Up Approach
    maxNumber = 0
    points = {i: 0 for i in range(max(nums) + 1)}  # or use collection.defaultdict(int)
    # First we map each number to get the points we can get from it and get the maximum of the numbers at the same time
    for num in nums:
        points[num] = num + points.get(num, 0)
        maxNumber = max(num, maxNumber)
    maxEarnings = [0] * (maxNumber + 1)
    maxEarnings[1] = points[1]
    # The maximum you can earn at any points will the max of you taking the current price n (which means you can't take
    # the previous one (n-1) or you ignore the current the current price which means you've taken (n-1)
    for i in range(2, maxNumber):
        maxEarnings[i] = max(maxEarnings[i - 1], maxEarnings[i - 2] + nums[i])
    return maxEarnings[maxNumber]


# 26 Coin Change
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
