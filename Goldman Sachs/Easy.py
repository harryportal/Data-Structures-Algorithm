import collections
import math
from typing import List, Optional
from heapq import heappush
import heapq
from Linkedlist import ListNode

# 1 Check if the Sentence Is Pangram
from Trees import TreeNode


# 1 Check Pangram
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
    hashmap = {}  # map the numbers in the list to their index
    for index, value in enumerate(nums):
        remainder = target - value
        if remainder in hashmap:
            return [index, hashmap[remainder]]
        hashmap[value] = index
    return []


# 10 Count Binary Strings   *
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


# 13 Pascal Triangle II  *
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

    # dynamic programming approach -- bottom up iteration
    fib = [0] * (n + 1)
    fib[1] = 1
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib[n]

    # dynamic programming approach  II - Top down memoization
    # cache = {}
    # if n <= 1: return n
    # if n not in cache:
    #   cache[n] = fib(n-1) + fib(n-2)
    # return cache[n]


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
    def count(num: int):
        count = 0
        while num:
            if num & 1: count += 1
            num >>= 1
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
        while n:  # this will continue until n is zero(runtime is O(32) if in worst case all the bits are 1
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
    if n <= 2: return n
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
    # Our minimum start value should be able to make the minimum of all step by step sum equal to exactly 1
    total, minStep = 0, 0
    for num in nums:
        total += num
        minStep = min(total, minStep)
    return 1 - minStep


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


# 22 Reverse String
def reverseString(self, s: List[str]) -> None:
    # we simply make use of two pointers to swap the letters in place
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1


# 23 Height Checker
def heightChecker(self, heights: List[int]) -> int:
    expected = sorted(heights)
    indices = 0
    for index, height in enumerate(heights):
        if expected[index] != height:
            indices += 1
    return indices


# 24 Middle of Linked list
def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
    # One approach is to store all the value in the linked list and return the value at the middle
    # We can reduce the space complexity to 0(1) by using a slow and fast pointer
    # the fast pointer will traverse the linked list twice as fast as the slow pointer so when we fast pointer is at
    # the end, the slow will definitely point to the middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


# 25 Invert Tree
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    # start with root, invert the child nodes, do the same recursively for the child nodes
    # O(n) runtime going through all the nodes and 0(n) for the recursive function call stack
    if not root: return None  # edge case
    root.left, root.right = root.right, root.left
    self.invertTree(root.left)
    self.invertTree(root.right)
    return root


# 26 Binary Tree Inorder Traversal
def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    # inorder traversal --> left -- root -- right
    # let's solve this with an iterative dfs using stack
    stack, result, curr = [], [], root
    if not curr: return []  # edge case
    while curr or stack:
        while curr:  # keep going along the left branch of the current node
            stack.append(curr.val)
            curr = curr.left
        node = stack.pop()
        result.append(node.val)
        if node.right: curr = node.right
    return result


# 27 Reverse Linked list
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    # reversing the linked list means we now want the tail to be the head, first off remember the tail points to None
    # if the head is going to be the tail now, it has to point to None and then we keep reversing the pointers
    if not head: return None
    prev, curr = None, head
    while curr:
        temp = curr.next  # temporarily store the next node in the linked list
        curr.next = prev  # make the current node point to the prev node(in this case starts from None)
        prev = curr
        curr = temp
    return prev


# 28 Next Greater Element
def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    # Neetcode explains here well https://www.youtube.com/watch?v=68a1Dc_qVq4
    numsIndex = {num: index for index, num in enumerate(nums1)}
    result = [-1] * len(nums1)
    stack = []
    for num in nums2:
        while stack and stack[-1] < num:
            value = stack.pop()
            index = numsIndex[value]
            result[index] = num
        if num in numsIndex:
            stack.append(num)
    return result


# 29 Binary Search
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


# 30 Delete Duplicates from sorted Lists
def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
    pointer = head
    while pointer and pointer.next:
        if pointer.val == pointer.next.val:
            pointer.next = pointer.next.next
        else:
            pointer = pointer.next
    return head


# 31 Move Zeroes
def moveZeroes(self, nums: List[int]) -> None:
    # One approach is to create two arrays, store the zeroes in ones and the others in the second one and join the two
    # lists. We can optimize the space complexity to 0(1) by using the knowledge of quick sort algorithm to partition
    # also seeing the question as "moving the non zeroes to the front" rather than "moving the zeros to the back" makes
    # the solution more intuitive
    left = 0
    for right in range(len(nums)):
        if nums[right] > 0:
            nums[right], nums[left] = nums[left], nums[right]
            left += 1


# 32 Implement Stack using Queues
class MyStack:

    def __init__(self):
        self.queue = collections.deque()

    def push(self, x: int) -> None:
        self.queue.append(x)

    def pop(self) -> int:
        for i in range(len(self.queue) - 1):
            self.push(self.queue.popleft())
        return self.queue.popleft()

    def top(self) -> int:
        return self.queue[-1]

    def empty(self) -> bool:
        return len(self.queue) == 0


# 33 Symmetric Tree
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    if not root: return True  # edge case

    # we basically define an helper function to check the nodes recursively
    def check(nodeA, nodeB):
        if not nodeA and not nodeB: return True
        if not nodeA or not nodeB or nodeA.val != nodeB.val:
            return False
        return check(nodeA.left, nodeB.right) and check(nodeA.right, nodeB.left)

    return check(root.left, root.right)


# 34 Last Stone Weight
def lastStoneWeight(self, stones: List[int]) -> int:
    # Using a max heap will come in handy here as we need a way to efficiently get the two maximum stones as we
    # iterate through the list in one pass
    stones = [-stone for stone in stones]  # making all the values negative since python only supports min heaps
    heapq.heapify(stones)  # converts to a max heap
    for stone in stones:
        weightY = heapq.heappop(stones)
        weightX = heapq.heappop(stones)
        if abs(weightY) > abs(weightX):  # it's either they are equal or Y is greater
            heapq.heappush(stones, weightY - weightX)
    return 0 if not stones else abs(stones[-1])


# 35 Determine if Two Events Have Conflict
def haveConflict(self, event1: List[str], event2: List[str]) -> bool:
    # Two events are said to have conflicts if the start time or end time overalaps
    def convert(timeStr):  # an helper function to convert the string to a 24 hour time
        time = (int(timeStr[:2]) * 100) + int(timeStr[3:])
        return time

    startA, endA = convert(event1[0]), convert(event1[1])
    startB, endB = convert(event2[0]), convert(event2[1])
    return startA <= endB and startB <= endA


# 36 . Keep Multiplying Found Values by Two
def findFinalValue(self, nums: List[int], original: int) -> int:
    hashset = set(nums)  # clarify with your interviewer if the values will be unique, if they won't we use an hashmap
    while original in hashset:
        original *= 2
    return original
