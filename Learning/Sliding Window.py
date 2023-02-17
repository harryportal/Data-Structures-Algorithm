import collections
from typing import List


# Best time to buy and sell Stocks
def maxProfit(self, prices: List[int]) -> int:
    buy_price = prices[0]  # start with the first price
    maxProfit = 0
    for i in range(1, len(prices)):
        sell_price = prices[i]
        if sell_price < buy_price:  # simple finicial decision
            buy_price = sell_price
        else:
            maxProfit = max(maxProfit, sell_price - buy_price)
    return maxProfit


# Longest Substring Without Repeating Characters -- Medium
def lengthOfLongestSubstring(self, s: str) -> int:
    hashset = set()
    index, max_length = 0, 0
    for char in s:
        while char in hashset:
            hashset.remove(s[index])
            index += 1
        hashset.add(char)
        max_length = max(max_length, len(hashset))
    return max_length


# Longest Repeating Character Replacement -- omo this one choke
def characterReplacement(self, s: str, k: int) -> int:
    # the question can be seen mathematically as asking us to find the
    # maximum subarray that contains all but k letters the same
    # we maintain an hashmap and a sliding window as we iterate through the string
    # we keep track of the char that has the maximum occurrence and update our max_lenght
    # to be the length of the frequent + k(number of characters we are allowed to change)
    # whenever our window size exceeds that the maximum frequency + k, we shift our window by
    # decrementing the occurrence of the element at the window_start index
    max_length, hashmap = 0, {}
    max_freq, window_start = 0, 0
    for window_end, char in enumerate(s):
        hashmap[char] = 1 + hashmap.get(char, 0)
        max_freq = max(hashmap[char], max_freq)
        while window_end - window_start + 1 > max_freq + k:  # the window is invalid at this point
            # we shift our window
            char_start = s[window_start]
            hashmap[char_start] -= 1
            if hashmap[char_start] < 0: del hashmap[char_start]
            window_start += 1
            max_freq = max(max_freq, hashmap[char])  # doing this cos the current character might be the
            # same one we're reducing it frequency
        max_length = max(window_end - window_start + 1, max_length)
    return max_length


# Fruits into Basket -- Medium
def totalFruit(self, fruits: List[int]) -> int:
    # this problem can be viewed mathematically as finding the largest subarray containing only two unique
    # integers
    hashmap = {}  # two basket for the two unique fruit
    start, max_picked = 0, 0
    for end, fruit in enumerate(fruits):
        hashmap[fruit] = 1 + hashmap.get(fruit, 0)

        # slide the window and adjust the hashmap if the length is more than 2
        while len(hashmap) > 2:
            hashmap[fruits[start]] -= 1
            if hashmap[fruits[start]] == 0: del hashmap[fruits[start]]
            start += 1
        max_picked = max(max_picked, end - start + 1)
    return max_picked


# Maximum Average Subarray I - Easy
def findMaxAverage(self, nums: List[int], k: int) -> float:
    # we use sliding window approach here to
    # we maintain a window of fixed size k and compute the sum along side
    # whenever our window is of size k, we compute the average, compare it with the existing average and shift our
    # window by removing the first item
    max_average = float("-inf")
    start, window = 0, 0
    for end in range(len(nums)):
        window += nums[end]  # our window should always
        if end - start + 1 >= k:
            max_average = max(max_average, window / k)
            window -= nums[start]
            start += 1
    return max_average


def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
    # we maintain of max size k using an hashset
    # if we get to a value that already occurs in our hashset, we simply run it
    # we remove the first element in our hashset whenever it size exceeds k
    hashset = set()
    for index in range(len(nums)):
        if nums[index] in hashset:
            return True
        hashset.add(nums[index])
        if len(hashset) > k:
            hashset.remove(nums[index - k])
    return False


def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    queue = collections.deque()  # a monotonically decreasing queue to score the index of the
    # maximum number in each window
    start, max_window = 0, []
    for end, number in enumerate(nums):
        while queue and nums[queue[-1]] < number:
            queue.pop()
        queue.append(end)
        # pop from the queue the left most value if the index is less than 'start'
        if queue[0] < start:
            queue.popleft()

        # append the top of the queue once we reach our window size
        if end - start + 1 >= k:
            max_window.append(nums[queue[0]])
            start += 1
    return max_window
