import heapq
from typing import List

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
