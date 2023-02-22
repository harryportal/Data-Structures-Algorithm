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


# 373. Find K Pairs with Smallest Sums
def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    """ Whenever you see a "top k" Question, the first thing you should mention to your interviewer is sorting algorithm
    and then talk about how using heaps can optimise the previous approach especially for large inputs.
    For this question the idea centers around maintaining a min heap of size K"""
    # initialise a min heap of size k using nums1 against the first value in nums2
    heap = [(nums1[i]+nums2[0], i, 0) for i in range(min(len(nums1), k))]
    heapq.heapify(heap)
    checked = {(i,0) for i in range(min(len(nums1), k))} # hashset to avoid reusing the same pair of index

    result = []
    while heap and k > 0:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        k -= 1

        if i + 1 < len(nums1) and (i+1, j) not in checked:
            heapq.heappush((nums1[i+1] + nums2[j], i+1, j), heap)
            checked.add((i+1, j))

        if j + 1 < len(nums2) and (i, j + 1) not in checked:
            heapq.heappush((nums1[i] + nums2[j + 1], i, j+1), heap)
            checked.add((i, j+1))

    return result
