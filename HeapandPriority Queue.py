# Kth largest element in a stream
import heapq
from typing import List


class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.min_heap, self.k = nums, k
        heapq.heapify(self.min_heap)
        while len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.min_heap, val)
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        return self.min_heap[0]


# Last stone weight
def lastStoneWeight(self, stones: List[int]) -> int:
    # build a max heap
    maxHeap = [-stone for stone in stones]
    heapq.heapify(maxHeap)
    while len(maxHeap) > 1:
        firstStone = heapq.heappop(maxHeap)
        secondStone = heapq.heappop(maxHeap)
        if secondStone > firstStone:
            heapq.heappush(maxHeap, firstStone - secondStone)

    return 0 if not maxHeap else abs(maxHeap[0])


# Kth largest element in an Array
def findKthLargest(self, nums: List[int], k: int) -> int:
    # build maxHeap
    maxheap = [-i for i in nums]
    heapq.heapify(maxheap)  # O(n)
    for _ in range(k):  # klog(n) -- average case
        val = heapq.heappop(maxheap)
    return -val


# Kth largest element in an Array -- Using Quick Select sort, space - O(1), runtime - O(n) - average case
# still not working omo, i'll check leetcode later
def findKthLargest_(self, nums: List[int], k: int) -> int:
    # using quick select algorithm with a sort of partition
    # hopefully i understand this when i see it some other time
    k = len(nums) - k  # since we want the kth value from the last

    def quickSelect(l, r):
        pivot, p = nums[r], l
        for i in range(l, r):
            if nums[i] <= pivot:
                # we swap the values
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        nums[p], nums[r] = nums[r], nums[p]
        if p > k:
            # we run the quick select algorithm on the left
            quickSelect(l, p - 1)
        elif p < k:
            # we run the quick select towards the right
            quickSelect(p + 1, r)
        else:
            return nums[p]

    return quickSelect(0, len(nums) - 1)


# Kth closest point to origin
def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    result = []
    minHeap = []
    for x, y in points:
        distance = x ** 2 + y ** 2
        minHeap.append([distance, x, y])
    heapq.heapify(minHeap)
    for _ in range(k):
        distance, x, y = heapq.heappop(minHeap)
        result.append([x, y])
    return result
