# 2 High Five
import heapq
from typing import List


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
        heapq.heappush(hashmap[id], -score)  # using negative here since python only offers min heap

    result = []
    # compute the average for each id
    for id, scores in hashmap.items():
        sum_five = 0
        for _ in range(5):
            sum_five += abs(heapq.heappop(scores))
        result.append([id, sum_five // 5])

    return sorted(result)

# 34 Last Stone Weight
def lastStoneWeight(self, stones: List[int]) -> int:
    # Using a max heap will come in handy here as we need a way to efficiently get the two maximum stones as we
    # iterate through the list in one pass
    stones = [-stone for stone in stones]  # making all the values negative since python only supports min heaps
    heapq.heapify(stones)  # converts to a max heap
    for _ in stones:
        weightY = heapq.heappop(stones)
        weightX = heapq.heappop(stones)
        if abs(weightY) > abs(weightX):  # it's either they are equal or Y is greater
            heapq.heappush(stones, weightY - weightX)
    return 0 if not stones else abs(stones[-1])