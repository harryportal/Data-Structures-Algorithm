# 45 Implement Queues using stacks
import collections
from typing import List


class MyQueue:

    def __init__(self):
        self.stackA = []  # only supports popping from the end(LIFO)
        self.stackB = []

    def push(self, x: int) -> None:
        self.stackA.append(x)

    def pop(self) -> int:
        for i in range(len(self.stackA) - 1):  # push the values after the first added to a tempoary stack
            self.stackB.append(self.stackA.pop())
        result = self.stackA.pop()  # pop the first added value
        while self.stackB:  # push the values back into the main stack
            self.push(self.stackB.pop())
        return result

    def peek(self) -> int:
        return self.stackA[0]

    def empty(self) -> bool:
        return len(self.stackA) == 0




# 28 Next Greater Element
def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    # Neetcode explains here well https://www.youtube.com/watch?v=68a1Dc_qVq4
    numsIndex = {num: index for index, num in enumerate(nums1)}
    result = [-1] * len(nums1)
    stack = []  # a monotonically decreasing stack
    for num in nums2:
        while stack and stack[-1] < num:
            value = stack.pop()
            index = numsIndex[value]
            result[index] = num
        if num in numsIndex:
            stack.append(num)
    return result



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


