# 54 Decode String -- Stack
from typing import List


def decodeString(self, s: str) -> str:
    """ As we go through the string in one pass, we keep track of the current number until we get to an opening square
bracket(which also mean s that the previous string has been processed), so we append the currString * it'c count to the
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





# 48 Min Stack -- Stack
class MinStack:
    """Create a main stack and another monotically decreasing stack for keeping track of
    minimum values as they are added to the main stack
    """

    def __init__(self):
        self.stack = []
        self.minStack = []  # A monotically decreasing stack

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


# 32 Remove Adjacent Duplicate from a string  II -- stack
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
    return


# 31 Asteroid Collision  -- stack
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
