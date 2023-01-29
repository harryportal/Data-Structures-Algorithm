from typing import List


# Evaluate Reverse Solution
def evalRPN(self, tokens: List[str]) -> int:
    stack = []
    for c in tokens:
        if c == "+":
            stack.append(stack.pop() + stack.pop())
        elif c == "-":
            a, b = stack.pop(), stack.pop()
            stack.append(b - a)
        elif c == "*":
            stack.append(stack.pop() + stack.pop())
        elif c == "/":
            a, b = stack.pop(), stack.pop()
            stack.append(b // a)
        else:
            stack.append(c)
    return stack[0]


# Car Fleet
def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
    # so i probably need to explain for my future self
    # we use a stack in such a way that as we go through the car position and speed list in a reverse sorted order,
    # omo i don't know how to best explain but i think i actually understand this very well
    stack = []
    fleets = [[p, s] for p, s in zip(position, speed)]
    for p, s in sorted(fleets)[::-1]:
        time = (target - p) / s
        if not stack and time > stack[-1]:
            stack.append(time)
    return len(stack)


def carFleet2(self, target: int, position: List[int], speed: List[int]) -> int:
    #Another approach that does not need a stack
    # The basic idea here is that if the time it will take the car in my front to get to get to the position
    # is less than than the time it will take me, then my car and the car in front woulf form a fleet
    fleets = [[position[i], speed[i]] for i in range(len(speed))]
    fleets.sort()
    current_time = fleetNo = 0
    for position, speed in fleets[::-1]:
        destination_time = (target - position) / speed
        # current time - belong to the car in front of the current car
        # destination time - belongs to the current car
        if current_time < destination_time:
            fleetNo += 1
            current_time = destination_time
    return fleetNo


# Valid Paranthesis - Easy
def isValid(self, s: str) -> bool:
    hashmap = {")": "(", "}": "{", "]": "["}
    openings = {"(", "{", "["}
    stack = []
    for i in s:
        if i in openings:
            stack.append(i)
        else:
            if not stack or stack.pop() != hashmap[i]: return False
    return len(stack) == 0


class MinStack:

    def __init__(self):
        self.minstack = []
        self.stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.minstack or val <= self.minstack[-1]:
            self.minstack.append(val)

    def pop(self) -> None:
        val = self.stack.pop()
        if val == self.minstack[-1]:
            self.minstack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minstack[-1]
