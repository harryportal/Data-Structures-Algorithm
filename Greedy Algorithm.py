from typing import List


# Medium
def findMinArrowShots(self, points: List[List[int]]) -> int:
    # we of course start by sorting the list by the end value of each points
    points.sort(key=lambda x: x[1])
    balloon_end, arrows = points[0][-1], 1
    for start, end in points:
        """increase the number of arrows if the current ballon's start value is more that
        the previous ballon end value"""
        if start > balloon_end:
            arrows += 1
            balloon_end = end
    return arrows


def maxIceCream(self, costs: List[int], coins: int) -> int:
    max_bars = 0
    costs.sort()
    for i in range(len(costs)):
        current_cost = costs[i]
        # if the cost of the current bar is less than the coins you have left
        # this means you can buy any longer, just return the number of bars you've bought
        if current_cost > coins:
            return max_bars
        else:
            coins -= current_cost
            max_bars += 1
    return max_bars


def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
    total_tank, curr_tank = 0, 0
    starting_index = 0
    for i in range(len(gas)):
        remaining = gas[i] - cost[i]
        total_tank += remaining
        curr_tank += remaining
        if curr_tank < 0:
            starting_index = i + 1
            curr_tank = 0
    return starting_index if total_tank >= 0 else -1
