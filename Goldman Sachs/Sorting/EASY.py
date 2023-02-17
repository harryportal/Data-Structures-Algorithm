from typing import List

# 23 Height Checker
def heightChecker(self, heights: List[int]) -> int:
    expected = sorted(heights)
    indices = 0
    for index, height in enumerate(heights):
        if expected[index] != height:
            indices += 1
    return indices

# 35 Determine if Two Events Have Conflict
def haveConflict(self, event1: List[str], event2: List[str]) -> bool:
    # Two events are said to have conflicts if the start time or end time overalaps
    def convert(timeStr):  # an helper function to convert the string to a 24 hour time
        time = (int(timeStr[:2]) * 100) + int(timeStr[3:])
        return time

    startA, endA = convert(event1[0]), convert(event1[1])
    startB, endB = convert(event2[0]), convert(event2[1])
    return startA <= endB and startB <= endA

