from typing import List


# Merge Intervals
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key=lambda x: x[0])
    merged_interval = []
    for interval in intervals:
        # if the start time of the current one is greater then the end time of
        # the previous one then they don't overlap
        if not merged_interval or interval[0] > merged_interval[-1][1]:
            merged_interval.append(interval)
        else:
            merged_interval[-1][1] = max(interval[1], merged_interval[-1][1])
    return merged_interval


# Insert Intervals
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    result = []
    for i in range(len(intervals)):
        # appened to the result and return if the end time of the new interval is less than the current interval
        # start time
        if newInterval[1] < intervals[i][0]:
            result.append(newInterval)
            return result + intervals[i:]
        # only append the current interval to the result if the new interval start time is less than
        # the current interval end time, this is so we can check against other intervals in the list
        elif newInterval[0] < intervals[i][1]:
            result.append(intervals[i])
        # now they overlap, so we update the new interval
        else:
            newInterval = [min(newInterval[0], intervals[i][0]), max(newInterval[1], intervals[i][1])]
    # if the code get's here we have to append the new interval to the list since it was only being update
    result.append(newInterval)
    return result


# Meeting Rooms
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end


def can_attend_meetings(self, intervals: List[Interval]) -> bool:
    # sort by starting times
    intervals.sort(key=lambda x: x.start)
    for i in range(len(intervals) - 1):
        # simple check for overlapping in start and end times
        if intervals[i].end > intervals[i + 1].start:
            return False
    return True


# Meeting Rooms 2
def min_meeting_rooms(self, intervals: List[Interval]) -> int:
    # group starting and ending times in sorted order
    start_time = sorted([interval.start for interval in intervals])
    end_time = sorted([interval.end for interval in intervals])
    rooms = max_room = 0
    start = end = 0
    # check for the numbers of rooms at each point and return max
    while start < len(start_time):
        if start_time[start] < end_time[end]:
            rooms += 1  # a meeting is going on
            start += 1
            max_room = max(max_room, rooms)
        else:
            rooms -= 1
            end += 1
            max_room = max(max_room, rooms)
    return max_room


# Non Overlapping intervals
def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    res = 0
    intervals.sort(key=lambda x: x[0])
    prevEnd = intervals[0][1]
    for start, end in intervals[1:]:
        if start >= prevEnd:
            prevEnd = end
        else:
            res += 1
            prevEnd = min(prevEnd, end)
    return res
