from typing import List

# 57 Insert Intervals -- Intervals
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    result = []
    for i in range(len(intervals)):
        # check the new interval end time against the current interval start time
        # if the end time is less than the start time, then they do not overlap
        # and this means it definitely won't overlap with the rest of the intervals
        if newInterval[1] < intervals[i][0]:
            result.append(newInterval)
            return result + intervals[i:]

        # if the start time of the new interval is more than the endtime of the current interval, then they also do
        # not overlap but there is a possibility of it overlapping with one of the remaining intervals

        elif newInterval[0] > intervals[i][1]:
            result.append(intervals[i])

        else:  # If the code get's here that means they overlap
            newInterval = [min(intervals[i][0], newInterval[0]), max(intervals[i][1], newInterval[1])]
    result.append(newInterval)  # an edge case where the new interval overlaps with the last interval in the list
    return result

# 49 Merge Intervals -- Intervals
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    """Two intervals overlap if the end time of the previous intervals is more than the start time of the current
    interval. Sorting the intervals by their start time make the intervals that can possibly be merged appear in a
    contigous run """
    intervals.sort(key=lambda x: x[0])
    merged_intervals = []
    for interval in intervals:
        # clarify with your interviewer if two intervals where the start time of one equals the end time can be consider
        # ed as overlapping
        if not merged_intervals or merged_intervals[-1][1] < interval[0]:
            merged_intervals.append(interval)
        else:
            # if they overlap, we pick the maximum end time
            merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
    return merged_intervals
