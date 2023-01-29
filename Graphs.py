import collections
from typing import List
from collections import deque


# Graph Basics
# Basically contains nodes connected by edges
# Union Find Data Structure
# It is used to check if the nodes in a graph are connected
# It has different implementation but the most efficient is
# called union find by ranking
class UnionFind:
    def __init__(self, n):
        self.par = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        while i != self.par[i]:
            self.par[i] = self.par[self.par[i]]
            i = self.par[i]
        return i

    def union(self, a, b):
        aRoot = self.find(a)
        bRoot = self.find(b)

        if aRoot == bRoot:
            return False
        if self.rank[aRoot] < self.rank[bRoot]:
            self.par[aRoot] = bRoot
            self.rank[bRoot] += self.rank[aRoot]
        else:
            self.par[bRoot] = aRoot
            self.rank[aRoot] += self.rank[bRoot]
        return True


def numberOfGoodPaths(self, vals: List[int], edges: List[List[int]]) -> int:
    # create an adjacency list from the edges
    adjdict = {node: [] for node, _ in edges}
    for node, neighbour in edges:
        adjdict[node].append(neighbour)
        adjdict[neighbour].append(node)

    # create an hashmap to map the exact values given in vals array to indexes that have that value
    valIndex = collections.defaultdict(list)
    for index, val in enumerate(vals):
        valIndex[val].append(index)

    result = 0
    unionfind = UnionFind(len(vals))
    # we go through each value in the valIndex hashmap in sorted order
    # then go through the each of the index that have the val
    # then go through the neighbour of that index using the adjDict hashmap
    # use the union find class to unite the two nodes at this point if the current neighbout if the value is less
    # than or equal to the current node's value
    for value in sorted(valIndex.keys()):
        for index in valIndex[value]:
            for neighbour in adjdict[index]:
                if vals[neighbour] <= vals[index]:
                    unionfind.union(index, neighbour)

        count = collections.defaultdict(int)
        # sum of the number of disjoint set the value is present in
        for i in valIndex[val]:
            count[unionfind.find(i)] += 1
            result += count[unionfind.find(i)]
    return result


# Breadth First Search
# Take a node as your root, start with it and traverse every of it neighbours
# do the same for your neighbours and the rest is story
def breadthfirstSearch(root, graph: dict):
    queue, visited = deque([root]), set()
    # we need to keep track of the nodes visited since this is not a tree
    while queue:
        node = queue.popleft()
        visited.add(node)
        for i in graph[node]:
            if i not in visited:
                queue.append(i)
    return visited


graph = dict()
graph['A'] = ['B', 'S']
graph['B'] = ['A']
graph['S'] = ['A', 'G', 'C']
graph['D'] = ['C']
graph['G'] = ['S', 'F', 'H']
graph['H'] = ['G', 'E']
graph['E'] = ['C', 'H']
graph['F'] = ['C', 'G']
graph['C'] = ['D', 'S', 'E', 'F']
print(breadthfirstSearch('E', graph))


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


# Number of Islands
def numIslands(self, grid: List[List[str]]) -> int:
    """We do a depth first search whenever we get to a value of 1
    Time complexity - O(n x m) in worst case if the the grid is entirely
    filled with 1.
    Space complexity - O(n X m) because the dfs has to go to all values in the grid"""
    rows, cols = len(grid), len(grid[0])
    islands = 0

    def dfs(r, c):
        # we do a dfs recursively
        # define base case
        if not (r in range(rows) and c in range(cols)) or grid[r][c] != "1":
            return
        grid[r][c] = "2"  # changing the number sort of marks it as visited
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                dfs(r, c)
                islands += 1

    return islands


# Max Area of Island
def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    maxArea = 0

    def dfs(r, c):
        # base case
        if not (r in range(rows) and c in range(cols)) or grid[r][c] != 1:
            return 0
        grid[r][c] = 2
        return 1 + dfs(r + 1, c) + dfs(r - 1, c) + dfs(r, c + 1) + dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                maxArea = max(maxArea, dfs(r, c))
    return maxArea


# Check if path exist in a graph
def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
    # let's use a depth first search algorithm here with a stack
    visited = set()  # to keep track of visited nodes
    stack = [source]

    # build graph from edges
    graph = {node: [] for node, _ in edges}
    for node, neighbour in edges:
        graph[node].append(neighbour)
        graph[neighbour].append(node)

    # dfs algorithm
    while stack:
        node = stack.pop()
        visited.add(node)
        if node == destination:
            return True
        for neighbour in graph[node]:
            if neighbour not in visited:
                stack.append(node)
    return False


def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
    # let's solve the same problem but now with recursive dfs
    visited = set()
    graph = collections.defaultdict(list)
    for node, neighbour in edges:
        graph[node].append(neighbour)
        graph[neighbour].append(node)

    def dfs(node):
        visited.add(node)
        if node == destination:
            return True
        for neighbour in graph[node]:
            if neighbour not in visited:
                dfs(neighbour)
        return False

    return dfs(source)


def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
    # This solution was approached with backtracking and dynamic programming in leetcode
    # official solution but i think it's an overkill
    # we can simply solve this with breadth first search
    # it is also an acyclic graph so we don't need to worry about keeping track of visited nodes
    queue, result = [[0]], []
    while queue:
        path = queue.pop(0)
        if path[-1] == len(graph) - 1:  # if we have a path that ends with n-1
            result.append(path)
            continue
        for node in graph[path[-1]]:
            queue.append(path + [node])

    return result


def cloneGraph(self, node):
    # we use a hashmap to map the old to new nodes
    # we do the whole cloning thing with depth first search
    if not node:  # edge case
        return None
    hashmap = {}

    def clone(node):
        if node in hashmap:
            return hashmap[node]  # returns the clone if the node has already been cloned
        copy = Node(node.val)
        hashmap[node] = copy
        for neighbour in node.neighbours:
            copy.neighbors.append(clone(neighbour))
        return copy

    return clone(node)


# Reconstruct Itenary - omo this one choke oh!
def findItinerary(self, tickets: List[List[str]]) -> List[str]:
    """so basically there are few things to consider before jumping into the code this time
    As usual, we have to construct the graph using a hashmap to map each node to their neighbours
    but before that, we want to first sort the tickets since we are asked to return the one with
    a higher lexicological order incase of multiple paths
    However after sorting, we might find ourselves picking a neighbour that of course has high lexicol..wo
    order so we need to make sure that the node we're picking has at least one neighbour
    The rest is depth first search logic
    """
    # Build the adjacency list using a hashmap
    graph = {src: [] for src, dst in tickets}
    for src, dst in tickets:
        graph[src].append(dst)

    result = ["JFK"]  # we first append the starting point

    def dfs(node):
        if len(result) == len(tickets) + 1:
            return True  # we have found a valid path
        if node not in graph:  # if the node has no neighbour
            return False

        temp = list(graph[node])  # creating a copy cos we don't want to modify the array during iteration
        for index, curr in enumerate(temp):
            result.append(curr)
            graph[node].pop(index)
            if dfs(curr): return True
            result.pop()
            graph[node].insert(index, curr)

    dfs("JFK")
    return result


def snakesAndLadders(self, board: List[List[int]]) -> int:
    # omo which kind question be this
    # we first need an helper function to get the row and column just in case
    # the index we move has a ladder or snake
    # we also need to reverse the board for it to make sense i think
    length = len(board)
    board.reverse()

    def intToPos(value):
        row = (value - 1) // length
        col = (value - 1) % length
        if row % 2:
            # added this due to alternating rows
            col = length - col - 1
        return [row, col]

    queue = [[1, 0]]  # a pair of the value, move it took to get to that value
    visited = set()  # to make sure we don't visit a value twice
    while queue:
        value, moves = queue.pop(0)
        for i in range(1, 7):  # throw the dice lol
            nextPosition = value + i
            row, col = intToPos(nextPosition)
            if board[row][col] != -1:  # there's a snake or ladder here then
                nextPosition = board[row][col]
            if nextPosition == length ** 2:
                return moves + 1
            if nextPosition not in visited:
                visited.add(nextPosition)
                queue.append([nextPosition, moves + 1])
    return -1
