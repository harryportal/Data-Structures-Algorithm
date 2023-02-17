from typing import Optional, List


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Reverse Linked List
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    prev = None
    current = head
    while current:
        temp = current.next
        current.next = prev
        prev = current
        current = temp
    return prev


# Merge two sorted linked list
def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    dummynode = dummy = ListNode()
    val1, val2 = list1, list2
    if not val1 and not val2:  # edge case
        return None
    while val1 and val2:
        if val1.val > val2.val:
            dummy.next = val2
            val2 = val2.next
        else:
            dummy.next = val1
            val1 = val1.next
        dummy = dummy.next
    if val1:
        dummy.next = val1
    if val2:
        dummy.next = val2
    return dummynode.next


# Reorder List
def reorderList(self, head: Optional[ListNode]) -> None:
    # get the middle of the linked list
    # reverse the second half and merge the two halfs

    # let's get the middle with fast and slow pointers
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    second = slow.next  # the second half of the list
    slow.next = None  # break the second half from the original list
    # reverse the second part

    prev = None
    while second:
        temp = second.next
        second.next = prev
        prev = second
        second = temp

    first, second = head, prev
    while second:
        temp1, temp2 = first.next, second.next
        first.next = second
        second.next = temp1
        first, second = temp1, temp2


# Remove Nth node from linked list
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    dummynode = ListNode(0, head)
    left = dummynode
    right = head

    while n > 0 and right:
        right = right.next
        n -= 1

    while right:
        left = left.next
        right = right.next

    left.next = left.next.next
    return dummynode.next


class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


# Copy list with random pointer
def copyRandomList(self, head: Optional[Node]) -> Optional[Node]:
    # the intuitive thing is to use a hashmap to map old nodes to new nodes and random
    # now let's implement this with a constant space
    if not head: return None
    hashmap = {}
    current = head
    while current:
        hashmap[current] = Node(current.val)
        current = current.next
    current = head
    while current:
        node = hashmap[current]
        node.random = hashmap.get(current.random, None)
        node.next = hashmap.get(current.next, None)
        current = current.next
    return hashmap.get(head, None)


# Linked list Cycle
def hasCycle(self, head: Optional[ListNode]) -> bool:
    if not head:
        return False
    # we use fast and slow pointers
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if fast == slow:
            return True
    return False


# Find the duplicate Number in a list -- constant space lol
def findDuplicate(self, nums: List[int]) -> int:
    # detect the cycle
    slow, fast = 0, 0
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    # Find the duplicate
    slow2 = 0
    while True:
        slow = nums[slow]
        slow2 = nums[slow2]
        if slow == slow2:
            return slow

#LRu Cache -- we use a double linked list and a hashmap
class Double:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = self.prev = None


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.right = self.left = Double(0,0)
        self.right.prev, self.left.next = self.left, self.right

    # helper method for the Double linked list
    def insert(self, node):
        prev, current = self.right.prev, self.right
        prev.next = current.prev = node
        node.next, node.prev = current, prev

    def remove(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self.remove(node)
            self.insert(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        node = Double(key, value)
        self.insert(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            node = self.left.next
            self.remove(node)
            del self.cache[node]


def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
    hashmap = {}
    # get the occurrences of each node
    pointer = head
    dummy = pointer
    while dummy:
        current_value = dummy.val
        hashmap[current_value] = 1 + hashmap.get(current_value, 0)
        dummy = dummy.next

    newHead = ListNode(0, head)
    pointer2 = newHead
    while pointer2.next:
        if hashmap[pointer2.next.val] > 1:
            pointer2.next = pointer2.next.next
        else:
            pointer2 = pointer2.next
    return newHead.next
