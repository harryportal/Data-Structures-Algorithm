from typing import Optional, List
from Learning.Linkedlist import ListNode

# 52 Add Power II -- Linked list
def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    if not l1 and not l2:  # never forget to handle edge case
        return None

    # reverse the linked list and add
    def reverse(node):
        prev = None
        while node:
            temp = node.next
            node.next = prev
            prev = node
            node = temp
        return prev

    l1, l2 = reverse(l1), reverse(l2)
    carry, dummynode = 0, ListNode()
    pointer = dummynode
    while carry or l1 or l2:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        value_sum = val1 + val2 + carry
        carry = value_sum // 10
        pointer.next = ListNode(value_sum % 10)
        pointer = pointer.next
        l1 = l1.next if l1 else 0
        l2 = l2.next if l2 else 0
    return reverse(dummynode.next)


# 50 Add Two Numbers  -- Linked List
def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    dummyNode = ListNode()  # create a new node to hold the sum
    dummy, carry = dummyNode, 0
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        value_sum = val1 + val2 + carry
        carry = value_sum // 10
        dummy.next = ListNode(value_sum % 10)
        dummy = dummy.next
        l1 = l1.next if l1 else 0
        l2 = l2.next if l2 else 0
    return dummyNode.next


# 44 Delete Node in a linked list  -- Linked List
def deleteNode(self, node):
    nextNode = node.next  # store the next node temporarily
    node.val = nextNode.val
    node.next = nextNode.next
    nextNode = None  # delete the next Node from memory




# 82 Find the duplicate number  -- linked list(weird but yeah)
def findDuplicate(self, nums: List[int]) -> int:
    """
     This can be solved easily using a hashset but of course that's tp trivial for a leetcode medium
     So if you look at the question very well, it is somehow similar to a Linked list Cycle Proble(It's actually hard
     to figure this out in an interview condition).. So basically we find if a cycle exist using fast and slow pointers.
     Then we find the start of the cycle using floyd's algorithm.
     We're sort of using the indices to represent the nodes of the linked list
     https://www.youtube.com/watch?v=wjYnzkAhcNk Neetcode Explains pretty well here!
     """
    fast, slow = 0, 0  # starting with 0 because the cycle cannot start here( remember the numbers ranges from 1 - n)
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if fast == slow:  # we found a cycle
            break

    # now let's find the beginning of the cycle which also happens to be the duplicate number
    slow2 = 0
    while slow2 != slow:
        slow = nums[slow]
        slow2 = nums[slow2]

    return slow2


# 3 Delete duplicates from an unsorted linked list  -- Linked List
def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
    # get the occurence of each node
    # create a new Linked list and only add nodes that occur once
    counter = {}  # {value:frequency}
    current = head
    while current:
        counter[current.val] = 1 + counter.get(current.val, 0)
        current = current.next

    newHead = ListNode(0, head)  # build a new linked list starting from the head and remove nodes that occur more than
    # once, adding the head as the next node of the linked list instead of it being the first node caters for the edge
    # case where the head's value occurs more than once
    current = newHead
    while current.next:
        if counter[current.next.val] > 1:
            current.next = current.next.next
        else:
            current = current.next
    return newHead.next


# 8 LRU Cache  -- Linked List
class Double:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = self.prev = None


class LRUCache:
    # This problem is actually straightforward
    # We only need a data structure that can keep track of items as they are added and can also push them to the
    # front whenever they are used, to get the least recently used we simply return the item at the end.
    # This is where double linked list come in because now we can model the cache as a linked list and specifically
    # a double linked list since they have 'previous' pointers to allow nodes to be push to the front wheneve they
    # are used and also allows insertion and removal of nodes possible without having a reference to the head
    # the head of the double will point to the most recently used while the tail points to the least

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = self.tail = Double(0, 0)
        # the tail and the head should point initially to each other
        self.head.prev, self.tail.next = self.tail, self.head

    # helper method for the Double linked list
    def insert(self, node):
        prev, current = self.head.prev, self.head
        prev.next = current.prev = node
        node.next, node.prev = current, prev

    def remove(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self.remove(node)
            self.insert(node)  # push the cache to the front since it has just been used
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])
        node = Double(key, value)
        self.insert(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            node = self.tail.next  # remove the least recenlty used cache
            self.remove(node)
            del self.cache[node.key]

# 83 Linked List Cycle II
def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head: return None

    def check(head):  # check if cycle exist
        fast = slow = head
        while fast.next and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return slow
        return None

    slow = check(head)
    if not slow: return None  # Find the start of the cycle
    slow2 = head
    while slow2 != slow:
        slow = slow.next
        slow2 = slow2.next

    return slow if slow else None

class MyCircularDeque:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the deque to be k.
        """
        self.deque = [None] * k
        self.front = 0
        self.rear = 0
        self.size = 0
        self.capacity = k

    def insertFront(self, value: int) -> bool:
        """
        Adds an item at the front of Deque. Return true if the operation is successful.
        """
        if self.isFull():
            return False
        self.front = (self.front - 1) % self.capacity
        self.deque[self.front] = value
        self.size += 1
        return True

    def insertLast(self, value: int) -> bool:
        """
        Adds an item at the rear of Deque. Return true if the operation is successful.
        """
        if self.isFull():
            return False
        self.deque[self.rear] = value
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1
        return True

    def deleteFront(self) -> bool:
        """
        Deletes an item from the front of Deque. Return true if the operation is successful.
        """
        if self.isEmpty():
            return False
        self.deque[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return True

    def deleteLast(self) -> bool:
        """
        Deletes an item from the rear of Deque. Return true if the operation is successful.
        """
        if self.isEmpty():
            return False
        self.rear = (self.rear - 1) % self.capacity
        self.deque[self.rear] = None
        self.size -= 1
        return True

    def getFront(self) -> int:
        """
        Get the front item from the deque.
        """
        return self.deque[self.front] if not self.isEmpty() else -1

    def getRear(self) -> int:
        """
        Get the last item from the deque.
        """
        return self.deque[self.rear - 1] if not self.isEmpty() else -1

    def isEmpty(self) -> bool:
        """
        Checks whether the circular deque is empty or not.
        """
        return self.size == 0

    def isFull(self) -> bool:
        """
        Checks whether the circular deque is full or not.
        """
        return self.size == self.capacity
