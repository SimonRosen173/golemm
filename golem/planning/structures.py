import heapq


class PriorityQueue:
    def __init__(self):
        self._q = []

    def is_empty(self):
        return len(self._q) == 0

    def pop(self):
        priority, item = heapq.heappop(self._q)
        return item

    def push(self, item, priority):
        heapq.heappush(self._q, (priority, item))

    def to_list(self):
        return self._q