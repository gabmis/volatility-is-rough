class Link:
    def __init__(self, value, next):
        self.value = value
        self.next = next

    def setValue(value):
        self.value = value

    def setNext(next):
        self.next = next

class LinkedList:
    def __init__(self, value):
        head = Link(value, None)
        self.head = head
        self.last = head
        self.lenght = 1

    def addLast(self, value):
        node = Link(value, None)
        if (self.lenght < 30):
            self.last.next = node
            self.last = node
            self.lenght += 1
        else:
            self.last.next = node
            self.last = node
            self.head = self.head.next

    def getValues(self):
        res = []
        cur = self.head
        for i in range (self.lenght):
            res.append(cur.value)
            cur = cur.next
        return res
