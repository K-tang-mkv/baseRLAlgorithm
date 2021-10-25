class Balance():
    def __init__(self, balance=0):
        self.balance = balance
    def check(self):
        print(self.balance)

class child(Balance):
    def __init__(self, balance):
        super(child, self).__init__(balance)
    
    def go(self):
        super(child, self).check()

li = child(8)
li.go()