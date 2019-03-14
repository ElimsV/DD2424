class Dummy():
    def __init__(self):
        self.a = 1
        self.b = 2
        self.para = [self.a, self.b]

    def update(self):
        self.a += 1
        self.b += 2

if __name__ == "__main__":
    dummy = Dummy()
    print(dummy.para)
    dummy.update()
    print(dummy.para)