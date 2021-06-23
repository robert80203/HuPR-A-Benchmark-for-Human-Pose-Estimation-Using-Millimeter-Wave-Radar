

class Logger():
    def __init__(self):
        self.totalAcc = 0.0
        self.totalCnt = 0
        self.bestAcc = 0.0
    
    def update(self, acc, cnt):
        self.totalAcc += acc
        self.totalCnt += cnt
    
    def clear(self):
        self.totalAcc = 0.0
        self.totalCnt = 0
    
    def updateBestAcc(self, bestAcc):
        self.bestAcc = bestAcc
    
    def showAcc(self, mode='avg'):
        if mode == 'avg':
            return self.totalAcc/self.totalCnt
        elif mode == 'best':
            return self.bestAcc
    
    def isBestAcc(self):
        avgAcc = self.totalAcc/self.totalCnt
        if avgAcc > self.bestAcc:
            self.bestAcc = avgAcc
            return True
        else:
            return False