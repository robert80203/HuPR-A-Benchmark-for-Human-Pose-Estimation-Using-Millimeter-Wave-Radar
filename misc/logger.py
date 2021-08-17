

class Logger():
    def __init__(self):
        self.totalAcc = 0.0
        self.totalCnt = 0
        self.bestAcc = 0.0
        self.accTable = None
    
    def update(self, acc, accTable, cnt):
        self.totalAcc += acc
        self.totalCnt += cnt
        if self.accTable is None:
            self.accTable = accTable
        else:
            self.accTable += accTable

    def clear(self):
        self.totalAcc = 0.0
        self.totalCnt = 0
        self.accTable = None
    
    def updateBestAcc(self, bestAcc):
        self.bestAcc = bestAcc
    
    def showAcc(self, mode='avg'):
        if mode == 'avg':
            return self.totalAcc/self.totalCnt if self.totalCnt != 0 else 0
        elif mode == 'best':
            return self.bestAcc
    
    def showAccTable(self, numKeypoints, idxToJoints):
        print('----------------------------------------------------')
        for idx in range(numKeypoints):
            print('%s:\t%.3f\t' %(idxToJoints[idx], self.accTable[idx+1]/self.totalCnt*numKeypoints), end='')
            if (idx + 1) % 3 == 0:
                print('')
        print('\n----------------------------------------------------')

    def isBestAcc(self):
        avgAcc = self.totalAcc/self.totalCnt if self.totalCnt != 0 else 0
        if avgAcc > self.bestAcc:
            self.bestAcc = avgAcc
            return True
        else:
            return False