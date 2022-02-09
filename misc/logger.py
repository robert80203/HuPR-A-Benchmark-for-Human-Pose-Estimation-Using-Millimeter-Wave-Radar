from tqdm import tqdm
import numpy as np
# from tqdm import tqdm as std_tqdm
# from functools import partial


# class Logger():
#     def __init__(self, findMax=True):
#         self.totalAcc = 0.0
#         self.totalCnt = 0
#         self.bestAcc = -1
#         self.accTable = None
#         self.progressBar = None
#         self.findMax = findMax
    
#     def update(self, acc, accTable, cnt):
#         self.totalAcc += acc
#         self.totalCnt += cnt
#         if self.accTable is None:
#             self.accTable = accTable
#         else:
#             self.accTable += accTable
    
#     def clear(self, loaderSize):
#         self.totalAcc = 0.0
#         self.totalCnt = 0
#         self.accTable = None
#         self.progressBar = tqdm(total=loaderSize)
    
#     def updateBestAcc(self, bestAcc):
#         self.bestAcc = bestAcc
    
#     def display(self, loss, loss2, updateSize, epoch):
#         if loss2 is not None:
#             self.progressBar.set_postfix(EP=epoch, Loss=loss.item(), Loss2=loss2.item(), ACC=self.showAcc())
#         else:
#             self.progressBar.set_postfix(EP=epoch, Loss=loss.item(), ACC=self.showAcc())
#         self.progressBar.update(updateSize)

#     def showAcc(self, mode='avg'):
#         if mode == 'avg':
#             return self.totalAcc/self.totalCnt if self.totalCnt != 0 else 0
#         elif mode == 'best':
#             return self.bestAcc
    
#     def showAccTable(self, numKeypoints, idxToJoints):
#         print('----------------------------------------------------')
#         for idx in range(numKeypoints):
#             print('%s:\t%.3f\t' %(idxToJoints[idx], self.accTable[idx+1]/self.totalCnt*numKeypoints), end='')
#             if (idx + 1) % 3 == 0:
#                 print('')
#         print('\n----------------------------------------------------')

#     def isBestAcc(self):
#         avgAcc = self.totalAcc/self.totalCnt if self.totalCnt != 0 else 0
#         if self.bestAcc == -1:
#             self.bestAcc = avgAcc
#             return True
        
#         if self.findMax and avgAcc > self.bestAcc:
#             self.bestAcc = avgAcc
#             return True
#         elif not self.findMax and avgAcc < self.bestAcc:
#             self.bestAcc = avgAcc
#             return True
#         else:
#             return False


class Logger():
    def __init__(self, findMax=True):    
        self.bestAcc = -1
        self.totalAcc = None
        self.totalCnt = None
        self.accTable = None
        self.progressBar = None
        self.findMax = findMax
        np.set_printoptions(precision=3)
    
    def update(self, acc, accTable, cnt):
        if self.accTable is None:
            self.accTable = np.array(accTable)
            self.totalAcc = np.array(acc)
            self.totalCnt = np.array(cnt)
        else:
            self.accTable += np.array(accTable)
            self.totalAcc += np.array(acc)
            self.totalCnt += np.array(cnt)
    
    def clear(self, loaderSize):
        self.datasize = 0
        self.totalAcc = None
        self.totalCnt = None
        self.accTable = None
        #tqdm = partial(std_tqdm, dynamic_ncols=True)
        self.progressBar = tqdm(total=loaderSize)
        

    def updateBestAcc(self, bestAcc):
        self.bestAcc = bestAcc
    
    def display(self, loss, loss2, updateSize, epoch):
        if loss2 is not None:
            self.progressBar.set_postfix(EP=epoch, Loss=loss.item(), Loss2=loss2.item(), ACC=self.showAcc())
        else:
            self.progressBar.set_postfix(EP=epoch, Loss=loss.item(), ACC=self.showAcc())
        self.progressBar.update(updateSize)

    def showAcc(self, mode='avg'):
        if mode == 'avg':
            return self.totalAcc/self.totalCnt
        elif mode == 'best':
            return self.bestAcc
    
    def showAccTable(self, numKeypoints, idxToJoints, metric):
        if metric == 'APCK':
            idxgroup = [0, 6] # PCKh@0.2, PCKh@0.5
            for j in idxgroup:
                print('======================table=============================')
                for idx in range(numKeypoints):
                    print('%s:\t%.3f\t' %(idxToJoints[idx], self.accTable[j][idx]/self.totalCnt[j]), end='')
                    if (idx + 1) % 3 == 0:
                        print('')
                print('\n========================================================')
        elif metric == 'OKS':
            idxgroup = [0] #
            for j in idxgroup:
                print('======================table=============================')
                for idx in range(numKeypoints):
                    print('%s:\t%.3f\t' %(idxToJoints[idx], self.accTable[j][idx]/self.totalCnt[j]), end='')
                    if (idx + 1) % 3 == 0:
                        print('')
                print('\n========================================================')

    def isBestAcc(self):
        avgAcc = self.totalAcc[0]/self.totalCnt[0] if self.totalCnt[0] != 0 else 0
        if self.bestAcc == -1:
            self.bestAcc = avgAcc
            return True
        
        if self.findMax and avgAcc > self.bestAcc:
            self.bestAcc = avgAcc
            return True
        elif not self.findMax and avgAcc < self.bestAcc:
            self.bestAcc = avgAcc
            return True
        else:
            return False