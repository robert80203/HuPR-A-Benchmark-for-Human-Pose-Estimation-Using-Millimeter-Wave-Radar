import os

num = 276


try:
    os.mkdir('data')
    os.mkdir('data/HuPR')
    os.mkdir('visualization')
    os.mkdir('logs')
    os.mkdir('preprocessing/raw_data')
    os.mkdir('preprocessing/raw_data/iwr1843')
except ValueError as err:
    print(err)


for i in range(1, num+1):
    root = 'data/HuPR/'
    dirName = root + 'single_' + str(i)
    dirVertName = dirName + '/vert'
    dirHoriName = dirName + '/hori'
    dirAnnotName = dirName + '/annot'
    dirVisName = dirName + '/visualization'
    os.mkdir(dirName)
    os.mkdir(dirVertName)
    os.mkdir(dirHoriName)
    os.mkdir(dirAnnotName)
    os.mkdir(dirVisName)
