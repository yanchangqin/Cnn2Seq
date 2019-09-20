import numpy as np
import os
import matplotlib.pyplot as plt

Image_path = r'code1'
class Imagedata1:
    def __init__(self):
        self.data = []
        for name in os.listdir(Image_path):
            x = plt.imread('{0}/{1}'.format(Image_path,name))
            # x = plt.imread(os.path.join(Image_path,name))
            y =name.split('.')[0]
            y1=self.one_hot(y)
            self.data.append([x,y1])

    def one_hot(self,y):
        z = np.zeros((4,10))
        for i in range(4):
            index = int(y[i])
            z[i][index]=1
        return z

    def getcode(self,batch):
        xss = []
        yss = []
        num = np.random.randint(0,len(self.data))
        for j in range(batch):
            data = self.data[num][0]
            label = self.data[num][1]
            xss.append(data)
            yss.append(label)
        xss = np.array(xss)/255-0.5
        return xss,yss
data = Imagedata1()
# data.one_hot('1628')
