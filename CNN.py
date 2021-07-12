import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 資料預處理
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalization
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# 利用sklearn，切0.2比例的Validation Data
features_train, features_test, targets_train, targets_test = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)

# 將切好的data轉成torch的tensor
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)

featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

# 將input和output打包
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

# Hyper parameter
learningRate=0.01
batchSize=100
nIters=10000
numEpochs =int( nIters/(len(features_train)/batchSize) )

# Pytorch的DataLoader
trainLoader = torch.utils.data.DataLoader(train, batch_size=batchSize, shuffle=True)
testLoader = torch.utils.data.DataLoader(test, batch_size=batchSize, shuffle=True)

# 建立Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        #輸入的大小(1,28,28)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        #output的大小 (16,24,24)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        #output的大小(16,12,12)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        #output的大小(32,8,8)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        #output的大小(32,4,4)
        self.fc = nn.Linear(32*4*4, 10)

    def forward(self, x):
        # Conv 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        # Conv 2
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        # fully connected layer
        out = self.fc(out)

        # output
        return out

model = CNNModel()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
lossFunction = nn.CrossEntropyLoss()
inputShape = (-1, 1, 28, 28)

# 訓練模型
def fitModel(model, lossFunction, optimizer, inputShape, numEpochs, trainLoader, testLoader):
    trainingLoss=[]
    trainingAccuracy=[]
    validationLoss=[]
    validationAccuracy=[]

    for epoch in range(numEpochs):
        correctTrain=0
        totalTrain=0

        for i, (images,labels) in enumerate(trainLoader):
            # 定義變數
            train = Variable(images.view(inputShape))
            labels = Variable(labels)
            # 清空gradient
            optimizer.zero_grad()
            # 向前傳播
            outputs = model(train)
            # 計算loss
            trainLoss=lossFunction(outputs, labels)
            # 計算gradient
            trainLoss.backward()
            # 更新參數
            optimizer.step()
            # 從最大值取得preditions
            predicted = torch.max(outputs.data, 1)[1]
            # label的總數量
            totalTrain = totalTrain + len(labels)
            # 正確預測的數量
            correctTrain = correctTrain + (predicted==labels).float().sum()

        # 儲存每一個epoch的accuracy
        trainAccuracy = 100*correctTrain/float(totalTrain)
        trainingAccuracy.append(trainAccuracy)
        # 儲存每一個epoch的loss
        trainingLoss.append(trainLoss.data)

        correctTest = 0
        totalTest = 0
        for images, labels in testLoader:
            # 定義變數
            test = Variable(images.view(inputShape))
            # 向前傳播
            outputs = model(test)
            # 計算softmax和cross entropy loss
            valLoss = lossFunction(outputs, labels)
            # 從最大值取得preditions
            predicted = torch.max(outputs.data, 1)[1]
            # 總label數量
            totalTest = totalTest + len(labels)
            # 正確預測的數量
            correctTest = correctTest + (predicted == labels).float().sum()

        # 儲存每一個epoch的accuracy
        valAccuracy = 100*correctTest/float(totalTest)
        validationAccuracy.append(valAccuracy)
        # 儲存每一個epoch的loss
        validationLoss.append(valLoss.data)

        print('Train Epoch: {}/{} Training Loss: {} Train_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, numEpochs, trainLoss.data, trainAccuracy, valLoss.data, valAccuracy))
    return trainLoss, trainingAccuracy, validationLoss, validationAccuracy

trainingLoss, trainingAccuracy, validationLoss, validationAccuracy = fitModel(model, lossFunction, optimizer, inputShape, numEpochs, trainLoader, testLoader)
