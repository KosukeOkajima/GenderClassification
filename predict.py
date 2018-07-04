# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import os
import imutils
import face_recognition  # Dlib library
import numpy as np
import glob
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset
from chainer import datasets, iterators, optimizers, serializers
import matplotlib.pyplot as plt

class Alex(chainer.Chain):

    def __init__(self):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3, 48, 3, stride=1),
            bn1=L.BatchNormalization(48),
            conv2=L.Convolution2D(48, 128, 3, pad=1),
            bn2=L.BatchNormalization(128),
            conv3=L.Convolution2D(128, 192, 3, pad=1),
            conv4=L.Convolution2D(192, 192, 3, pad=1),
            conv5=L.Convolution2D(192, 128, 3, pad=1),
            fc6=L.Linear(None, 1024),
            fc7=L.Linear(None, 1024),
            fc8=L.Linear(None, 2)
        )
        self.train = True

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.bn2(self.conv2(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(F.relu(h), 2, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h

def image2data(pathsAndLabels):
    allData = []
    for pathAndLabel in pathsAndLabels:
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + "*")
        for imgName in imagelist:
            allData.append([imgName, label])
    allData = np.random.permutation(allData)

    imageData = []
    labelData = []
    for pathAndLabel in allData:
        img = Image.open(pathAndLabel[0])
        try:
            imgData = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
        except ValueError:
            continue
        imageData.append(imgData)
        labelData.append(np.int32(pathAndLabel[1]))
    predict = tuple_dataset.TupleDataset(imageData, labelData)

    return predict


def main():
    if os.path.exists("input") == False:
        os.mkdir("input")

    if os.path.exists("output") == False:
        os.mkdir("output")

    filename = "input/Image.jpg"

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=1920)
        cv2.imshow('img', frame)

        c = cv2.waitKey(1)
        if c == 27:  # ESCを押してウィンドウを閉じる
            exit()
        if c == 32:  # spaceで保存
            cv2.imwrite(filename, frame)
            print('save done')
            break

    cap.release()
    cv2.destroyAllWindows()


    # Dlib
    dlib_img = face_recognition.load_image_file(filename)

    # Dlib CNN
    #faces = face_recognition.face_locations(dlib_img, model="cnn")  # CNNモデルで顔認識
    faces = face_recognition.face_locations(dlib_img)  # CNNモデルを使わず顔認識


    print "検出結果", len(faces), "人"

    img = Image.open(filename)
    for i in range(len(faces)):
        [top, right, bottom, left] = faces[i]
        print (faces[i])
        imgCroped = img.crop((left, top, right, bottom)).resize((96, 96))
        imgCroped.save("output/faceImage_%02d.jpg" % i)

    cls_names = ['女', '男']

    # まず同じネットワークのオブジェクトを作る
    infer_net = Alex()
    infer_net = L.Classifier(infer_net)

    # そのオブジェクトに保存済みパラメータをロードする
    serializers.load_npz('Alexlike.model', infer_net)

    pathsAndLabels = []
    pathsAndLabels.append(np.asarray(["./output/", 0]))
    predict = image2data(pathsAndLabels)

    for x, t in predict:
        infer_net.to_cpu()
        y = infer_net.predictor(x[None, ...]).data.argmax(axis=1)[0]

        print("予測値 : " + cls_names[y])
        # 推論させる画像を表示
        plt.imshow(x.transpose(1, 2, 0))
        plt.show()


if __name__ == '__main__':

    main()
