from ultralytics import YOLO
import clr
import os

# sys.path.append(".")
clr.AddReference("ModelCrypto")
from ModelCrypto import *

if __name__ == '__main__':
    # freeze_support()
    # Load a model
    # model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
    # # Train the models
    # results = model.train(data=r"C:\Users\liyun\Desktop\PythonYolov12\datasets\cifar10", epochs=2, imgsz=32)
    # success = model.export(format="onnx")
    # print(success)
    # # get the folder path of onnx;
    # onnxPath = os.path.dirname(success)
    # ModelCrypto.EncryptFolder(onnxPath)
    # print("complete!")


    modelPath1= r"D:\WorkRepository\ultralytics\runs\classify\train2\weights\best.gqpt"

    if modelPath1.endswith(".gqpt"):
        #判断是否存在Cache文件夹，不存在则创建；
        if not os.path.exists("Cache"):
            os.makedirs("Cache")
        modelParserPath="./Cache/trainTemp.pt"
        succeed=ModelCrypto.DecryptFunc(modelPath1,modelParserPath)
        if succeed:
            print("Complete!")
        else:
            print("Decrypt fail!")


