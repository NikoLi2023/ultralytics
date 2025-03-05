from ultralytics import YOLO
import GPUtil
import time
import os
from pathlib import Path
from ultralytics.utils import get_user_config_dir
import shutil
import clr
import sys
import io

clr.AddReference("ModelCrypto")
ClassifyDatasFolder: str = "ClassifyDatas"
from ModelCrypto import *


class ModelType:
    Nano = 0
    Small = 1
    Medium = 2
    Large = 3
    ExtraLarge = 4


class TaskType:
    Detect = 0
    Segment = 1
    Classify = 2
    Pose = 3


def GQImageTrainV2(
        modelType, datacfg, epochs, tasktype=TaskType.Detect, IsExportOnnx=True, devices="0"):
    ptFilePath = "PT_V2/GQImagen.gqpt"
    if modelType == ModelType.Nano:
        if tasktype == TaskType.Detect:
            ptFilePath = "PT_V2/GQImagen.gqpt"
        elif tasktype == TaskType.Segment:
            ptFilePath = "PT_V2/GQImagen-seg.gqpt"
        elif tasktype == TaskType.Classify:
            ptFilePath = "PT_V2/GQImagen-cls.gqpt"
    elif modelType == ModelType.Small:
        if tasktype == TaskType.Detect:
            ptFilePath = "PT_V2/GQImages.gqpt"
        elif tasktype == TaskType.Segment:
            ptFilePath = "PT_V2/GQImages-seg.gqpt"
        elif tasktype == TaskType.Classify:
            ptFilePath = "PT_V2/GQImages-cls.gqpt"
    elif modelType == ModelType.Medium:
        if tasktype == TaskType.Detect:
            ptFilePath = "PT_V2/GQImagem.gqpt"
        elif tasktype == TaskType.Segment:
            ptFilePath = "PT_V2/GQImagem-seg.gqpt"
        elif tasktype == TaskType.Classify:
            ptFilePath = "PT_V2/GQImagem-cls.gqpt"
    elif modelType == ModelType.Large:
        if tasktype == TaskType.Detect:
            ptFilePath = "PT_V2/GQImagel.gqpt"
        elif tasktype == TaskType.Segment:
            ptFilePath = "PT_V2/GQImagel-seg.gqpt"
        elif tasktype == TaskType.Classify:
            ptFilePath = "PT_V2/GQImagel-cls.gqpt"
    elif modelType == ModelType.ExtraLarge:
        if tasktype == TaskType.Detect:
            ptFilePath = "PT_V2/GQImagex.gqpt"
        elif tasktype == TaskType.Segment:
            ptFilePath = "PT_V2/GQImagex-seg.gqpt"
        elif tasktype == TaskType.Classify:
            ptFilePath = "PT_V2/GQImagex-cls.gqpt"

    isDelete: bool = False
    modelParserPath = "./Cache/trainTemp1.pt"
    # 判断是否存在Cache文件夹，不存在则创建；
    if not os.path.exists("Cache"):
        os.makedirs("Cache")
    succeed = ModelCrypto.DecryptFunc(ptFilePath, modelParserPath)
    if succeed:
        isDelete = True
    else:
        print("[Error]   Decrypt fail!")
        return

    model = YOLO(modelParserPath)  # load a pretrained model which is decrypted;
    if tasktype == TaskType.Detect or tasktype == TaskType.Segment:
        results = model.train(data=datacfg, epochs=epochs, device=devices)
        if IsExportOnnx:
            success = model.export(format="onnx")
            onnxPath = os.path.dirname(success)
            ModelCrypto.EncryptFolder(onnxPath)
            print("complete: 【" + onnxPath + "】")
    elif tasktype == TaskType.Classify:
        results = model.train(data=datacfg, epochs=epochs, device=devices, imgsz=224)
        if IsExportOnnx:
            success = model.export(format="onnx")
            onnxPath = os.path.dirname(success)
            ModelCrypto.EncryptFolder(onnxPath)
            print("complete: 【" + onnxPath + "】")

    if isDelete:  # 最后删除解密的模型文件
        os.remove(modelParserPath)


def GQImageTrain2(
        modePath, datacfg, epochs, tasktype=TaskType.Detect, IsExportOnnx=True, devices="0"):
    isDelete: bool = False
    modelParserPath = "./Cache/trainTemp.pt"
    if modePath.endswith(".gqpt"):
        # 判断是否存在Cache文件夹，不存在则创建；
        if not os.path.exists("Cache"):
            os.makedirs("Cache")
        succeed = ModelCrypto.DecryptFunc(modePath, modelParserPath)
        if succeed:
            modePath = modelParserPath
            isDelete = True
        else:
            print("[Error]   Decrypt fail!")
            return
    model = YOLO(modePath)  # load a pretrained model (recommended for training)
    if tasktype == TaskType.Detect or tasktype == TaskType.Segment:
        results = model.train(data=datacfg, epochs=epochs, device=devices)
        if IsExportOnnx:
            success = model.export(format="onnx")
            onnxPath = os.path.dirname(success)
            ModelCrypto.EncryptFolder(onnxPath)
            print("complete: 【" + onnxPath + "】")

    elif tasktype == TaskType.Classify:
        results = model.train(data=datacfg, epochs=epochs, device=devices, imgsz=224)
        if IsExportOnnx:
            success = model.export(format="onnx")
            onnxPath = os.path.dirname(success)
            ModelCrypto.EncryptFolder(onnxPath)
            print("complete: 【" + onnxPath + "】")
    if isDelete:  # 最后删除解密的模型文件
        os.remove(modelParserPath)


if __name__ == "__main__":
    print("Current Version: 25.3.3")
    try:
        GPUs = GPUtil.getGPUs()
        gpuCount = 0
        devices = "0"
        if len(GPUs) > 1:
            for gpu in GPUs:
                print(
                    f"【{gpuCount}】 GPUID: {gpu.id}  GPU Name:{gpu.name} GPU Total Memory: <<{gpu.memoryTotal}>> GPU Used Memory: <<{gpu.memoryUsed}>> GPU Used Free: <<{gpu.memoryFree}>>"
                )
                gpuCount += 1
            print(
                "please choose GPUs to Train, When using Multiple GPUs, Please use comma to split. Format Example: 0,1,2"
            )
            devices = input("Your Choice:")
        # 检查字体
        name = Path("Arial.ttf").name

        # Check USER_CONFIG_DIR
        file = Path(
            os.getenv("YOLO_CONFIG_DIR") or get_user_config_dir()
        )  # Ultralytics settings dir / name
        # get absolute path string
        file = os.path.join(file, name)
        if not os.path.exists(file):
            print("Arial.ttf not found,Copying to System folder!")
            shutil.copy("Arial.ttf", file)
        # 选择是进行视觉分割训练还是目标检测训练
        print("Please choose the task type:")
        print("1: Detect")
        print("2: Segment")
        print("3: Classify")
        modelType = int(input("Your choice:"))
        if modelType == 1:
            tasktype = TaskType.Detect
        elif modelType == 2:
            tasktype = TaskType.Segment
        elif modelType == 3:
            tasktype = TaskType.Classify
        # input 1: choose an existing model file ;input 2: choose an initial model type
        print("1:choose an existing model file and Continue incremental training ")
        print("2:choose an initial model type and retrain ")
        choice = int(input("Your choice:"))
        if choice == 1:  # train a existing model;
            # please input the model file path
            ptFilePath = input("please input the model file path:")
            if not os.path.exists(ptFilePath):
                print("Model file not found!")
                input("please enter any key to exit!")
                exit()
            else:
                if tasktype != TaskType.Classify:
                    file_names = []
                    for root, dirs, files in os.walk("DataCfg"):
                        for file in files:
                            file_names.append(file)
                    print("Please select blew a cfg file::")
                    index = 0
                    for eachfile in file_names:
                        print(index, " ", eachfile)
                        index += 1
                    datacfg = file_names[int(input("Your choice:"))]
                    print("Please input the epochs(输入训练的次数):")
                    epochs = int(input("epochs:"))
                    starttime = time.time()
                    GQImageTrain2(
                        ptFilePath,
                        "DataCfg\\" + datacfg,
                        epochs,
                        tasktype,
                        True,
                        devices.strip(),
                    )
                    endtime = time.time()
                    excution_time = endtime - starttime
                    print(f"Execution time: 【{excution_time}】 seconds")
                    input("please enter any key to exit!")
                    pass
                else:  # 分类训练
                    # 如果不存在ClassifyDatas创建该文件夹
                    if not os.path.exists(ClassifyDatasFolder):
                        os.makedirs(ClassifyDatasFolder)
                    # 遍历ClassifyDatas文件夹下的所有文件夹名称
                    folders = []
                    for folder_name in os.listdir(ClassifyDatasFolder):
                        folders.append(folder_name)
                    print("Please select blew a classify folder:")
                    index = 0
                    for eachfolder in folders:
                        print(index, " ", eachfolder)
                        index += 1
                    folderPath = os.path.join(ClassifyDatasFolder, folders[int(input("Your choice:"))])
                    print("Please input the epochs(输入训练的次数):")
                    epochs = int(input("epochs:"))
                    starttime = time.time()
                    GQImageTrain2(
                        ptFilePath,
                        folderPath,
                        epochs,
                        tasktype,
                        True,
                        devices.strip(),
                    )
                    endtime = time.time()
                    excution_time = endtime - starttime
                    print(f"Execution time: 【{excution_time}】 seconds")
                    input("please enter any key to exit!")
        else:  # train an initial new model;
            print("Please choose an initial model type:")
            print("0: Nano")
            print("1: Small")
            print("2: Medium")
            print("3: Large")
            print("4: ExtraLarge")
            modelType = int(input("Your choice:"))
            if tasktype != TaskType.Classify:
                file_names = []
                for root, dirs, files in os.walk("DataCfg"):
                    for file in files:
                        file_names.append(file)
                print("Please select blew a cfg file::")
                index = 0
                for eachfile in file_names:
                    print(index, " ", eachfile)
                    index += 1
                datacfg = file_names[int(input("Your choice:"))]
                print("Please input the epochs(输入训练的次数):")
                epochs = int(input("epochs:"))
                starttime = time.time()
                GQImageTrainV2(
                    modelType,
                    "DataCfg\\" + datacfg,
                    epochs,
                    tasktype,
                    True,
                    devices.strip(),
                )
                endtime = time.time()
                excution_time = endtime - starttime
                print(f"Execution time: 【{excution_time}】 seconds")
                input("please enter any key to exit!")
            else:
                # 如果不存在ClassifyDatas创建该文件夹
                if not os.path.exists(ClassifyDatasFolder):
                    os.makedirs(ClassifyDatasFolder)
                # 遍历ClassifyDatas文件夹下的第一层件夹名称
                folders = []
                for folder_name in os.listdir(ClassifyDatasFolder):
                    folders.append(folder_name)
                print("Please select blew a classify folder:")
                index = 0
                for eachfolder in folders:
                    print(index, " ", eachfolder)
                    index += 1
                folderPath = os.path.join(ClassifyDatasFolder, folders[int(input("Your choice:"))])
                print("Please input the epochs(输入训练的次数):")
                epochs = int(input("epochs:"))
                starttime = time.time()
                GQImageTrainV2(
                    modelType,
                    folderPath,
                    epochs,
                    tasktype,
                    True,
                    devices.strip(),
                )
                endtime = time.time()
                excution_time = endtime - starttime
                print(f"Execution time: 【{excution_time}】 seconds")
    except Exception as e:
        print(str(e))
    finally:
        input("please enter any key to exit!")
        pass
