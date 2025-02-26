from ultralytics import YOLO

if __name__ == '__main__':
    # freeze_support()
    # Load a model
    model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

    # Train the models
    results = model.train(data=r"C:\Users\liyun\Desktop\PythonYolov12\datasets\cifar10", epochs=2, imgsz=32)
    model.export(format="onnx")
