if __name__ == "__main__":
    from ultralytics import YOLO

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", default="yolov8n-pose.pt", type=str)
    parser.add_argument("-o", "--opset", default=11, type=int)
    args = parser.parse_args()

    model = YOLO(args.model)

    model.export(format="onnx", dynamic=False, opset=args.opset)