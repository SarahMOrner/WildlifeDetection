from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    model = YOLO('yolov8n-seg.pt')
    model.train(
        data=r'C:\Users\sarah\WildlifeInstSeg\WildlifeSegmentation.v2i.yolov8\data.yaml',
        epochs=50,
        imgsz=640,
        task='segment'
    )