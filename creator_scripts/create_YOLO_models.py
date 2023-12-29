from ultralytics import YOLO
ModelNames = [ 'yolov8l-seg.pt','yolov8x-seg.pt','yolov8s-seg.pt','yolov8n-seg.pt','yolov8m-seg.pt'] 
for model_name in ModelNames:
  
  model = YOLO(model_name)
  model.export(format='saved_model', imgsz = 640)
  model.export(format = 'openvino', imgsz = 640, opset=10, keras = True, half = True)
  model.export(format='edgetpu', imgsz = 640)

