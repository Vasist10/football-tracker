from ultralytics import YOLO
import torch
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

model = YOLO('best.pt')
results = model.predict('input_files/15sec_input_720p.mp4',save=True,stream=True)

print(results[0])
print("--------------------------------")
for box in results[0].boxes:
    print(box)

