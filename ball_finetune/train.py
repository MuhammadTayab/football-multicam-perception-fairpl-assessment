from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(
    data="data.yaml",
    imgsz=960,        # high, because the ball is small
    epochs=25,
    batch=10,           # lower if M4 runs out of memory.
    patience=7,       # early stop if no improvement
    #device="mps",      # GPU/mps if applicable
    project="ball_finetune",
)
