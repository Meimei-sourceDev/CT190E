from ultralytics import YOLO

model = YOLO("weights/best.pt")

if hasattr(model, 'model') and model.model:
    # Extract the YAML configuration or class name from the PyTorch module
    print(f"Model Architecture Type: {model.model.__class__.__name__}")
    if hasattr(model.model, 'yaml'):
        print(f"Base Configuration File: {model.model.yaml.get('yaml_file', 'Unknown')}")
else:
    print("Could not retrieve internal network graph configuration.")