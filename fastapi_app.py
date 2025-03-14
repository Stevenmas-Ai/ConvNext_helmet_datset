import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torchvision.transforms as transforms

# Load the TorchScript model
model = torch.jit.load("helmet_detection_model.pt")
model.eval()

# Define class labels
class_mapping = {0: "No Helmet", 1: "Helmet"}

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize FastAPI app
app = FastAPI(title="Helmet Detection API", description="Detects helmets in images", version="1.0")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, 1).item()

    return {"prediction": class_mapping[predicted_class]}

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)