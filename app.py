import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Model Architecture Define Karein
class HumanDetectionCNN(nn.Module):
    def __init__(self):
        super(HumanDetectionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 2. Model Load Karein (Caching ke sath taaki fast chale)
@st.cache_resource
def load_model():
    model = HumanDetectionCNN()
    model.load_state_dict(torch.load('human_detection_pytorch.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 3. Streamlit Web UI Design
st.title("👤 Human Detection App")
st.write("Apni image upload karein aur check karein ki model kya predict karta hai!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        
    predicted_class = "HUMAN" if prob >= 0.5 else "NON-HUMAN"
    confidence = prob if prob >= 0.5 else (1 - prob)
    
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence * 100:.2f}%")