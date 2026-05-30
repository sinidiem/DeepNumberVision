import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
import numpy as np
import cv2
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Scan(nn.Module):
    def __init__(self):
        super(Scan, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout2(self.relu3(self.bn3(self.fc1(x))))
        x = self.fc2(x)
        return x

model = Scan().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_digits(input_image):
    if input_image is None:
        return "Please upload an image."
    
    gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)
        
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 5 and h > 15:
            digit_boxes.append((x, y, w, h))
            
    if not digit_boxes:
        return "No distinct digits detected. Please use a thicker brush tool."
        
    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])
    
    predictions = []
    confidences = []
    
    for idx, (x, y, w, h) in enumerate(digit_boxes):
        crop = thresh[y:y+h, x:x+w]
        
        pad = max(w, h) // 4
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        
        pil_crop = Image.fromarray(crop)
        image_tensor = transform(pil_crop).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, predicted = torch.max(outputs, 1)
            
            digit = predicted.item()
            confidence = probabilities[digit].item() * 100
            
            predictions.append(str(digit))
            confidences.append(f"Digit {idx+1} ('{digit}'): {confidence:.1f}% certainty")
            
    final_sequence = "".join(predictions)
    
    prediction_text = f"Scanned Number: {final_sequence}\n"
    prediction_text += f"Total Digits Found: {len(predictions)}\n"
    prediction_text += "\n".join(confidences)
    
    return prediction_text

theme = gr.themes.Monochrome(
    font=[gr.themes.GoogleFont("Share Tech Mono"), "Courier New", "monospace"],
    primary_hue="green",
    neutral_hue="slate",
).set(
    body_background_fill_dark="#030712",
    block_background_fill_dark="#0f172a",
    block_border_width="2px",
    block_border_color_dark="#22c55e",
    button_primary_background_fill_dark="#15803d",
    button_primary_text_color_dark="#030712"
)

css = """
footer {visibility: hidden}
textarea {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    color: #22c55e !important;
    background-color: #030712 !important;
}
"""

interface = gr.Interface(
    fn=predict_digits,
    inputs=gr.Image(sources=["upload"]),
    outputs=gr.Textbox(lines=12, label="MAIN_OUTPUT"),
    title="@sinidiem // Neural Scanner v1",
    description="Attach files for model desconstruction. Github: @sinidiem",
    allow_flagging="never",
    theme=theme,
    css=css
)

if __name__ == "__main__":
    interface.launch(theme=theme, css=css)