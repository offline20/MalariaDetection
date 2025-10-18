import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
MODEL_PATH = r"D:\MalariaDetection\model\best_blood_model.pth"
CLASSES = ['Parasitized','Uninfected']

# Load model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

# Select images via GUI
root = Tk()
root.withdraw()
file_paths = filedialog.askopenfilenames(title="Select Images for Prediction")
root.update()
root.destroy()

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# Predict
for path in file_paths:
    img = Image.open(path).convert("RGB")
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out,dim=1)[0]
        pred = CLASSES[int(prob.argmax())]
        print(f"File: {os.path.basename(path)} | Predicted: {pred} | Probabilities: {dict(zip(CLASSES,[float(p) for p in prob]))}")
