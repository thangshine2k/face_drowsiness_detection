import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import shutil, random
from PIL import Image

# =========================
# CONFIG
# =========================
DATA_DIR = "data/processed"   # data/processed/train, data/processed/val
MODEL_PATH = "drowsiness_detector.pth"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# AUTO SPLIT DATA
# =========================
def auto_split_data(train_dir, val_dir, split_ratio=0.8):
    if os.path.exists(val_dir):
        print("✅ Validation folder already exists → skip splitting")
        return

    print("⚠ Validation folder not found → splitting train data into train/val ...")

    for label in ["awake", "drowsy"]:
        src = os.path.join(train_dir, label)
        if not os.path.exists(src):
            print(f"❌ Missing folder: {src}")
            continue

        files = os.listdir(src)
        random.shuffle(files)
        split_idx = int(len(files) * split_ratio)

        train_out = os.path.join(train_dir, label)
        val_out = os.path.join(val_dir, label)
        os.makedirs(val_out, exist_ok=True)

        for f in files[split_idx:]:  # move 20% → val
            src_path = os.path.join(src, f)
            dst_path = os.path.join(val_out, f)
            shutil.copy(src_path, dst_path)

    print("✅ Done splitting train/val")


# =========================
# MODEL DEFINITION
# =========================
def build_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Awake / Drowsy
    return model


# =========================
# TRAIN FUNCTION
# =========================
def train_model():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    auto_split_data(train_dir, val_dir)  # 👈 Gọi auto split

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                output = model(imgs)
                _, preds = torch.max(output, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f} - Val Acc: {acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model trained and saved:", MODEL_PATH)


# =========================
# REAL-TIME DETECTION
# =========================
def run_realtime():
    # Load model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Haar Cascade để detect mắt
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    cap = cv2.VideoCapture(0)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    labels = ["Drowsy", "Awake"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

        status = "Drowsy"   # mặc định là Awake

        if len(eyes) > 0:
            (x, y, w, h) = eyes[0]
            eye_img = frame[y:y+h, x:x+w]
            eye_rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
            eye_pil = Image.fromarray(eye_rgb)
            eye_tensor = transform(eye_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(eye_tensor)
                _, pred = torch.max(output, 1)
                status = labels[pred.item()]

        # Hiển thị text cố định góc trái trên
        cv2.putText(frame, status, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if status=="Drowsy" else (0,0,255), 2)

        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# MAIN ENTRY
# =========================
if __name__ == "__main__":
    print("Select mode:")
    print("1 - Train model")
    print("2 - Run real-time detection")
    choice = input("Enter choice: ")

    if choice == "1":
        train_model()
    elif choice == "2":
        run_realtime()
    else:
        print("❌ Invalid choice")
