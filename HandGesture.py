import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# CONFIGURATION
BASE_DIR = "C:/Users/Atharva/PycharmProjects/SkillCraft/TASK 4"
DATA_DIR = os.path.join(BASE_DIR, "data", "gesture_images")
MODEL_PATH = os.path.join(BASE_DIR, "models", "gesture_model.pkl")
CSV_PATH = os.path.join(BASE_DIR, "results", "predictions.csv")
CM_PATH = os.path.join(BASE_DIR, "results", "confusion_matrix.png")
IMG_SIZE = 64  # All images resized to 64x64

# Create missing output folders
for path in [os.path.dirname(MODEL_PATH), os.path.dirname(CSV_PATH), os.path.dirname(CM_PATH)]:
    os.makedirs(path, exist_ok=True)

# Check data folder
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ Image folder not found: {DATA_DIR}")

# Load gesture classes
class_labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
if not class_labels:
    raise ValueError(f"❌ No class folders found inside: {DATA_DIR}")

print(f"📁 Classes detected: {class_labels}")
X, y = [], []

print("\n📥 Loading images...")
for label in class_labels:
    folder = os.path.join(DATA_DIR, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)
print(f"✅ Loaded {len(X)} samples.")

if len(X) == 0:
    raise ValueError("❌ No valid images loaded. Please check dataset folders.")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print("\n🧠 Training SVM Classifier...")
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(clf, f)

# Save predictions
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.to_csv(CSV_PATH, index=False)

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(CM_PATH)
plt.close()

print("\n✅ Model saved at:", MODEL_PATH)
print("✅ Predictions saved at:", CSV_PATH)
print("✅ Confusion matrix saved at:", CM_PATH)
