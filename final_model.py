from clearml import Task
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import logging

task = Task.init(
    project_name='Agri-Pest-Detection',
    task_name='Step 5 - Final Model Training'
)

params = task.connect({
    'processed_dataset_id': '',  # 必须从 Step 2 继承
    'test_queue': 'pipeline',
    'learning_rate': 0.001,
    'batch_size': 16,
    'weight_decay': 1e-5,
    'dropout_rate': 0.5,
    'num_epochs': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
})

# Execute remotely if a test_queue is specified (comment out if running locally)
task.execute_remotely()

step2_task = Task.get_task(task_id=params['processed_dataset_id'])
X_train = np.load(step2_task.artifacts['X_train.npy'].get_local_copy())
X_test = np.load(step2_task.artifacts['X_test.npy'].get_local_copy())
y_train = np.load(step2_task.artifacts['y_train.npy'].get_local_copy())
y_test = np.load(step2_task.artifacts['y_test.npy'].get_local_copy())
label2id = np.load(step2_task.artifacts['label2id.npy'].get_local_copy(), allow_pickle=True).item()

batch_size = int(params.get('batch_size'))
learning_rate = float(params.get('learning_rate'))
weight_decay = float(params.get('weight_decay'))
dropout_rate = float(params.get('dropout_rate'))
num_epochs = int(params.get('num_epochs'))

X_train = torch.FloatTensor(X_train.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
X_test = torch.FloatTensor(X_test.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

class PestCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(PestCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.global_pool(self.conv3(self.conv2(self.conv1(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

device = torch.device(params['device'])
model = PestCNN(num_classes=len(label2id), dropout_rate=dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return (
        total_loss / len(loader),
        correct / len(loader.dataset),
        precision_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average='macro')
    )

train_accs, val_accs = [], []
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, precision, recall = evaluate(model, test_loader, criterion, device)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"[Epoch {epoch+1}/{num_epochs}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    task.get_logger().report_scalar("accuracy", "train", iteration=epoch, value=train_acc)
    task.get_logger().report_scalar("accuracy", "validation", iteration=epoch, value=val_acc)
    task.get_logger().report_scalar("precision", "val", iteration=epoch, value=precision)
    task.get_logger().report_scalar("recall", "val", iteration=epoch, value=recall)

torch.save(model.state_dict(), "final_model.pth")
task.upload_artifact("final_model_weights", artifact_object="final_model.pth")

plt.figure(figsize=(6, 4))
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("final_acc_curve.png")
task.upload_artifact("accuracy_curve", artifact_object="final_acc_curve.png")

task.get_logger().report_single_value("final_val_accuracy", val_acc)
task.get_logger().report_single_value("final_val_precision", precision)
task.get_logger().report_single_value("final_val_recall", recall)

print("✅ Final model training completed.")
