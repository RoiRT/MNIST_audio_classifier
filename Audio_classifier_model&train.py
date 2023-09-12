import torch
from torch.utils.data import DataLoader
from Tools import *
from torch.utils.data import random_split
from sklearn.metrics import recall_score, precision_score, confusion_matrix

root_dir = 'D:/Datasets/data'
dataset = load_audio_dataset(root_dir)

audio_ds = AudioDataset(dataset[0], dataset[1])

num_train = round(len(audio_ds) * 0.8)
num_val = len(audio_ds) - num_train
train_ds, val_ds = random_split(audio_ds, [num_train, num_val])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=True)

model = torch.nn.Sequential(
    torch.nn.BatchNorm2d(1),
    torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding='same'),
    torch.nn.LeakyReLU(0.2),
    torch.nn.BatchNorm2d(4),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding='same'),
    torch.nn.LeakyReLU(0.2),
    torch.nn.BatchNorm2d(8),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'),
    torch.nn.LeakyReLU(0.2),
    torch.nn.BatchNorm2d(16),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Flatten(),
    torch.nn.Linear(16 * 8 * 10, 10),
)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
val_criterion = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

EPOCHS = 25
history = []

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    true_labels = []
    pred_labels = []

    model.train()
    for i, data in enumerate(train_dl):
        inputs, labels = data[0].float().to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, prediction = torch.max(outputs, 1)
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        true_labels.extend(labels.tolist())
        pred_labels.extend(prediction.tolist())

        # if (i + 1) % 50 == 0:
        #     print('[%d, %5d/%5d] loss: %.3f' % (epoch + 1, i + 1, len(train_dl), running_loss / (i + 1)))

    # Calcular métricas de rendimiento en el conjunto de entrenamiento
    train_loss = running_loss / len(train_dl)
    train_acc = correct_prediction / total_prediction
    train_recall = recall_score(true_labels, pred_labels, average='macro')
    train_precision = precision_score(true_labels, pred_labels, average='macro')

    # Validación
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    true_labels = []
    pred_labels = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dl):
            inputs, labels = data[0].float().to(device), data[1].to(device)

            outputs = model(inputs)
            loss = val_criterion(outputs, labels)

            running_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            true_labels.extend(labels.tolist())
            pred_labels.extend(prediction.tolist())

    val_loss = running_loss / len(val_dl)
    val_acc = correct_prediction / total_prediction
    val_recall = recall_score(true_labels, pred_labels, average='macro')
    val_precision = precision_score(true_labels, pred_labels, average='macro')

    history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'train_precision': train_acc,
        'train_recall': train_recall,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_precision': val_acc,
        'val_recall': val_recall
    })

    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}')
    print(f'         Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}')
