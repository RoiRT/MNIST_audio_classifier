import torch
from torch.utils.data import DataLoader
from Tools import *
from torch.utils.data import random_split
from sklearn.metrics import f1_score, recall_score

root_dir = 'D:/Datasets/data'
dataset = load_audio_dataset(root_dir)

audio_ds = AudioDataset(dataset[0], dataset[1])

num_train = round(len(audio_ds) * 0.8)
num_val = len(audio_ds) - num_train
train_ds, val_ds = random_split(audio_ds, [num_train, num_val])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

model = torch.nn.Sequential(
    torch.nn.BatchNorm2d(1),
    torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same'),
    torch.nn.LeakyReLU(0.2),
    torch.nn.BatchNorm2d(16),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same'),
    torch.nn.LeakyReLU(0.2),
    torch.nn.BatchNorm2d(16),
    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
    torch.nn.LeakyReLU(0.2),
    torch.nn.BatchNorm2d(32),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
    torch.nn.LeakyReLU(0.2),
    torch.nn.BatchNorm2d(32),
    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
    torch.nn.LeakyReLU(0.2),
    torch.nn.BatchNorm2d(64),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
    torch.nn.LeakyReLU(0.2),
    torch.nn.BatchNorm2d(64),
    torch.nn.Flatten(),
    torch.nn.Linear(64 * 8 * 10, 10),
    torch.nn.Softmax(dim=1)
)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
EPOCHS = 100
history = []
for epoch in range(EPOCHS):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    true_labels = []
    pred_labels = []
    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        # inputs, labels = data[0].to(device), data[1].to(device)
        inputs, labels = data[0].float(), data[1]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs, 1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        true_labels.extend(labels.tolist())
        pred_labels.extend(prediction.tolist())

        if (i + 1) % 50 == 0:  # print every 10 mini-batches
            print('[%d, %5d/%5d] loss: %.3f' % (epoch + 1, i + 1, len(train_dl), running_loss / (i + 1)))

    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction / total_prediction
    f1 = f1_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    history.append({'loss': avg_loss, 'accuracy': acc, 'f1_score': f1, 'recall': recall})
    print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Recall: {recall:.4f}')
