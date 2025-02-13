import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Определение кастомного Dataset
class TextDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, encoding='utf-8')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx, 0]  # Входные данные
        target_text = self.data.iloc[idx, 1]  # Целевые значения
        return input_text, target_text

# Функция для преобразования данных в батчи

def pad_sequence(seq, max_length, pad_value=0):
    if len(seq) < max_length:
        seq.extend([pad_value] * (max_length - len(seq)))
    else:
        seq = seq[:max_length]
    return seq

def csv_to_batches(file_path, batch_size, val_split=0.2):
    dataset = TextDataset(file_path)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Пример использования
file_path = 'True-Test-answet.csv'
batch_size = 64
train_loader, val_loader = csv_to_batches(file_path, batch_size)

# Печать первых батчей для тренировки и валидации
print("Training Batches:")
for i, batch in enumerate(train_loader):
    if i == 2: break  # Печатаем только первые два батча
    input_texts, target_texts = batch
    print(f'Batch {i + 1}:')
    print('Input texts:', input_texts)
    print('Target texts:', target_texts)

print("\nValidation Batches:")
for i, batch in enumerate(val_loader):
    if i == 2: break  # Печатаем только первые два батча
    input_texts, target_texts = batch
    print(f'Batch {i + 1}:')
    print('Input texts:', input_texts)
    print('Target texts:', target_texts)
print("Определение модели")
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
       # self.embedding = nn.Embedding(, 512)
        self.linear_2 = nn.Linear(input_dim, 256)
        self.act_2 = nn.ReLU()
        self.linear_3 = nn.Linear(256, 128)
        self.act_3 = nn.ReLU()
        self.linear_4 = nn.Linear(128, 64)
        self.act_4 = nn.ReLU()
        self.linea_5 = nn.Linear(64, 32)
        self.act_5 = nn.ReLU()
        self.linear_6 = nn.Linear(32, 16)
        self.act_6 = nn.ReLU()
        self.linear_7 = nn.Linear(16, output_dim)

    def forward(self, x):
       # x = self.embedding(x)
        x = self.linear_2(x)
        x = self.act_2(x)
        x = self.linear_3(x)
        x = self.act_3(x)
        x = self.linear_4(x)
        x = self.act_4(x)
        x = self.linear_5(x)
        x = self.act_5(x)
        x = self.linear_6(x)
        x = self.act_6(x)
        x = self.linear_7(x)
        return x

input_dim = 600  # Размер словаря для Embedding слоя
output_dim = 10
model = SimpleModel(input_dim, output_dim)


# Шаг 9: Определение функции потерь и оптимизатора
criterion = nn.MSELoss()  # Метрический критерий (среднеквадратичная ошибка)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Определение функции потерь и оптимизатора")
# Шаг 10: Обучение модели с валидацией
num_epochs = 5

# Функция обучения с валидацией
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            # Преобразование входных данных в индексированные последовательности
            inputs = [ord(char) for char in ''.join(inputs)]  #   inputs = torch.tensor([ord(char) for char in ''.join(inputs)], dtype=torch.long).view(-1, 1)
            inputs = torch.tensor(inputs, dtype=torch.float).view(1, -1)
            inputs = inputs.reshape(-1, 100000).to(device)
            targets = torch.tensor(targets, dtype=torch.float)
            targets = targets.reshape(-1).to(device)
            targets = torch.eye(10)[targets].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_loader):.4f}')

        # Валидация модели
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = torch.tensor([ord(char) for char in ''.join(inputs)], dtype=torch.long).view(-1, 1)
                targets = targets.long()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')


# Обучение модели с валидацией
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)