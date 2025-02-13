from fileinput import filename
import torch
import torch.nn as nn
import torch.optim as optim
from fontTools.misc.timeTools import epoch_diff
from openpyxl.styles.builtins import total
from sympy import srepr, vectorize
from torch.distributed.checkpoint import load_state_dict
from torch.nn import Conv1d
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import nltk
import openpyxl
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import xlsxwriter
import joblib

loss_train = 0
train_acc = 0
count = 0
best_loss = None
br_p = 0
epochs = 80
vectorize_num = 1
print("Обучение - 1 // Работа - 0")
LearnTask = int(input())
if LearnTask:
    print("Загрузка отдельного файла validate 1 - да // 0 - нет")
    validate_too = int(input())


def convert_excel_to_utf8_csv(excel_file, csv_file):
    # Чтение Excel файла
    df = pd.read_excel(excel_file)

    # Сохранение в CSV файл с кодировкой UTF-16
    df.to_csv(csv_file, index=False, encoding='utf-16')


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Доступ:{torch.cuda.is_available()}, факт:{device} Верси куда: {torch.version.cuda}")


# Dataset class
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, encoding='utf-16')
        self.stop_words = set(stopwords.words('russian'))
        self.vectorizer = CountVectorizer(max_features=16384)
        if LearnTask and vectorize_num:
            self.vectorizer.fit(self.data['Description'].apply(self.preprocess_text))
            joblib.dump(self.vectorizer, "vectorizer_0915.pkl")
            print("save vectorizer")
        else:
            print("load vectorizer")
            self.vectorizer = joblib.load("vectorizer_0915.pkl")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        description = self.data.iloc[idx, 0]
        target = self.data.iloc[idx, 1]
        self.preprocess_text(description)
        description_vector = self.vectorizer.transform([description]).toarray()
        description_tensor = torch.tensor(description_vector, dtype=torch.float32).squeeze()
        target_tensor = torch.tensor(target, dtype=torch.long)  # Изменено на torch.long для CrossEntropyLoss
        return description_tensor, target_tensor

    def preprocess_text(self, text):
        words = word_tokenize(text.lower(), language='russian')
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        return ' '.join(words)


# Neural network model with RNN (GRU)
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4   ):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Единый многослойный GRU
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        # Полносвязные слои
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # Добавляем временную размерность (sequence_length=1)
        x = x.unsqueeze(1)  # [batch, 1, input_size]

        # Инициализация скрытого состояния
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Проход через GRU
        out, _ = self.gru(x, h0)

        # Берем последние скрытое состояние
        out = out[:, -1, :]

        # Полносвязные слои
        return self.fc(out)


# Function to train the model
def train(model, data_loader, criterion, optimizer):
    global loss_train, train_acc, epochs
    model.train()  # Важно: активируем режим обучения перед началом эпохи
    for epoch in range(epochs):
        if br_p:
            break
        model.train()  # Дополнительная гарантия режима обучения
        correct = 0
        total = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Явно указываем режим обучения
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Используем torch.max для получения предсказанных классов
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        loss_train = loss.item()
        train_acc = correct / total
        loss_val, accuracy_val, hot_val_loss = validate(model, val_loader, criterion, epoch)
        print(
            f'Epoch:[{epoch + 1}/{epochs}], Train_Loss: {loss.item():.4f}, Train_Accuracy: {train_acc * 100:.2f}%, Total_val_Loss: {loss_val:.4f}, Accuracy_val: {accuracy_val * 100:.2f}%, Hot Val Loss: {hot_val_loss:.4f}')
    return model


# Function to validate the model
def validate(model, data_loader, criterion, epoch):
    model.eval()
    global br_p, best_loss, count
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            # Расчет метрики
            predicted = round(outputs[0].tolist()[0])
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.numel()
    accuracy = total_correct / total_samples
    count += 1
    # Функция Остановки и hot сохранения модели
    if best_loss == None:
        best_loss = total_loss / len(data_loader)
        print("Best Loss start")
    elif total_loss / len(data_loader) < best_loss and best_loss - total_loss / len(data_loader) >= 0.002:
        best_loss = total_loss / len(data_loader)
        count = 0
        save_model(model, 'model_0915.pth', loss_train, total_loss / len(data_loader), best_loss, train_acc, accuracy,
                   epoch)
        print(
            f'Модель сохранена Epoch: {epoch + 1}, Hot_Loss_Val: {loss.item():.4f}, Accuracy_val: {accuracy * 100:.2f}% Total_val_loss: {total_loss / len(data_loader):4f}')
    elif count >= 50:
        print(f"Обучение оставлено на {epoch + 1}")
        br_p = 1
    return total_loss / len(data_loader), accuracy, loss.item()


# Function to test the model
def test(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Используем torch.max для получения предсказанных классов
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            accuracy = total_correct / total_samples
    return total_loss / len(data_loader), accuracy


# Function to save the model
def save_model(model, path, train_loss, val_loss, best_loss, train_acc, val_acc, epoch):
    checpoint = {
        'state_model': model.state_dict(),
        'state_opt': optimizer.state_dict(),
        'loss': {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': best_loss,
        },
        'metric': {
            'train_acc': train_acc,
            'val_acc': val_acc,
        },
        'lr': lr,
        'epoch': epoch
    }
    torch.save(checpoint, path)


# Function to load the model
def load_model(path_model, path_file_test, csv_file_test):
    convert_excel_to_utf8_csv(path_file_test, csv_file_test)
    dataset_test = MyDataset(csv_file_test)
    test_loader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
    new_model = MyModel(16384, 512, 9)  # Изменены параметры для RNN
    load_model_state = torch.load(path_model, weights_only=False)
    new_model.load_state_dict(load_model_state['state_model'])
    new_model.to(device)
    criterion_test = nn.CrossEntropyLoss()  # Используем CrossEntropyLoss для классификации
    test_loss_test, test_results_test = test(new_model, test_loader_test, criterion_test)
    print(f"Test LOAD Results{test_results_test * 100:.2f}%, Test Loss{test_loss_test:.4f}")
    return new_model


# Function to predict with the possibility to load csv file
def predict(model, excel_file, csv_file):
    convert_excel_to_utf8_csv(excel_file, csv_file)
    model.eval()
    if csv_file:
        dataset = MyDataset(csv_file)
        data_loaderss = DataLoader(dataset, batch_size=1, shuffle=False)
        answer_list = []
        with torch.no_grad():
            for inputs, _ in data_loaderss:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)  # Используем torch.max для получения предсказанных классов
                answer_list.append(predicted.item())
        return answer_list
    else:
        return "PROBLEM CSV FILE PREDICTION"


if __name__ == "__main__":
    if LearnTask:
        print('Напишите названия файла для тренировки')
        xlsx_file_train = input()
        csv_file_train = f'{xlsx_file_train}.csv'
        xlsx_file_train = f'{xlsx_file_train}.xlsx'
        convert_excel_to_utf8_csv(xlsx_file_train, csv_file_train)
        dataset = MyDataset(csv_file_train)

        if validate_too:
            vectorize_num = 0
            print('Напишите названия файла для валидации')
            xlsx_file_val = input()
            csv_file_val = f'{xlsx_file_val}.csv'
            xlsx_file_val = f'{xlsx_file_val}.xlsx'
            convert_excel_to_utf8_csv(xlsx_file_val, csv_file_val)
            val_dataset = MyDataset(csv_file_val)
            train_dataset = dataset
        else:
            train_size = int(0.8 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        print('epochs? Стандартно 80')
        epochs = int(input())
        print('lr? Стандартно 0.01')
        lr = float(input())





        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = MyModel(16384, 512, 9).to(device)  # Изменены параметры для RNN
        criterion = nn.CrossEntropyLoss()  # Используем CrossEntropyLoss для классификации
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model = train(model, train_loader, criterion, optimizer)

        test_loss, test_results = test(model, test_loader, criterion)
        print(f"Test Results{test_results * 100:.2f}%, Test Loss{test_loss:.4f}")
    else:
        loaded_model = load_model('model_0915.pth', 'data_work.xlsx', 'data_work.csv')
        prediction = predict(loaded_model, 'data_work.xlsx', 'data_work.csv')
        print("Prediction:", prediction, len(prediction))