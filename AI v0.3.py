from fileinput import filename
import torch
import torch.nn as nn
import torch.optim as optim
from fontTools.misc.timeTools import epoch_diff
from openpyxl.styles.builtins import total
from sympy import srepr
from sympy.codegen import Print
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

loss_train = 0
count = 0
best_loss = None
br_p = 0
print("Обучение - 1 // Работа - 0")
LearnTask = int(input())
unique_words = set() # Создаем пустое множество для уникальных слов


def add_unique_words_to_set_from_string(input_string):
    global unique_word
    words = input_string.split()  # Разбиваем строку на слова
    for word in words:
        unique_words.add(word)  # Добавляем каждое слово в множество
    return unique_words

def convert_excel_to_utf8_csv(excel_file, csv_file):
    # Чтение Excel файла
    df = pd.read_excel(excel_file)

    # Сохранение в CSV файл с кодировкой UTF-16
    df.to_csv(csv_file, index=False, encoding='utf-16')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Доступ:{torch.cuda.is_available()}, факт:{device} Верси куда: {torch.version.cuda}")


# Dataset class
class MyDataset(Dataset):
    counter = 0
    load_count = 0
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, encoding='utf-16')
        self.stop_words = set(stopwords.words('russian'))
        self.vectorizer = CountVectorizer(max_features=1000)  # Adjust the number of features as needed
        self.vectorizer.fit(self.data['Description'].apply(self.preprocess_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        global LearnTask, unique_words, counter, load_count
        description = self.data.iloc[idx, 0]
        target = self.data.iloc[idx, 1]
        if LearnTask:
            description = self.preprocess_text(description)
            unique_words = add_unique_words_to_set_from_string(description)
            self.counter += 1
            if self.counter >= 76040*1.5 and epoch == 1:
                self.save_set_to_file('Uniq.txt')
                print(self.counter)
        else:
            if load_count == 0:
                self.load_set_from_file('Uniq.txt')
                self.preprocess_text(description)
                load_count += 1
            else:
                self.preprocess_text(description)
        #print(description)
        description_vector = self.vectorizer.transform([description]).toarray()
        description_tensor = torch.tensor(description_vector, dtype=torch.float32).squeeze()
        target_tensor = torch.tensor([target] * 9, dtype=torch.float32)  # Adjust target size to match output
        return description_tensor, target_tensor

    def preprocess_text(self, text):
        words = word_tokenize(text.lower(), language='russian')
        #print(unique_words)
        if LearnTask:
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
        else:
            words = [word for word in words if word.isalnum() and word in unique_words]
        return ' '.join(words)

    def save_set_to_file(self, filen):
        global unique_words
        # Преобразование множества в строку
        set_string = ",".join(map(str, unique_words))
        # Сохранение строки в файл
        with open(filen, "w", encoding='utf-16') as file:
            file.write(set_string)
        print(f"Файл сохранен с названием{filen}")

    def load_set_from_file(self, filen):
        global unique_words
        # Чтение данных из файла
        with open(filen, "r", encoding='utf-16') as file:
            data = file.read()
        # Преобразование строки в множество
        data_set = set(data.split(","))
        unique_words = data_set



# Neural network model
class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyModel, self).__init__()

        self.fc1 = nn.Linear(in_channels, 2048)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 4096)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, 8192)
        self.act3 = nn.ReLU()
        self.fc8 = nn.Linear(8192, 4096)
        self.act8 = nn.ReLU()
        self.fc9= nn.Linear(4096, 2048)
        self.act9 = nn.ReLU()
        self.fc10= nn.Linear(2048, 1024)
        self.act10 = nn.ReLU()
        self.fc11 = nn.Linear(1024, 512)
        self.act11 = nn.ReLU()
        self.fc12 = nn.Linear(512, 256)
        self.act12 = nn.ReLU()
        self.fc13 = nn.Linear(256, out_channels)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc8(x)
        x = self.act8(x)
        x = self.fc9(x)
        x = self.act9(x)
        x = self.fc10(x)
        x = self.act10(x)
        x = self.fc11(x)
        x = self.act11(x)
        x = self.fc12(x)
        x = self.act12(x)
        x = self.fc13(x)
        return x


# Function to train the model
def train(model, data_loader, criterion, optimizer):
    global loss_train, train_acc, epochs, epoch
    model.train()
    for epoch in range(epochs):

        if br_p:
            break
        correct = 0
        total = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predicted = outputs.round()
            correct += (predicted == targets).sum().item()
            total += targets.numel()
        loss_train = loss.item()
        train_acc = correct / total
        loss_val, accuracy_val, hot_val_loss = validate(model, val_loader, criterion, epoch)
        print(f'Epoch:[{epoch + 1}/{epochs}], Train_Loss: {loss.item():.4f}, Train_Accuracy: {train_acc * 100:.2f}%, Total_val_Loss: {loss_val:.4f}, Accuracy_val: {accuracy_val * 100:.2f}%, Hot Val Loss: {hot_val_loss:.4f}')
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
            #Расчет метрики
            predicted = outputs.round()
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.numel()
    # Расчет метрики
    accuracy = total_correct / total_samples

    count += 1
    # Функция Остановки и hot сохранения  модели
    if best_loss == None:
        best_loss = total_loss / len(data_loader)
        print("Best Loss start")
    elif total_loss / len(data_loader) < best_loss and best_loss - total_loss / len(data_loader) >= 0.01:
        best_loss = total_loss / len(data_loader)
        count = 0
        save_model(model, 'model.pth', loss_train, total_loss / len(data_loader), best_loss, train_acc, accuracy, epoch)
        print(f'Модель сохранена Epoch: {epoch + 1}, Hot_Loss_Val: {loss.item():.4f}, Accuracy_val: {accuracy * 100:.2f}% Total_val_loss: {total_loss / len(data_loader):4f}')
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

            predicted = outputs.round()
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.numel()
            accuracy = total_correct / total_samples
    return total_loss / len(data_loader), accuracy

# Function to save the model
def save_model(model, path, train_loss, val_loss, best_loss, train_acc, val_acc, epoch):
    checpoint = {
        #'info': str_info,
        'state_model': model.state_dict(),
        'state_opt': optimizer.state_dict(),
        #'state_lr_sheduler': lr# на будущее
        'loss':{
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': best_loss,
        },
        'metric':{
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
    new_model = MyModel(1000, 9)
    load_model_state = torch.load(path_model, weights_only=False) # weights_only=True - загрузка только весов модели, для обучения надо сменить weights_only на False
    new_model.load_state_dict(load_model_state['state_model'])
    new_model.to(device)
    criterion_test = nn.MSELoss()
    test_loss_test, test_results_test = test(new_model, test_loader_test, criterion_test)
    print(f"Test LOAD Results{test_results_test * 100:.2f}%, Test Loss{test_loss_test:.4f}")
    return new_model

# Function to predict with the possibility to load csv file
def predict(model, excel_file, csv_file):
    convert_excel_to_utf8_csv(excel_file, csv_file)
    model.eval()
    if csv_file:
        datasetss = MyDataset(csv_file)
        data_loaderss = DataLoader(datasetss, batch_size=1, shuffle=False)
        answer_list = []
        with torch.no_grad():
            for inputs, _ in data_loaderss:
                inputs = inputs.to(device)
                outputs = model(inputs)
                maxi = -9999999999
                hot_ans = outputs[0].tolist()
                #print(hot_ans)
                ans = None
                for i in range(9):
                    if maxi < hot_ans[i]:
                        maxi = hot_ans[i]
                        ans = i
                #print(ans)
                answer_list.append(ans)
        return answer_list
    else:
        return "PROBLEM CSV FILE PREDICTION"


if __name__ == "__main__":
    if LearnTask:
        xlsx_file_train = 'Train_Real_data.xlsx'
        csv_file_train = 'Train_Real_data.csv'
        convert_excel_to_utf8_csv(xlsx_file_train, csv_file_train)
        global lr
        lr = 0.01
        epochs = 80
        dataset = MyDataset(csv_file_train)
       # xlsx_file_val = 'Data_New_train.xlsx.'
       # csv_file_val= 'Data_New_train.csv'
       # convert_excel_to_utf8_csv(xlsx_file_val, csv_file_val)
       # val_dataset = MyDataset(csv_file_val)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = MyModel(1000, 9).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        model = train(model, train_loader, criterion, optimizer)

        test_loss, test_results = test(model, test_loader, criterion)
        print(f"Test Results{test_results*100:.2f}%, Test Loss{test_loss:.4f}")
    else:
        convert_excel_to_utf8_csv('data_true_9.xlsx','data_true_9.csv')
        dataset_test = MyDataset('data_true_9.csv')
        #loaded_model = load_model('model.pth','data_true_10.xlsx','data_true_10.csv')
        #prediction = predict(loaded_model, 'data_true_10.xlsx', 'data_true_10.csv')
        #print("Prediction:", prediction)
