import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import nltk
import openpyxl
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import xlsxwriter


def convert_excel_to_utf8_csv(excel_file, csv_file):
    # Чтение Excel файла
    df = pd.read_excel(excel_file)

    # Сохранение в CSV файл с кодировкой UTF-8
    df.to_csv(csv_file, index=False, encoding='utf-16')

# Пример использования
excel_file = 'answet.xlsx'

#def convert_xlsx_to_csv(xlsx_file, csv_file):
#    # Read the Excel file
#    df = pd.read_excel(xlsx_file)
#
#    # Save it to a CSV file
#    df.to_csv(csv_file, index=False)
#
# Example usage
#xlsx_file = 'answet.xlsx'
csv_file = 'True-Test-answet.csv'
convert_excel_to_utf8_csv(excel_file, csv_file)

#print(f'File {xlsx_file} has been successfully converted to {csv_file}')



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Dataset class
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.stop_words = set(stopwords.words('russian'))
        self.vectorizer = CountVectorizer(max_features=1000)  # Adjust the number of features as needed
        self.vectorizer.fit(self.data['Description'].apply(self.preprocess_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        description = self.data.iloc[idx, 0]
        target = self.data.iloc[idx, 1]
        description = self.preprocess_text(description)
        description_vector = self.vectorizer.transform([description]).toarray()
        description_tensor = torch.tensor(description_vector, dtype=torch.float32).squeeze()
        target_tensor = torch.tensor([target] * 10, dtype=torch.float32)  # Adjust target size to match output
        return description_tensor, target_tensor

    def preprocess_text(self, text):
        words = word_tokenize(text.lower(), language='russian')
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        return ' '.join(words)

# Neural network model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(1000, 2048)  # Adjust input size as necessary
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 4096)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, 2048)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(2048, 1024)
        self.act4 = nn.ReLU()
        self.fc5 = nn.Linear(1024, 512)
        self.act5 = nn.ReLU()
        self.fc6 = nn.Linear(512, 256)
        self.act6 = nn.ReLU()
        self.fc7 = nn.Linear(256, 128)
        self.act7 = nn.ReLU()
        self.fc8 = nn.Linear(128, 10)
        self.act8 = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        x = self.act4(x)
        x = self.fc5(x)
        x = self.act5(x)
        x = self.fc6(x)
        x = self.act6(x)
        x = self.fc7(x)
        x = self.act7(x)
        x = self.fc8(x)
        x = self.act8(x)
        return x

# Function to train the model
def train(model, data_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
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

        accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%')
    return model

# Function to validate the model
def validate(model, data_loader, criterion):
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

# Function to test the model
def test(model, data_loader):
    model.eval()
    results = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            results.append(outputs)
    return torch.cat(results)

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Function to load the model
def load_model(path):
    model = MyModel()
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model

# Function to predict with the possibility to load csv file
def predict(model, inputs=None, csv_file=None):
    model.eval()
    if csv_file:
        dataset = MyDataset(csv_file)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        results = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                results.append(outputs)
        return torch.cat(results)
    else:
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)
        return outputs

# Example usage
if __name__ == "__main__":
    dataset = MyDataset('True-Test-answet.csv')

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MyModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    model = train(model, train_loader, criterion, optimizer, epochs=10)
    val_loss, val_accuracy = validate(model, val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

    test_results = test(model, test_loader)
    print("Test Results:", test_results)

    save_model(model, 'model.pth')
#    loaded_model = load_model('model.pth')

#    sample_input = torch.tensor([0.5] * 100).to(device)  # Adjust input size as necessary
#    prediction = predict(loaded_model, inputs=sample_input)
#    print("Prediction:", prediction)