import json
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service as ChromeService
import torch
import torch.nn as nn
import torch.optim as optim
from fontTools.misc.timeTools import epoch_diff
from openpyxl.styles.builtins import total
from torch.distributed.checkpoint import load_state_dict
from torch.utils.data import DataLoader, Dataset, random_split
import nltk
import openpyxl
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import xlsxwriter

page_links,url_next,rows, chet = 0,0,0, 0

data_revers = str # временная строка для добрых дел
all_data = [] # сбор всех дел в одну переменную
counter = 0 #количиство собранных добрых дел
numL = 0 #переключение страниц
answer = 0 #ответ ии
driver = 0 #данные браузера
quantity = 0 #количество страниц обработки
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Доступ:{torch.cuda.is_available()}, факт:{device} Верси куда: {torch.version.cuda}")

def convert_excel_to_utf8_csv(excel_file, csv_file):
    # Чтение Excel файла
    df = pd.read_excel(excel_file)

    # Сохранение в CSV файл с кодировкой UTF-8
    df.to_csv(csv_file, index=False, encoding='utf-16')

class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, encoding='utf-16')
        self.stop_words = set(stopwords.words('russian'))
        self.vectorizer = CountVectorizer(max_features=8000)  # Adjust the number of features as needed
        self.vectorizer.fit(self.data['Description'].apply(self.preprocess_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        description = self.data.iloc[idx, 0]
        target = self.data.iloc[idx, 1]
        description = self.preprocess_text(description)
        description_vector = self.vectorizer.transform([description]).toarray()
        description_tensor = torch.tensor(description_vector, dtype=torch.float32).squeeze()
        target_tensor = torch.tensor([target] * 9, dtype=torch.float32)  # Adjust target size to match output
        return description_tensor, target_tensor

    def preprocess_text(self, text):
        words = word_tokenize(text.lower(), language='russian')
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        return ' '.join(words)

class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(in_channels, 2048)  # Adjust input size as necessary
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
        self.fc13 = nn.Linear(256, 128)
        self.act13 = nn.ReLU()
        self.fc14 = nn.Linear(128, out_channels)


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
        x = self.act13(x)
        x = self.fc14(x)
        return x

# Function to load the model
def load_model(path):
    new_model = MyModel(8000, 9)
    load_model_state = torch.load(path, weights_only=False) # weights_only=True - загрузка только весов модели, для обучения надо сменить weights_only на False
    new_model.load_state_dict(load_model_state['state_model'])
    new_model.to(device)
    return new_model

# Function to predict with the possibility to load csv file
def predict(model, excel_file, csv_file):
    convert_excel_to_utf8_csv(excel_file, csv_file)
    model.eval()
    if csvs_file:
        datasetss = MyDataset(csv_file)
        data_loaderss = DataLoader(datasetss, batch_size=1, shuffle=False)
        answer_list = []
        with torch.no_grad():
            for inputs, _ in data_loaderss:
                inputs = inputs.to(device)
                outputs = model(inputs)
                maxi = 0
                hot_ans = outputs[0].tolist()
                #print(hot_ans)
                ans = None
                for i in range(10):
                    if maxi < hot_ans[i]:
                        maxi = hot_ans[i]
                        ans = i
                #print(ans)
                answer_list.append(ans)
        return answer_list
    else:
        return "PROBLEM CSV FILE PREDICTION"



def launch_browser():

    capabilities = DesiredCapabilities.CHROME
    capabilities["goog:loggingPrefs"] = {"performance": "ALL", 'browser': 'ALL'}

    options = webdriver.ChromeOptions()
    service = ChromeService(executable_path=".\\chromium\\bin\\chromedriver.exe")
    options.add_argument(f"--user-data-dir=.\\chromium\\profile")
    options.add_argument(f"--profile-directory=Profile 1")
    options.add_argument(f"--remote-debugging-port=9222")
    #options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument('log-level=3')
    options.add_experimental_option(
        'excludeSwitches',
        ['password-store',
         'use-mock-keychain'],)
    global numL, chet, counter, driver
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(2)
    start_work_dobro(driver)


def take_inf():
    global numL, chet, counter, driver, quantity
    quantity = int(input("Кол-во страниц * 10 = кол-во мероприятий но не меньше 4. 4 > вашего значения - Все мероприятия"))
    if quantity < 4:
        list_pages = driver.find_element(By.CLASS_NAME, "col-6").find_elements(By.TAG_NAME, "a")
        quantity = int(list_pages[len(list_pages) - 2].get_attribute('text'))
    for i in range(quantity):
        # Находим все строки таблицы
        find_next(driver)

        for row in rows:
            event_link_finder(driver, row)
        numL = 1
        open_link_in_new_tab(driver, url_next)
        driver.switch_to.window(driver.window_handles[-1])
        if i >= 1:
            driver.switch_to.window(driver.window_handles[1])
            driver.close()
            driver.switch_to.window(driver.window_handles[-1])
    return driver


def finals():
    global counter
    counter = 0
    data_set_resulte = data_set()
    pd.DataFrame(data_set_resulte).to_excel(f'data_work.xlsx')
    file_path = 'data_work.xlsx'
    df = pd.read_excel(file_path)

    # Удаление первого столбца
    df.drop(df.columns[0], axis=1, inplace=True)

    # Сохранение изменений в новый Excel файл
    df.to_excel(file_path, index=False)
    print(f"Файл сохранен")


def accept(driver):
    confirm_link = row.find_element(By.CLASS_NAME, "js-confirm-modal")
    confirm_link.click()
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "modal-content")))
    div_dialog = driver.find_elements(By.CLASS_NAME, "modal-content")
    if len(div_dialog) == 1:
        ok_link = div_dialog[0].find_element(By.TAG_NAME, "a")
        ok_link.click()


def decline(driver):
    global row, answer
    links = row.find_elements(By.CLASS_NAME, "js-confirm-modal")
    links[1].click()
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "modal-content")))
    div_dialog = driver.find_elements(By.CLASS_NAME, "modal-content")
    if len(div_dialog) == 1:
        options = div_dialog[0].find_elements(By.TAG_NAME, "option")
        accept_btn = div_dialog[0].find_elements(By.TAG_NAME, "button")[-1]
        if answer == 1:
            options[1].click()
        elif answer >= 2:
            options[answer + 1].click()
        accept_btn.click()


def open_link_in_new_tab(driver, url):
    # Открываем ссылку в новой вкладке
    driver.execute_script("window.open('', '_blank');")
    driver.switch_to.window(driver.window_handles[-1])
    driver.get(url)


def start_work_dobro(driver):
    #Обычная driver.get("https://dobro.ru/admin/abstract/moderation-tasks/?moderation_task_metadata_filter_admin%5Bsearch%5D=&moderation_task_metadata_filter_admin%5Bstatuses%5D%5B0%5D=200&moderation_task_metadata_filter_admin%5Btype%5D=event&moderation_task_metadata_filter_admin%5BmoderatorId%5D=13530330&moderation_task_metadata_filter_admin%5Bmoderator%5D=&moderation_task_metadata_filter_admin%5BpostModeration%5D=&moderation_task_metadata_filter_admin%5BcompletedAtFrom%5D=&moderation_task_metadata_filter_admin%5BcompletedAtTo%5D=&sort=a.id&direction=asc&page=1")
    #Отклон driver.get("https://dobro.ru/admin/abstract/moderation-tasks/?moderation_task_metadata_filter_admin%5Bsearch%5D=&moderation_task_metadata_filter_admin%5Bstatuses%5D%5B0%5D=300&moderation_task_metadata_filter_admin%5Btype%5D=event&moderation_task_metadata_filter_admin%5BmoderatorId%5D=&moderation_task_metadata_filter_admin%5Bmoderator%5D=&moderation_task_metadata_filter_admin%5BpostModeration%5D=&moderation_task_metadata_filter_admin%5BcompletedAtFrom%5D=&moderation_task_metadata_filter_admin%5BcompletedAtTo%5D=&sort=a.createdAt&direction=desc&page=1")
    #driver.get("https://dobro.ru/admin/abstract/moderation-tasks/?moderation_task_metadata_filter_admin%5Bsearch%5D=&moderation_task_metadata_filter_admin%5Bstatuses%5D%5B0%5D=400&moderation_task_metadata_filter_admin%5Btype%5D=event&moderation_task_metadata_filter_admin%5BmoderatorId%5D=&moderation_task_metadata_filter_admin%5Bmoderator%5D=&moderation_task_metadata_filter_admin%5BpostModeration%5D=&moderation_task_metadata_filter_admin%5BcompletedAtFrom%5D=&moderation_task_metadata_filter_admin%5BcompletedAtTo%5D=&page=2&sort=a.createdAt&direction=desc")
    driver.get("https://dobro.ru/admin/abstract/moderation-tasks/?moderation_task_metadata_filter_admin%5Bsearch%5D=&moderation_task_metadata_filter_admin%5Bstatuses%5D%5B%5D=400&moderation_task_metadata_filter_admin%5Btype%5D=event&moderation_task_metadata_filter_admin%5BmoderatorId%5D=&moderation_task_metadata_filter_admin%5Bmoderator%5D=&moderation_task_metadata_filter_admin%5BpostModeration%5D=&moderation_task_metadata_filter_admin%5BcompletedAtFrom%5D=&moderation_task_metadata_filter_admin%5BcompletedAtTo%5D=&sort=a.createdAt&direction=desc&page=1")


def extract_info(driver):
    def extract_json(elem):
        # Получаем текст из элемента
        json_text = elem.get_attribute('innerHTML')

        # Преобразуем текст в словарь
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            print(f"Ошибка при разборе JSON: {json_text}")
    # Находим элементы с тегом <script> и типом "application/ld+json"
    result = [extract_json(driver.find_element(By.XPATH, "//script[@type='application/ld+json']"))]
    result.append(extract_json(driver.find_element(By.XPATH, "//script[@id='__NEXT_DATA__'][@type='application/json']")))

    return result


def data_set():
    global all_data
    data_list = {
        'Description': all_data,
        'Targets': str(0)
    }
    all_data = []
    return data_list


def find_next(driver):
    global chet
    try:
        global page_links,url_next,rows
        WebDriverWait(driver, 4).until(EC.presence_of_element_located((By.LINK_TEXT, "Вперёд")))
        page_links = driver.find_element(By.LINK_TEXT, "Вперёд")
        url_next = page_links.get_attribute('href')
        rows = driver.find_elements(By.XPATH, "//table[@class='table table-bordered']/tbody/tr")
        chet = 0
    except:
        chet += 1
        if chet <= 2:
            driver.refresh()
            time.sleep(0.5)
            find_next(driver)
        else:
            print("Ошибка поиска LINK_TEXT, Вперёд")
            return driver


def event_link_finder(driver, row):
    global chet,quantity
    #try:
    time.sleep(1)
    WebDriverWait(driver, 4).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
    event_link = row.find_elements(By.TAG_NAME, "a")
    event_link_url = event_link[0].get_attribute('href')
    author_link = event_link[1].get_attribute('text')
    chet = 0
    open_link_in_new_tab(driver, event_link_url)

    open_link(driver, event_link_url, author_link)
    if counter >= quantity*10:
            finals()

    """except:
        chet += 1
        if chet <= 2:
            driver.refresh()
            time.sleep(0.5)
            event_link_finder(driver, row)
        else:
            print("Ошибка поиска TAG_NAME, а")
            return driver
            """


def open_link(driver, event_link_url, author_link):

    global counter, data_revers, all_data, take, numL, quantity, chet, answer
    try:
        json_data = extract_info(driver)
        name = json_data[0][0]['name']
        description = json_data[0][0]['description']
        event_info = json_data[1]['props']['pageProps']['initialState']['eventReducer']
        shortTitle = event_info['event']['eventPeriod']['shortTitle']
        vacanct_info = event_info['eventVacancies'][0]
        tasks_text = '; '.join(task['title'] for task in vacanct_info['tasks'])
        requirements_text = '; '.join(requirement['title'] for requirement in vacanct_info['requirements'])

        counter += 1
        chet = 0

        data_revers = f"{name}" + f" {description}" + f" {shortTitle}" +f" {tasks_text}" + f" {requirements_text}"

        all_data.append(data_revers)
        data_revers = str()

        """print('name:' + name)
        print('description:' + description)
        print('shortTitle:' + shortTitle)
        print('tasks_text:' + tasks_text)
        print('requirements_text:' + requirements_text)
        time.sleep(0.5)
        links = row.find_elements(By.CLASS_NAME, "js-confirm-modal")
        confirm_link = links[0]
        decline_link = links[1]
        print(confirm_link.get_attribute('title'), confirm_link.get_attribute('href'))
        print(decline_link.get_attribute('title'), decline_link.get_attribute('href'), "\n")"""
        time.sleep(0.5)
        driver.close()
        driver.switch_to.window(driver.window_handles[numL])
    except:
        chet += 1
        if chet <= 2:
            driver.refresh()
            time.sleep(0.5)
            open_link(driver, event_link_url, author_link)
        else:
            time.sleep(0.5)
            driver.close()
            driver.switch_to.window(driver.window_handles[numL])
            answer = 2
            decline(driver, answer)


def processing_data_ai():
    global loaded_model, driver
    print("Start AI")
    prediction = predict(loaded_model, 'data_work.xlsx', "Work.csv")
    print("Prediction:", prediction)
    global answer, driver, row, rows
    i = 0
    for y in range(quantity):
        find_next(driver)
        for row in rows:
            answer = prediction[i]
            i+=1
            if answer == 0:
                accept(driver)
            else:
                decline(driver, answer, row)
        driver.refresh()

#loaded_model = load_model('model1.pth')
processing_data_ai()
#launch_browser()
#take_inf()
#launch_browser()
#processing_data_ai()