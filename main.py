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

page_links,url_next,rows, chet = 0,0,0, 0
quantity = int(input("Кол-во страниц * 10 = кол-во мероприятий"))
data_name = []
data_description = []
data_shortTitle = []
data_tasks_text = []
data_requirements_text = []
data_status = []
author_name = []
counter = 0
number = 0
numL = 0
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
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(2)
    global numL, chet, counter
    start_work_dobro(driver)


    # Теперь вы можете работать с каждой строкой отдельно range(len(page_links-2))
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
    global number, counter
    number += 1
    counter = 0
    data_set_resulte = data_set()
    pd.DataFrame(data_set_resulte).to_excel(f'data_true_{str(number)}.xlsx')
    print(f"Колво файлов {number}")

def decline(driver,decline_link):
    decline_link.click()
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "modal-content")))
    div_dialog = driver.find_elements(By.CLASS_NAME, "modal-content")
    if len(div_dialog) == 1:
        options = div_dialog[0].find_elements(By.TAG_NAME, "option")
        filtered_options = [option for option in options if len(option.get_attribute('value')) > 0]
        accept_btn = div_dialog[0].find_elements(By.TAG_NAME, "button")[-1]
        for option in filtered_options:
            print(option.text)

def open_link_in_new_tab(driver, url):
    # Открываем ссылку в новой вкладке
    driver.execute_script("window.open('', '_blank');")
    driver.switch_to.window(driver.window_handles[-1])
    driver.get(url)


def start_work_dobro(driver):
    #Обычная driver.get("https://dobro.ru/admin/abstract/moderation-tasks/?moderation_task_metadata_filter_admin%5Bsearch%5D=&moderation_task_metadata_filter_admin%5Bstatuses%5D%5B0%5D=200&moderation_task_metadata_filter_admin%5Btype%5D=event&moderation_task_metadata_filter_admin%5BmoderatorId%5D=13530330&moderation_task_metadata_filter_admin%5Bmoderator%5D=&moderation_task_metadata_filter_admin%5BpostModeration%5D=&moderation_task_metadata_filter_admin%5BcompletedAtFrom%5D=&moderation_task_metadata_filter_admin%5BcompletedAtTo%5D=&sort=a.id&direction=asc&page=1")
    #Отклон driver.get("https://dobro.ru/admin/abstract/moderation-tasks/?moderation_task_metadata_filter_admin%5Bsearch%5D=&moderation_task_metadata_filter_admin%5Bstatuses%5D%5B0%5D=300&moderation_task_metadata_filter_admin%5Btype%5D=event&moderation_task_metadata_filter_admin%5BmoderatorId%5D=&moderation_task_metadata_filter_admin%5Bmoderator%5D=&moderation_task_metadata_filter_admin%5BpostModeration%5D=&moderation_task_metadata_filter_admin%5BcompletedAtFrom%5D=&moderation_task_metadata_filter_admin%5BcompletedAtTo%5D=&sort=a.createdAt&direction=desc&page=1")
    #driver.get("https://dobro.ru/admin/abstract/moderation-tasks/?moderation_task_metadata_filter_admin%5Bsearch%5D=&moderation_task_metadata_filter_admin%5Bstatuses%5D%5B0%5D=400&moderation_task_metadata_filter_admin%5Btype%5D=event&moderation_task_metadata_filter_admin%5BmoderatorId%5D=&moderation_task_metadata_filter_admin%5Bmoderator%5D=&moderation_task_metadata_filter_admin%5BpostModeration%5D=&moderation_task_metadata_filter_admin%5BcompletedAtFrom%5D=&moderation_task_metadata_filter_admin%5BcompletedAtTo%5D=&page=2&sort=a.createdAt&direction=desc")
    driver.get("https://dobro.ru/admin/abstract/moderation-tasks/?moderation_task_metadata_filter_admin%5Bsearch%5D=&moderation_task_metadata_filter_admin%5Bstatuses%5D%5B%5D=400&moderation_task_metadata_filter_admin%5Btype%5D=event&moderation_task_metadata_filter_admin%5BmoderatorId%5D=&moderation_task_metadata_filter_admin%5Bmoderator%5D=&moderation_task_metadata_filter_admin%5BpostModeration%5D=&moderation_task_metadata_filter_admin%5BcompletedAtFrom%5D=&moderation_task_metadata_filter_admin%5BcompletedAtTo%5D=&sort=a.createdAt&direction=desc&page=13000")

def accept(driver,confirm_link):
    confirm_link.click()
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "modal-content")))
    div_dialog = driver.find_elements(By.CLASS_NAME, "modal-content")
    print(len(div_dialog))
    if len(div_dialog) == 1:
        ok_link = div_dialog[0].find_element(By.TAG_NAME, "a")
        print(ok_link.get_attribute('href'))
        ok_link.click()

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
    global data_name, data_description,data_shortTitle, data_tasks_text,data_requirements_text,data_status,author_name
    data_list = {
        'name': data_name,
        'description': data_description,
        'shortTitle': data_shortTitle,
        'tasks_text': data_tasks_text,
        'requirements_text': data_requirements_text,
        'status': data_status,
        'author_name': author_name
     }
    data_name, data_description, data_shortTitle,data_tasks_text,data_requirements_text,data_status,author_name = [],[],[],[],[],[],[]
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
    global chet
    try:

        time.sleep(1)
        WebDriverWait(driver, 4).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
        event_link = row.find_elements(By.TAG_NAME, "a")
        event_link_url = event_link[0].get_attribute('href')
        author_link = event_link[1].get_attribute('text')
        chet = 0
        open_link_in_new_tab(driver, event_link_url)

        open_link(driver, event_link_url, author_link)
        if counter >= 2000:
            finals()

    except:
        chet += 1
        if chet <= 2:
            driver.refresh()
            time.sleep(0.5)
            event_link_finder(driver, row)
        else:
            print("Ошибка поиска TAG_NAME, а")
            return driver

def open_link(driver, event_link_url, author_link):
    global chet, numL, counter
    try:
        json_data = extract_info(driver)
        name = json_data[0][0]['name']
        description = json_data[0][0]['description']
        event_info = json_data[1]['props']['pageProps']['initialState']['eventReducer']
        shortTitle = event_info['event']['eventPeriod']['shortTitle']
        vacanct_info = event_info['eventVacancies'][0]
        tasks_text = '; '.join(task['title'] for task in vacanct_info['tasks'])
        requirements_text = '; '.join(requirement['title'] for requirement in vacanct_info['requirements'])
        global data_name, data_description, data_shortTitle, data_tasks_text, data_requirements_text, data_status, numL, author_name
        counter += 1
        chet = 0
        data_name.append(name)
        data_description.append(description)
        data_shortTitle.append(shortTitle)
        data_tasks_text.append(tasks_text)
        data_requirements_text.append(requirements_text)
        data_status.append("True")
        author_name.append(author_link)
        print(counter)

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
launch_browser()

