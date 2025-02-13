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
from selenium.common.exceptions import NoSuchElementException
import keyboard
from apscheduler.schedulers.blocking import BlockingScheduler
import pyautogui

numL = int(0)
confirm_link, decline_link, rows = 0, 0, 0,

def launch_browser():
    capabilities = DesiredCapabilities.CHROME
    capabilities["goog:loggingPrefs"] = {"performance": "ALL", 'browser': 'ALL'}
    options = webdriver.ChromeOptions()
    service = ChromeService(executable_path=".\\chromium\\bin\\chromedriver.exe")
    options.add_argument(f"--user-data-dir=.\\chromium\\profile")
    options.add_argument(f"--profile-directory=Profile 1")
    options.add_argument(f"--remote-debugging-port=9222")
    # options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument('log-level=3')
    options.add_experimental_option(
        'excludeSwitches',
        ['password-store',
         'use-mock-keychain'], )
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(2)
    start_work_dobro(driver)
    #time.sleep(1000)
    if int(input("Выбирать авто? - нет -0 да - 1")) == 0:
        quantity = int(input("Сколько взять *10"))
    elif check_exists_by_class(driver):
        quantity = driver.find_elements(By.CLASS_NAME, "page-item")
        quantity = int(quantity[len(quantity)-2].text)
    else:
        quantity = 1
    for i in range(quantity):
       # Находим все строки таблицы
            rows = driver.find_elements(By.XPATH, "//table[@class='table table-bordered']/tbody/tr")
            for row in rows:
                event_link = row.find_element(By.CLASS_NAME, "js-confirm-modal")
                event_link.click()
                time.sleep(0.5)
            driver.get(driver.current_url)
            time.sleep(2)
            driver.refresh()
    # Теперь вы можете работать с каждой строкой отдельно range(len(page_links-2))

    return driver

def open_link_in_new_tab(driver, url):
    # Открываем ссылку в новой вкладке
    driver.execute_script("window.open('', '_blank');")
    driver.switch_to.window(driver.window_handles[-1])
    driver.get(url)

def check_exists_by_class(driver):
    try:
        WebDriverWait(driver, 2).until(EC.presence_of_element_located((By.CLASS_NAME, "page-item")))
    except TimeoutException:
        return False
    return True

def start_work_dobro(driver):

    driver.get("https://dobro.ru/admin/abstract/moderation-tasks/?moderation_task_metadata_filter_admin%5Bsearch%5D=&moderation_task_metadata_filter_admin%5Bstatuses%5D%5B%5D=100&moderation_task_metadata_filter_admin%5Btype%5D=event&moderation_task_metadata_filter_admin%5BmoderatorId%5D=&moderation_task_metadata_filter_admin%5Bmoderator%5D=&moderation_task_metadata_filter_admin%5BpostModeration%5D=&moderation_task_metadata_filter_admin%5BcompletedAtFrom%5D=&moderation_task_metadata_filter_admin%5BcompletedAtTo%5D=")
    #driver.get("https://dobro.ru/admin/abstract/moderation-tasks/?moderation_task_metadata_filter_admin%5Bsearch%5D=&moderation_task_metadata_filter_admin%5Bstatuses%5D%5B0%5D=300&moderation_task_metadata_filter_admin%5Btype%5D=event&moderation_task_metadata_filter_admin%5BmoderatorId%5D=&moderation_task_metadata_filter_admin%5Bmoderator%5D=&moderation_task_metadata_filter_admin%5BpostModeration%5D=&moderation_task_metadata_filter_admin%5BcompletedAtFrom%5D=&moderation_task_metadata_filter_admin%5BcompletedAtTo%5D=&sort=a.createdAt&direction=desc&page=1")


for i in range(24):
    launch_browser()
    time.sleep(1800)

"""
def scheduler_start():
    scheduler = BlockingScheduler()
    scheduler.add_job(launch_browser(), 'interval', hours=1)
    scheduler.start()

scheduler_start()"""

