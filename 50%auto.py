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
import random
numL = int(0)
confirm_link, decline_link, rows, a,b = 0, 0, 0, int(input("Рандомный режим? 1-да ")), 1000
if a == 1:
    b = int(input("Сколько модерить?"))

def launch_browser():

    def on_triggered1():
        accept(driver)
    def on_triggered2():
        decline(driver)
    def on_triggered3():
        global confirm_link, decline_link, rows
        rows = driver.find_elements(By.XPATH, "//table[@class='table table-bordered']/tbody/tr")

        event_link = rows[0].find_element(By.TAG_NAME, "a")
        event_link_url = event_link.get_attribute('href')
        open_link_in_new_tab(driver, event_link_url)

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
    # Теперь вы можете работать с каждой строкой отдельно range(len(page_links-2))

    on_triggered3()
    for i in range(b):
        # Находим все строки таблицы
        print(i)
        time.sleep(1)
        if a == 1:
            if random.randint(0,100) < 3:
                on_triggered2()
                time.sleep(0.5)
                on_triggered3()
            else:
                on_triggered1()
                time.sleep(1)
                on_triggered3()
        else:
            keyboard.add_hotkey('alt+3', on_triggered3)
            keyboard.add_hotkey('alt+2', on_triggered2)
            keyboard.add_hotkey('alt+1', on_triggered1)


            keyboard.wait('alt+5')

    return driver


def decline(driver):
    driver.switch_to.window(driver.window_handles[0])
    decline_link = driver.find_elements(By.CLASS_NAME, "js-confirm-modal")
    decline_link[1].click()
    driver.switch_to.window(driver.window_handles[-1])
    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    if a == 1:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "modal-content")))
        div_dialog = driver.find_elements(By.CLASS_NAME, "modal-content")
        div_dialog[0].find_element(By.ID, "moderation_reject_form_reason").click()
        time.sleep(0.5)
        options = div_dialog[0].find_elements(By.TAG_NAME, "option")
        filtered_options = [option for option in options if len(option.get_attribute('value')) > 0]
        filtered_options[2].click()
        time.sleep(0.5)
        links_yes = div_dialog[0].find_elements(By.TAG_NAME, "button")[1]
        links_yes.click()




def open_link_in_new_tab(driver, url):
    # Открываем ссылку в новой вкладке
    driver.execute_script("window.open('', '_blank');")
    driver.switch_to.window(driver.window_handles[-1])
    driver.get(url)


def start_work_dobro(driver):
      driver.get("https://dobro.ru/admin/abstract/moderation-tasks/?moderation_task_metadata_filter_admin%5Bsearch%5D=&moderation_task_metadata_filter_admin%5Bstatuses%5D%5B0%5D=200&moderation_task_metadata_filter_admin%5Btype%5D=event&moderation_task_metadata_filter_admin%5BmoderatorId%5D=13530330&moderation_task_metadata_filter_admin%5Bmoderator%5D=&moderation_task_metadata_filter_admin%5BpostModeration%5D=&moderation_task_metadata_filter_admin%5BcompletedAtFrom%5D=&moderation_task_metadata_filter_admin%5BcompletedAtTo%5D=&sort=a.id&direction=asc&page=1")


def accept(driver):
    driver.switch_to.window(driver.window_handles[0])
    confirm_link = driver.find_element(By.CLASS_NAME, "js-confirm-modal")
    confirm_link.click()
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "modal-content")))
    div_dialog = driver.find_elements(By.CLASS_NAME, "modal-content")
    if len(div_dialog) == 1:
        ok_link = div_dialog[0].find_element(By.TAG_NAME, "a")
        ok_link.click()
    driver.switch_to.window(driver.window_handles[-1])
    driver.close()
    driver.switch_to.window(driver.window_handles[0])


launch_browser()

