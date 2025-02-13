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
    time.sleep(3000)



def start_work_dobro(driver):
    driver.get(
        "https://26gosuslugi.ru/login?tab=performance&backUrl=%252Fpersoncab%252Finfo_pou%253Ftab%253Dperformance")

launch_browser()
