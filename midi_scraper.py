import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import UnexpectedAlertPresentException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome('/Users/faraaz/Downloads/chromedriver')

bach_url = "http://www.musicalion.com/en/scores/sheet-music/23/johann-sebastian-bach#metaarrangementid=66&arrangementgroupid=1&arrangementid=1277&ppage="
beethoven_url = "http://www.musicalion.com/en/scores/sheet-music/24/ludwig-van-beethoven#metaarrangementid=66&arrangementgroupid=1&arrangementid=1277&ppage="
mozart_url = "http://www.musicalion.com/en/scores/sheet-music/36/wolfgang-amadeus-mozart#metaarrangementid=66&arrangementgroupid=1&arrangementid=1277&ppage="
strauss_url = "http://www.musicalion.com/en/scores/notes/composition/search-result?ma=66&ag=1&a=1277&cn=853&skip_composer_info=1&ppage=19"
test_url = "http://www.musicalion.com/en/scores/sheet-music/2179/antonio-caldara#metaarrangementid=66&arrangementgroupid=1&arrangementid=1277&ppage="


all_urls = [strauss_url]
isinit = True

for url in all_urls:
	driver.get(url)
	print("new composer")
	page_num = 19
	# loop over pages
	while True:
		print("page", page_num)
		page_num += 1
		# wait for page item to load
		while True:
			try:
				link_elements = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.cm")))
				break
			except UnexpectedAlertPresentException:
				driver.switch_to.alert.accept()
				print("closed alert")
		links = [link_element.get_attribute('href') for link_element in link_elements]
		print(len(links), "links")
		for i, link in enumerate(links):
			print("getting midi", i)
			while True:
				try:
					driver.get(link)
					break
				except UnexpectedAlertPresentException:
					driver.switch_to.alert.accept()
					print("closed alert")
			# get the midi file
			try:
				midi_link = driver.find_element_by_id("cpButtonPlay").get_attribute('href')
				driver.get(midi_link)
			# midi file does not exist
			except WebDriverException:
				print("no midi", i)
			driver.execute_script("window.history.go(-1)")
		complete = False
		while True:
			try:
				next_page_link = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.LINK_TEXT, ">"))).get_attribute('href')
				driver.get(next_page_link)
				break
			except UnexpectedAlertPresentException:
				driver.switch_to.alert.accept()
				print("closed alert")
			except TimeoutException:
				print("download complete")
				complete = True
				break
		if complete:
			break
driver.quit()
