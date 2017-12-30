import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome('/Users/faraaz/Downloads/chromedriver')

bach_url = "http://www.musicalion.com/en/scores/sheet-music/23/johann-sebastian-bach#metaarrangementid=66&arrangementgroupid=1&arrangementid=1277&ppage="
beethoven_url = "http://www.musicalion.com/en/scores/sheet-music/24/ludwig-van-beethoven#metaarrangementid=66&arrangementgroupid=1&arrangementid=1277&ppage="
mozart_url = "http://www.musicalion.com/en/scores/sheet-music/36/wolfgang-amadeus-mozart#metaarrangementid=66&arrangementgroupid=1&arrangementid=1277&ppage="
strauss_url = "http://www.musicalion.com/en/scores/sheet-music/853/johann-strauss#metaarrangementid=66&arrangementgroupid=1&arrangementid=1277&ppage="
test_url = "http://www.musicalion.com/en/scores/sheet-music/2179/antonio-caldara#metaarrangementid=66&arrangementgroupid=1&arrangementid=1277&ppage="

all_urls = [strauss_url]
isinit = True

for url in all_urls:
	driver.get(url)
	print("new composer")
	page_num = 1
	# loop over pages
	while True:
		try:
			driver.get(url+str(page_num))
			time.sleep(10)
		# next page does not exist, end of composer
		except WebDriverException:
			print("reached composer end")
			# TODO: messenger notification
			break
		cur_url = driver.current_url
		print("page", page_num)
		page_num += 1
		# optional initialization from last failed state
		if isinit:
			i = 0
		else:
			i = 0
		isinit = False
		# loop over page items
		while i < 10:
			time.sleep(3)
			print("getting midi", i)
			# wait for page item to load
			try:
				table_row = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "p__ptr_"+str(i))))
				link = table_row.find_elements_by_class_name("cm")[0]
				composition_url = link.get_attribute("href")
				driver.get(composition_url)
				# get the midi file
				try:
					midi_link = driver.find_element_by_id("cpButtonPlay")
					midi_url = midi_link.get_attribute("href")
					driver.get(midi_url)
				# midi file does not exist
				except WebDriverException:
					print("no midi", i)
			# item does not exist, end of composer
			except TimeoutException:
				print("NOT FOUND: page", page_num, "item", i)
			time.sleep(2)
			# go back to page items
			driver.get(cur_url)
			i += 1
		try:
			link = driver.find_element_by_link_text('>')
		# next page does not exist, end of composer
		except NoSuchElementException:
			print("reached composer end")
			# TODO: messenger notification
			break

driver.quit()
