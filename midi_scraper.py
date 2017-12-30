import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import WebDriverException

driver = webdriver.Chrome('/Users/faraaz/Downloads/chromedriver')

bach_url = "http://www.musicalion.com/en/scores/sheet-music/23/johann-sebastian-bach"
beethoven_url = "http://www.musicalion.com/en/scores/sheet-music/24/ludwig-van-beethoven"
mozart_url = "http://www.musicalion.com/en/scores/sheet-music/36/wolfgang-amadeus-mozart"

all_urls = [beethoven_url]

for url in all_urls:
	driver.get(url)
	print("new composer")
	solo_inst_link = driver.find_element_by_id("f__searchCategories__metaarrangementid_option_66")
	ActionChains(driver).move_to_element(solo_inst_link).click().perform()
	time.sleep(5)
	keyboard_inst_link = driver.find_element_by_id("f__searchCategories__arrangementid_arrangementGroup_1_expander")
	ActionChains(driver).move_to_element(keyboard_inst_link).click().perform()
	time.sleep(5)
	piano_link = driver.find_element_by_id("f__searchCategories__arrangementid_option_1277")
	ActionChains(driver).move_to_element(piano_link).click().perform()
	time.sleep(5)
	page_num = 1
	while True:
		cur_url = driver.current_url
		print("page", page_num)
		page_num += 1
		i = 0
		while True:
			time.sleep(3)
			print("getting midi", i)
			try:
				table_row = driver.find_element_by_id("p__ptr_"+str(i))
			except NoSuchElementException:
				break
			link = table_row.find_elements_by_class_name("cm")[0]
			composition_url = link.get_attribute("href")
			driver.get(composition_url)
			try:
				midi_link = driver.find_element_by_id("cpButtonPlay")
				midi_url = midi_link.get_attribute("href")
				driver.get(midi_url)
			except WebDriverException:
				print("no midi", i)
			time.sleep(2)
			driver.get(cur_url)
			i += 1
		try:
			link = driver.find_element_by_link_text(">")
			ActionChains(driver).move_to_element(link).click().perform()
		except NoSuchElementException:
			break

driver.quit()
