from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import os
import time
import requests
from bs4 import BeautifulSoup


def get_google_search_links(query, k):
    # Setup Chrome options to run headless (no GUI)
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Path to your ChromeDriver
    service = Service(executable_path=os.path.join(os.getcwd(), "chromedriver"))

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Navigate to Google search
    search_url = f"https://www.google.com/search?q={query}"
    driver.get(search_url)

    # Wait for the page to load
    time.sleep(3)

    # Find search result items on the page
    results = driver.find_elements(By.CSS_SELECTOR, 'div.g')
    links = []

    for result in results[:k]:  # Limit to the top k results
        try:
            link_element = result.find_element(By.CSS_SELECTOR, 'a')
            url = link_element.get_attribute('href')

            # Add URL to the list
            links.append(url)
        except Exception as e:
            print(f"An error occurred: {e}")

    driver.quit()

    return links





def scrape_url_content(url):
    try:
        # Send an HTTP request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the page content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get the title of the page
        title = soup.title.string if soup.title else 'No title'

        # Extract all meaningful text content from the page
        # We'll extract text from <p>, <h1>, <h2>, and other relevant tags
        content_tags = soup.find_all(['h1', 'h2', 'h3', 'p', 'li'])
        full_content = '\n'.join([tag.get_text() for tag in content_tags])

        # Return the title and the full content
        return {
            'url': url,
            'title': title,
            'full_content': full_content
        }

    except Exception as e:
        print(f"An error occurred while scraping {url}: {e}")
        return None


