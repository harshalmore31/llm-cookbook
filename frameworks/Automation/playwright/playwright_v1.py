from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    # Launch browser
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    # Open ChatGPT
    page.goto("https://chatgpt.com")
    time.sleep(7)

    # Type "hii" in the search box and press Enter
    page.fill("input[name='q']", "hii")
    page.press("input[name='q']", "Enter")

    # Wait for results
    page.wait_for_selector("#search")
    print("Search results loaded!")

    # Close browser
    browser.close()
