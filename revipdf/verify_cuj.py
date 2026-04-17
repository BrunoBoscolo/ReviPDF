from playwright.sync_api import sync_playwright
import os

def run_cuj(page):
    filepath = "file://" + os.path.abspath("/app/revipdf/src/index.html")
    page.goto(filepath)
    page.wait_for_timeout(1000)

    # We just want to verify the visual changes on the dashboard
    # such as the new title, subtitle, missing Recent section, etc.

    page.screenshot(path="/app/revipdf/verification.png")
    page.wait_for_timeout(1000)

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir="/app/revipdf"
        )
        page = context.new_page()
        try:
            run_cuj(page)
        finally:
            context.close()
            browser.close()
