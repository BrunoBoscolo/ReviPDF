from playwright.sync_api import sync_playwright
import http.server
import socketserver
import threading
import time

PORT = 8004
DIRECTORY = 'revipdf/src'

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    with socketserver.TCPServer(('', PORT), Handler) as httpd:
        httpd.serve_forever()

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
time.sleep(1)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Capture and print all console logs
    page.on("console", lambda msg: print(f"BROWSER LOG: {msg.text}"))

    page.add_init_script('''
        window.__TAURI_INTERNALS__ = {};
        window.__TAURI__ = {
            fs: {
                readTextFile: async () => '{}',
                BaseDirectory: { AppLocalData: 1 }
            }
        };
    ''')

    print("Navigating to index to set localStorage...")
    page.goto('http://localhost:8004/index.html')
    page.evaluate("localStorage.setItem('currentPdfHash', 'mock_hash_123')")

    print("Navigating to review.html...")
    page.goto('http://localhost:8004/review.html')

    page.wait_for_timeout(2000)
    print("Done waiting.")
    browser.close()
