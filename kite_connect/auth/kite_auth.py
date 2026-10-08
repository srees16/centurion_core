"""
Runs the Kite Connect login in your browser and captures the request token.

How it works:
1. Starts a local HTTP server on http://127.0.0.1:5000
2. Opens the Kite Connect login URL in your browser
3. You log in with your Zerodha credentials (+ TOTP/2FA); nothing is
   auto-filled (no automated Kite login, tracker U23)
4. Zerodha redirects back to the local server with the request_token
5. The script captures it, stores it in data/kite/request_token.txt
   (gitignored), and returns the token

IMPORTANT: Set your Kite Connect app's redirect URL to:
    http://127.0.0.1:5000
    (Go to https://developers.kite.trade -> Your App -> Redirect URL)
"""

import webbrowser
import subprocess
import tempfile
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Append kite_connect to path (not insert) to avoid shadowing top-level packages
_kite_root = os.path.dirname(os.path.dirname(__file__))
if _kite_root not in sys.path:
    sys.path.append(_kite_root)

from core.config import LOGIN_URL, KITE_APP_FILE

captured_token = None


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global captured_token
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if 'request_token' in query:
            captured_token = query['request_token'][0]
            status = query.get('status', ['unknown'])[0]

            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = "<html><body><script>window.close();</script></body></html>"
            self.wfile.write(html.encode())
        else:
            self.send_response(400)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Error: No request_token found.</h2></body></html>")

    def log_message(self, format, *args):
        # Suppress default request logging
        pass


def update_kite_app(token):
    """Store the request_token in ``KITE_APP_FILE`` (gitignored, owner-only)."""
    try:
        os.makedirs(os.path.dirname(KITE_APP_FILE), exist_ok=True)
        fd = os.open(KITE_APP_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(f"request_token='{token}'\n")
        print(f"  [OK] Stored the request_token in {KITE_APP_FILE}")
    except Exception as e:
        print(f"  [ERROR] Could not store the request_token: {e}")


def fetch_request_token():
    """
    Launch the Kite login flow, capture the request_token via local
    HTTP redirect, store it (``KITE_APP_FILE``), and return the new token.

    You type your user ID, password and TOTP in the browser yourself.

    Can be called from other modules:
        from kite_auth import fetch_request_token
        token = fetch_request_token()
    """
    global captured_token
    captured_token = None  # reset for re-entry

    server_address = ('127.0.0.1', 5000)
    httpd = HTTPServer(server_address, CallbackHandler)

    print("=" * 60)
    print("  Kite Connect - Request Token Generator")
    print("=" * 60)
    print(f"\n  Local callback server started on http://127.0.0.1:5000")
    print("  Opening Kite login page in your browser...\n")

    browser_proc = None
    temp_profile = None
    if sys.platform == 'win32':
        local = os.environ.get('LOCALAPPDATA', '')
        program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        program_files_x86 = os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')

        browser_paths = [
            os.path.join(local, r'BraveSoftware\Brave-Browser\Application\brave.exe'),
            os.path.join(program_files, r'BraveSoftware\Brave-Browser\Application\brave.exe'),
            os.path.join(program_files, r'Google\Chrome\Application\chrome.exe'),
            os.path.join(program_files_x86, r'Google\Chrome\Application\chrome.exe'),
            os.path.join(local, r'Google\Chrome\Application\chrome.exe'),
            os.path.join(program_files, r'Microsoft\Edge\Application\msedge.exe'),
            os.path.join(program_files_x86, r'Microsoft\Edge\Application\msedge.exe'),
            os.path.join(program_files, r'Mozilla Firefox\firefox.exe'),
            os.path.join(program_files_x86, r'Mozilla Firefox\firefox.exe'),
        ]

        for browser_path in browser_paths:
            if os.path.isfile(browser_path):
                try:
                    temp_profile = tempfile.mkdtemp(prefix='kite_login_')
                    browser_proc = subprocess.Popen(
                        [browser_path, f'--user-data-dir={temp_profile}',
                         '--no-first-run', '--no-default-browser-check',
                         LOGIN_URL]
                    )
                    print(f"  Using: {os.path.basename(browser_path)}")
                    break
                except Exception:
                    continue
    if browser_proc is None:
        webbrowser.open(LOGIN_URL)

    print("  Waiting for login redirect... (Ctrl+C to cancel)\n")

    while captured_token is None:
        httpd.handle_request()

    httpd.server_close()

    # Close the browser started above
    if browser_proc is not None:
        try:
            browser_proc.kill()
            browser_proc.wait(timeout=5)
        except Exception:
            pass
        if temp_profile and os.path.isdir(temp_profile):
            try:
                import shutil
                shutil.rmtree(temp_profile, ignore_errors=True)
            except Exception:
                pass

    print("=" * 60)
    print(f"  Request Token: {captured_token[:4]}... (stored, not shown)")
    print("=" * 60)

    update_kite_app(captured_token)

    print("\n  Done!")

    return captured_token


if __name__ == '__main__':
    fetch_request_token()
