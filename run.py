#!/usr/bin/env python3
"""
run.py — ProText Memory Quiz launcher
--------------------------------------
Usage:
    python run.py                  # local only (default)
    python run.py --host           # share with others on your network
    python run.py --port 8080      # custom port
    python run.py --no-browser     # don't auto-open browser
    python run.py --debug          # Flask debug mode (dev only)

Password is only required when running with --host.
To change it, edit the set_password() call in main() below.
"""

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, set_password
from vars import login_passcode


def parse_args():
    parser = argparse.ArgumentParser(
        description="ProText Memory Quiz — memory training app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                  # local only
  python run.py --host           # share with friends on same WiFi
  python run.py --host --port 8080
  python run.py --no-browser
        """,
    )
    parser.add_argument("--host",       action="store_true",
                        help="Allow other devices on your network to connect")
    parser.add_argument("--port",       type=int, default=5000,
                        help="Port to serve on (default: 5000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open the browser")
    parser.add_argument("--debug",      action="store_true",
                        help="Enable Flask debug mode (dev only)")
    return parser.parse_args()


def get_local_ip() -> str:
    """Best-effort detection of the machine's LAN IP address."""
    try:
        # Doesn't actually send traffic — just lets the OS pick the right interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def open_browser(url: str, delay: float = 1.2):
    """Opens the browser after a short delay to let Flask bind."""
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main():
    args = parse_args()

    bind_host = "0.0.0.0" if args.host else "127.0.0.1"
    local_url = f"http://127.0.0.1:{args.port}"

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║        ProText Memory Quiz — running         ║")
    print(f"  ║   Local:   {local_url:<34}║")

    if args.host:
        lan_ip   = get_local_ip()
        lan_url  = f"http://{lan_ip}:{args.port}"
        print(f"  ║   Network: {lan_url:<34}║")
        print("  ║   Share the Network URL with your friends    ║")

    print("  ║   Ctrl+C to stop                             ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    if args.host:
        print("  ⚠  Network mode: anyone on your WiFi can connect.")
        print("     Stop the server with Ctrl+C when done sharing.\n")
        # Enable password gate — only active when hosting for others
        set_password(login_passcode)

    if not args.no_browser:
        open_browser(local_url)

    app.run(
        host=bind_host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
