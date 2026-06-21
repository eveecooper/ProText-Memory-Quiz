#!/usr/bin/env python3
"""
run.py — ProText Memory Quiz launcher
--------------------------------------
Usage:
    python run.py                  # local only (default)
    python run.py --host           # share with others on your network (LAN)
    python run.py --public         # share publicly via Cloudflare Tunnel
    python run.py --port 8080      # custom port
    python run.py --no-browser     # don't auto-open browser
    python run.py --debug          # Flask debug mode (dev only)

Requires cloudflared for --public:
    winget install Cloudflare.cloudflared

Password is only required when running with --host or --public.
"""

import argparse
import os
import re
import socket
import subprocess
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
  python run.py --public         # share with anyone on the internet (Cloudflare Tunnel)
  python run.py --host --port 8080
  python run.py --no-browser
        """,
    )
    parser.add_argument("--host",       action="store_true",
                        help="Allow other devices on your LAN to connect")
    parser.add_argument("--public",     action="store_true",
                        help="Create a public Cloudflare Tunnel (requires cloudflared)")
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
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def start_cloudflare_tunnel(port: int):
    """
    Spawns cloudflared and returns (public_url, process).
    Returns (None, None) if cloudflared is not installed.
    """
    try:
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        return None, None

    url_found = threading.Event()
    public_url = [None]

    def _read():
        for line in proc.stdout:
            match = re.search(r"https://[\w\-]+\.trycloudflare\.com", line)
            if match:
                public_url[0] = match.group(0)
                url_found.set()
                break
        # drain remaining output silently
        for _ in proc.stdout:
            pass

    threading.Thread(target=_read, daemon=True).start()
    url_found.wait(timeout=20)
    return public_url[0], proc


def open_browser(url: str, delay: float = 1.2):
    """Opens the browser after a short delay to let Flask bind."""
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main():
    args = parse_args()

    bind_host = "0.0.0.0" if (args.host or args.public) else "127.0.0.1"
    local_url = f"http://127.0.0.1:{args.port}"

    tunnel_proc = None

    if args.public:
        print("\n  Starting Cloudflare Tunnel, please wait...")
        public_url, tunnel_proc = start_cloudflare_tunnel(args.port)
        if tunnel_proc is None:
            print("  ERROR: cloudflared not found.")
            print("  Install it with:  winget install Cloudflare.cloudflared")
            sys.exit(1)
        if public_url is None:
            print("  ERROR: Cloudflare Tunnel did not return a URL in time.")
            print("  Make sure cloudflared is installed and you have internet access.")
            tunnel_proc.terminate()
            sys.exit(1)

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║        ProText Memory Quiz — running         ║")
    print(f"  ║   Local:   {local_url:<34}║")

    if args.host:
        lan_ip  = get_local_ip()
        lan_url = f"http://{lan_ip}:{args.port}"
        print(f"  ║   Network: {lan_url:<34}║")
        print("  ║   Share the Network URL with your friends    ║")

    if args.public:
        print(f"  ║   Public:  {public_url:<34}║")
        print("  ║   Share this URL with anyone on the internet ║")

    print("  ║   Ctrl+C to stop                             ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    if args.host:
        print("  ⚠  Network mode: anyone on your WiFi can connect.")
        print("     Stop the server with Ctrl+C when done sharing.\n")

    if args.public:
        print("  ⚠  Public mode: anyone with the URL can connect.")
        print("     Stop the server with Ctrl+C when done sharing.\n")

    if args.host or args.public:
        set_password(login_passcode)

    if not args.no_browser:
        open_browser(local_url)

    try:
        app.run(
            host=bind_host,
            port=args.port,
            debug=args.debug,
            use_reloader=False,
        )
    finally:
        if tunnel_proc:
            tunnel_proc.terminate()


if __name__ == "__main__":
    main()
