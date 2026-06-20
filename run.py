#!/usr/bin/env python3
"""
run.py — ProText Memory Quiz launcher
--------------------------
Run from within the protext_memory_quiz/ directory:

    python run.py
    python run.py --port 8080
    python run.py --no-browser
    python run.py --debug
"""

import argparse
import os
import sys
import threading
import time
import webbrowser

# Ensure we can import app.py from this same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app   # Flask app object


def parse_args():
    parser = argparse.ArgumentParser(
        description="ProText Memory Quiz — local memory training app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py
  python run.py --port 8080
  python run.py --no-browser
  python run.py --debug
        """,
    )
    parser.add_argument("--port",       type=int, default=5000,
                        help="Port to serve on (default: 5000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open the browser")
    parser.add_argument("--debug",      action="store_true",
                        help="Enable Flask debug mode (development only)")
    return parser.parse_args()


def open_browser(port: int, delay: float = 1.2):
    """Opens the default browser after a short delay to let Flask bind."""
    def _open():
        time.sleep(delay)
        webbrowser.open(f"http://127.0.0.1:{port}")
    threading.Thread(target=_open, daemon=True).start()


def main():
    args = parse_args()

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║          ProText Memory Quiz — running           ║")
    print(f"  ║   http://127.0.0.1:{args.port:<5}             ║")
    print("  ║   Ctrl+C to stop                     ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    if not args.no_browser:
        open_browser(args.port)

    app.run(
        host="127.0.0.1",
        port=args.port,
        debug=args.debug,
        use_reloader=False,   # reloader would cause double browser launch
    )


if __name__ == "__main__":
    main()
