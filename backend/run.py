print("Starting Momentum...")

import os
import webbrowser
import threading
from flask import Flask, send_from_directory
from flask_cors import CORS
from app.routes import bp

# frontent static files
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static')
CORS(app, resources={r"/upload": {"origins": "*"}})
app.register_blueprint(bp)

# all html files
@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Catch-all: serves any file from /static (html, css, js)."""
    return send_from_directory(STATIC_DIR, filename)

# open browser
def open_browser():
    import time
    time.sleep(1.2)          # wait for Flask to finish starting
    webbrowser.open('http://localhost:5000')

if __name__ == "__main__":
    print("✅ Opening http://localhost:5000 in your browser...")
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=10000, debug=False, threaded=True)
