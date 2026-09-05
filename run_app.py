import os
import sys
import time
import threading
import webview
import streamlit.web.bootstrap as bootstrap

def get_script_path():
    """Gets absolute path to app.py whether running directly or compiled."""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, "app.py")

def start_streamlit():
    """Starts Streamlit bound explicitly to 127.0.0.1:8501."""
    script_path = get_script_path()
    
    flag_options = {
        "server.port": 8501,
        "server.address": "127.0.0.1",
        "server.headless": True,
        "global.developmentMode": False,
        "server.enableCORS": False,
        "server.enableXsrfProtection": False
    }
    
    # Pre-configure runtime settings
    bootstrap.load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True
    
    bootstrap.run(script_path, "streamlit run", [], flag_options)

if __name__ == "__main__":
    # Start Streamlit in a daemon thread
    server_thread = threading.Thread(target=start_streamlit, daemon=True)
    server_thread.start()

    # Wait for the Streamlit server to boot
    time.sleep(5)

    # Launch app targeting 127.0.0.1:8501
    webview.create_window(
        title="AetherAI Desktop",
        url="http://127.0.0.1:8501",
        width=1280,
        height=800,
        resizable=True
    )
    webview.start()