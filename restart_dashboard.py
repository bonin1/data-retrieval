import subprocess
import time
import os
import sys

def restart_dashboard():
    print("🔄 Restarting dashboard with fixes...")
    
    try:
        subprocess.run(['taskkill', '/f', '/im', 'streamlit.exe'], 
                      capture_output=True, check=False)
        print("✅ Stopped existing dashboard processes")
    except:
        pass
    
    time.sleep(2)
    
    print("🚀 Starting dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'real_time_dashboard.py'])

if __name__ == "__main__":
    restart_dashboard()
