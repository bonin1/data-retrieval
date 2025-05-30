"""
Setup script for Gjirafa50.com scraper
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have Python {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_requirements():
    """Install required packages"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Upgrade pip first
    run_command(f'"{sys.executable}" -m pip install --upgrade pip', "Upgrading pip")
    
    # Install requirements
    cmd = f'"{sys.executable}" -m pip install -r requirements.txt'
    return run_command(cmd, "Installing requirements")


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "output",
        "logs",
        "data",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ Directories created")
    return True


def create_env_file():
    """Create .env file template"""
    print("⚙️  Creating .env file template...")
    
    env_content = """# Gjirafa50 Scraper Configuration
# Copy this file to .env and fill in your values

# Proxy Configuration (optional)
PROXY_USERNAME=your_proxy_username
PROXY_PASSWORD=your_proxy_password

# API Keys (if needed)
GJIRAFA_API_KEY=your_api_key_if_needed

# Database (optional)
DATABASE_URL=sqlite:///products.db

# Advanced Settings
MAX_WORKERS=8
REQUEST_DELAY_MIN=1
REQUEST_DELAY_MAX=3
"""
    
    env_file = Path(".env.example")
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"✅ Created {env_file}")
    print("   Copy this to .env and customize your settings")
    return True


def create_proxies_file():
    """Create proxies.txt template"""
    print("🌐 Creating proxies.txt template...")
    
    proxies_content = """# Proxy list (one per line)
# Format: http://username:password@proxy:port
# or: http://proxy:port
# 
# Examples:
# http://user:pass@proxy1.example.com:8080
# http://proxy2.example.com:3128
# socks5://proxy3.example.com:1080
#
# Remove the # to uncomment and add your actual proxies
"""
    
    proxies_file = Path("proxies.txt.example")
    with open(proxies_file, "w") as f:
        f.write(proxies_content)
    
    print(f"✅ Created {proxies_file}")
    print("   Rename to proxies.txt and add your proxy servers if needed")
    return True


def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import requests
        import beautifulsoup4
        import pandas
        import aiohttp
        print("✅ All required packages imported successfully")
        
        # Test basic functionality
        from gjirafa_scraper import GjirafaScraper
        from config import BASE_URL, SCRAPING_CONFIG
        from data_models import Product
        print("✅ All modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def show_usage_instructions():
    """Show how to use the scraper"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📚 How to use the scraper:\n")
    
    print("1️⃣  Simple usage (recommended for beginners):")
    print("   python example.py")
    print()
    
    print("2️⃣  Command line interface:")
    print("   python main.py                    # Scrape all categories")
    print("   python main.py --list-categories  # List available categories")
    print("   python main.py --categories electronics fashion")
    print("   python main.py --dry-run          # Test without scraping")
    print()
    
    print("3️⃣  Programmatic usage:")
    print("   from gjirafa_scraper import GjirafaScraper")
    print("   # See example.py for complete examples")
    print()
    
    print("4️⃣  Configuration:")
    print("   - Edit config.py for scraping settings")
    print("   - Copy .env.example to .env for secrets")
    print("   - Add proxies to proxies.txt if needed")
    print()
    
    print("📁 Output files will be saved in the 'output' directory")
    print("📋 Logs will be saved in the 'logs' directory")
    print()
    print("⚠️  Please respect the website's robots.txt and terms of service!")
    print("⚠️  Use reasonable delays to avoid overloading the server!")
    print()


def main():
    """Main setup function"""
    print("🚀 GJIRAFA50.COM SCRAPER SETUP")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your Python/pip installation.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create config files
    create_env_file()
    create_proxies_file()
    
    # Run tests
    if not run_tests():
        print("❌ Setup verification failed. Please check the error messages above.")
        sys.exit(1)
    
    # Show usage instructions
    show_usage_instructions()


if __name__ == "__main__":
    main()
