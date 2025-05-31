import sys
import subprocess
import importlib
import platform
from pathlib import Path

def check_python_version():
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_dependencies():
    dependencies = [
        'requests',
        'beautifulsoup4',
        'selenium',
        'pandas',
        'lxml',
        'fake_useragent',
        'urllib3',
        'webdriver_manager',
        'python_dotenv',
        'tqdm',
        'python_dateutil',
        'validators',
        'cloudscraper',
        'aiohttp'
    ]
    
    print("\n📦 Checking dependencies...")
    missing_deps = []
    
    for dep in dependencies:
        try:
            import_name = dep
            if dep == 'beautifulsoup4':
                import_name = 'bs4'
            elif dep == 'python_dotenv':
                import_name = 'dotenv'
            elif dep == 'python_dateutil':
                import_name = 'dateutil'
            
            importlib.import_module(import_name)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} - Not installed")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("📋 To install missing dependencies, run:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies are installed")
        return True

def check_chrome_browser():
    print("\n🌐 Checking Chrome browser...")
    
    try:
        if platform.system() == "Windows":
            import winreg
            try:
                reg_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path):
                    print("  ✅ Chrome browser found in registry")
                    return True
            except FileNotFoundError:
                pass
        
        result = subprocess.run(['chrome', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✅ Chrome browser found: {version}")
            return True
        else:
            raise Exception("Chrome command failed")
            
    except Exception:
        try:
            for cmd in ['google-chrome', 'google-chrome-stable', 'chromium']:
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    print(f"  ✅ Chrome browser found: {version}")
                    return True
        except Exception:
            pass
    
    print("  ⚠️  Chrome browser not found")
    print("     Selenium features may not work properly")
    print("     Please install Google Chrome browser")
    return False

def test_webdriver():
    print("\n🚗 Testing ChromeDriver...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        import os
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-extensions")
        driver_path = ChromeDriverManager().install()
        print(f"     ChromeDriver path: {driver_path}")
        
        if not os.path.exists(driver_path) or not driver_path.endswith('.exe'):
            driver_dir = os.path.dirname(driver_path)
            found = False
            for root, dirs, files in os.walk(driver_dir):
                for file in files:
                    if file.lower() == 'chromedriver.exe':
                        driver_path = os.path.join(root, file)
                        found = True
                        break
                if found:
                    break
        
        if not os.path.exists(driver_path):
            raise Exception(f"ChromeDriver executable not found at {driver_path}")
        
        print(f"     Using ChromeDriver at: {driver_path}")
        
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        driver.get("https://www.google.com")
        title = driver.title
        driver.quit()
        
        print(f"  ✅ ChromeDriver working correctly")
        print(f"     Test page title: {title}")
        return True
        
    except Exception as e:
        print(f"  ❌ ChromeDriver test failed: {e}")
        print("     Selenium features may not work")
        print("     This is optional - the scraper can work without Selenium")
        return False

def test_basic_imports():
    print("\n📥 Testing module imports...")
    
    try:
        from config import ScraperConfig, SELECTORS
        print("  ✅ config.py imported successfully")
        
        from utils import DataValidator, DataExporter
        print("  ✅ utils.py imported successfully")
        
        from gjirafa_scraper import GjirafaScraper
        print("  ✅ gjirafa_scraper.py imported successfully")
        
        config = ScraperConfig()
        validator = DataValidator()
        
        print("  ✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        return False

def test_network_connectivity():
    print("\n🌍 Testing network connectivity...")
    
    try:
        import requests
        
        response = requests.get("https://gjirafa50.com", timeout=10)
        if response.status_code == 200:
            print("  ✅ gjirafa50.com is accessible")
        else:
            print(f"  ⚠️  gjirafa50.com returned status code: {response.status_code}")
        
        try:
            response = requests.get("https://gjirafamall.com", timeout=10)
            if response.status_code == 200:
                print("  ✅ gjirafamall.com is accessible")
            else:
                print(f"  ⚠️  gjirafamall.com returned status code: {response.status_code}")
        except Exception:
            print("  ⚠️  gjirafamall.com is not accessible")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Network connectivity test failed: {e}")
        print("     Please check your internet connection")
        return False

def create_output_directory():
    print("\n📁 Setting up output directory...")
    
    try:
        output_dir = Path("scraped_data")
        output_dir.mkdir(exist_ok=True)
        
        test_file = output_dir / "test.txt"
        test_file.write_text("Test file for permissions")
        test_file.unlink()
        
        print(f"  ✅ Output directory created: {output_dir.absolute()}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to create output directory: {e}")
        return False

def run_basic_scraper_test():
    print("\n🧪 Running basic scraper test...")
    
    try:
        from gjirafa_scraper import GjirafaScraper
        from config import ScraperConfig
        
        config = ScraperConfig()
        config.REQUEST_DELAY = 0.1
        config.TIMEOUT = 10
        
        scraper = GjirafaScraper(config)
        print("  ✅ Scraper initialized successfully")
        
        from utils import URLHelper
        test_url = URLHelper.normalize_url("/test", "https://gjirafa50.com")
        if test_url == "https://gjirafa50.com/test":
            print("  ✅ URL helper working correctly")
        
        from utils import DataValidator
        test_price = DataValidator.extract_price("€29.99")
        if test_price == 29.99:
            print("  ✅ Data validator working correctly")
        
        scraper.close()
        print("  ✅ Basic scraper test completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic scraper test failed: {e}")
        return False

def main():
    print("🔧 Gjirafa50.com Scraper Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Chrome Browser", check_chrome_browser),
        ("ChromeDriver", test_webdriver),
        ("Module Imports", test_basic_imports),
        ("Network Connectivity", test_network_connectivity),
        ("Output Directory", create_output_directory),
        ("Basic Scraper Test", run_basic_scraper_test)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name} {'='*20}")
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results[check_name] = False
    
    print("\n" + "="*50)
    print("📊 VERIFICATION SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! The scraper is ready to use.")
        print("\n📚 Next steps:")
        print("   1. Run examples: python examples.py")
        print("   2. Use CLI: python cli.py --help")
        print("   3. Start scraping: python cli.py --max-products 10")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before using the scraper.")
        
        if not results.get("Dependencies", True):
            print("\n🔧 To fix dependency issues:")
            print("   pip install -r requirements.txt")
        
        if not results.get("Chrome Browser", True):
            print("\n🔧 To fix Chrome browser issues:")
            print("   Download and install Google Chrome from:")
            print("   https://www.google.com/chrome/")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
