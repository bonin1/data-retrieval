"""
Anti-detection utilities for web scraping
"""
import random
import time
from typing import List, Dict, Optional
from fake_useragent import UserAgent
import urllib.parse


class AntiDetection:
    """Anti-detection measures for web scraping"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session_cookies = {}
        self.request_count = 0
        self.last_request_time = 0
        
    def get_random_user_agent(self) -> str:
        """Get a random user agent"""
        try:
            return self.ua.random
        except Exception:
            # Fallback user agents
            fallback_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
            ]
            return random.choice(fallback_agents)
    
    def get_random_headers(self, url: str = None) -> Dict[str, str]:
        """Generate randomized headers"""
        base_headers = {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': random.choice([
                'en-US,en;q=0.9',
                'en-US,en;q=0.8,sq;q=0.6',
                'sq-AL,sq;q=0.9,en;q=0.8',
                'en-GB,en;q=0.9',
            ]),
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': random.choice(['none', 'same-origin', 'cross-site']),
            'Cache-Control': random.choice(['no-cache', 'max-age=0', 'no-store']),
        }
        
        # Add referer occasionally
        if url and random.random() < 0.3:
            parsed_url = urllib.parse.urlparse(url)
            base_headers['Referer'] = f"{parsed_url.scheme}://{parsed_url.netloc}/"
        
        # Randomly add some optional headers
        optional_headers = {
            'DNT': '1',
            'Sec-GPC': '1',
            'Pragma': 'no-cache',
        }
        
        for header, value in optional_headers.items():
            if random.random() < 0.5:
                base_headers[header] = value
        
        return base_headers
    
    def calculate_delay(self, min_delay: float = 1.0, max_delay: float = 3.0) -> float:
        """Calculate intelligent delay between requests"""
        self.request_count += 1
        
        # Base delay
        base_delay = random.uniform(min_delay, max_delay)
        
        # Increase delay based on request frequency
        if self.request_count % 10 == 0:
            base_delay *= 1.5
        elif self.request_count % 50 == 0:
            base_delay *= 2.0
        elif self.request_count % 100 == 0:
            base_delay *= 3.0
        
        # Add some randomness to avoid pattern detection
        jitter = random.uniform(-0.2, 0.2) * base_delay
        final_delay = max(0.5, base_delay + jitter)
        
        return final_delay
    
    def wait_with_jitter(self, min_delay: float = 1.0, max_delay: float = 3.0):
        """Wait with intelligent delay"""
        delay = self.calculate_delay(min_delay, max_delay)
        time.sleep(delay)
        self.last_request_time = time.time()
    
    def should_use_proxy(self) -> bool:
        """Determine if proxy should be used based on request patterns"""
        # Use proxy after many requests or if rate limited
        return self.request_count > 200 or self.request_count % 75 == 0
    
    def generate_session_fingerprint(self) -> Dict[str, str]:
        """Generate a consistent session fingerprint"""
        if not hasattr(self, '_session_fingerprint'):
            self._session_fingerprint = {
                'screen_resolution': random.choice([
                    '1920x1080', '1366x768', '1440x900', '1536x864', '1280x720'
                ]),
                'color_depth': random.choice(['24', '32']),
                'timezone': random.choice([
                    'Europe/Tirane', 'Europe/London', 'Europe/Berlin', 'America/New_York'
                ]),
                'language': random.choice(['en-US', 'sq-AL', 'en-GB']),
                'platform': random.choice(['Win32', 'MacIntel', 'Linux x86_64']),
            }
        return self._session_fingerprint
    
    def obfuscate_url_parameters(self, url: str) -> str:
        """Add random parameters to URL to avoid caching"""
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        
        # Add cache busting parameter
        params['_t'] = [str(int(time.time()))]
        
        # Occasionally add other tracking-like parameters
        if random.random() < 0.3:
            params['ref'] = [random.choice(['direct', 'search', 'social'])]
        
        if random.random() < 0.2:
            params['utm_source'] = ['organic']
        
        new_query = urllib.parse.urlencode(params, doseq=True)
        return urllib.parse.urlunparse(parsed._replace(query=new_query))
    
    def get_realistic_viewport_size(self) -> tuple:
        """Get realistic browser viewport size"""
        viewports = [
            (1920, 1080), (1366, 768), (1440, 900), (1536, 864),
            (1280, 720), (1600, 900), (1024, 768), (1280, 1024)
        ]
        return random.choice(viewports)
    
    def simulate_human_behavior(self):
        """Simulate human-like browsing patterns"""
        # Random scroll simulation (for headless browsers)
        scroll_actions = random.randint(1, 5)
        
        # Random mouse movements (conceptual - for selenium)
        mouse_movements = random.randint(2, 8)
        
        # Page interaction delay
        interaction_delay = random.uniform(0.5, 2.0)
        
        return {
            'scroll_actions': scroll_actions,
            'mouse_movements': mouse_movements,
            'interaction_delay': interaction_delay
        }
    
    def detect_rate_limiting(self, response_text: str, status_code: int) -> bool:
        """Detect if we're being rate limited"""
        rate_limit_indicators = [
            'rate limit', 'too many requests', 'slow down',
            'captcha', 'blocked', 'forbidden', 'access denied',
            'temporarily unavailable', 'service unavailable'
        ]
        
        # Check status code
        if status_code in [429, 503, 403, 521, 522, 523, 524]:
            return True
        
        # Check response content
        response_lower = response_text.lower()
        return any(indicator in response_lower for indicator in rate_limit_indicators)
    
    def handle_rate_limiting(self, attempt: int = 1):
        """Handle rate limiting with exponential backoff"""
        base_delay = 2 ** attempt  # Exponential backoff
        jitter = random.uniform(0.5, 1.5)
        total_delay = min(300, base_delay * jitter)  # Max 5 minutes
        
        print(f"Rate limited detected. Waiting {total_delay:.1f} seconds...")
        time.sleep(total_delay)
    
    def rotate_session_data(self):
        """Rotate session data to appear as new user"""
        self.session_cookies.clear()
        self.request_count = 0
        if hasattr(self, '_session_fingerprint'):
            delattr(self, '_session_fingerprint')
        
        # Reset user agent
        try:
            self.ua = UserAgent()
        except Exception:
            pass


class ProxyManager:
    """Manage proxy rotation"""
    
    def __init__(self, proxy_file: Optional[str] = None):
        self.proxies = []
        self.current_proxy_index = 0
        self.failed_proxies = set()
        
        if proxy_file:
            self.load_proxies(proxy_file)
    
    def load_proxies(self, proxy_file: str):
        """Load proxies from file"""
        try:
            with open(proxy_file, 'r') as f:
                for line in f:
                    proxy = line.strip()
                    if proxy and not proxy.startswith('#'):
                        self.proxies.append(proxy)
        except FileNotFoundError:
            print(f"Proxy file {proxy_file} not found")
    
    def get_next_proxy(self) -> Optional[Dict[str, str]]:
        """Get next working proxy"""
        if not self.proxies:
            return None
        
        attempts = 0
        while attempts < len(self.proxies):
            proxy = self.proxies[self.current_proxy_index]
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
            
            if proxy not in self.failed_proxies:
                return {
                    'http': proxy,
                    'https': proxy
                }
            
            attempts += 1
        
        # If all proxies failed, reset and try again
        self.failed_proxies.clear()
        return self.get_next_proxy()
    
    def mark_proxy_failed(self, proxy: str):
        """Mark a proxy as failed"""
        self.failed_proxies.add(proxy)
    
    def get_proxy_count(self) -> int:
        """Get total number of proxies"""
        return len(self.proxies)
    
    def get_working_proxy_count(self) -> int:
        """Get number of working proxies"""
        return len(self.proxies) - len(self.failed_proxies)
