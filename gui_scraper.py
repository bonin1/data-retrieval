import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import queue
import os
import sys
import json
from typing import List, Dict, Any
import webbrowser

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gjirafa_scraper import GjirafaScraper
from config import ScraperConfig

class ScraperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🛒 Gjirafa50.com Advanced Scraper")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Success.TLabel', foreground='green')
        self.style.configure('Error.TLabel', foreground='red')
        
        self.scraper = None
        self.categories = []
        self.selected_category = tk.StringVar()
        self.headless_mode = tk.BooleanVar(value=True)
        self.product_count = tk.IntVar(value=10)
        self.export_json = tk.BooleanVar(value=True)
        self.export_csv = tk.BooleanVar(value=True)
        self.current_status = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar()
        
        self.message_queue = queue.Queue()
        
        self.setup_gui()
        
        self.check_queue()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        title_label = ttk.Label(main_frame, text="🛒 Gjirafa50.com Advanced Scraper", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        config_frame = ttk.LabelFrame(main_frame, text="⚙️ Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        
        ttk.Checkbutton(config_frame, text="🖥️ Headless Mode (faster)", 
                       variable=self.headless_mode).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        ttk.Label(config_frame, text="📦 Products to scrape:").grid(row=1, column=0, sticky=tk.W, pady=2)
        count_frame = ttk.Frame(config_frame)
        count_frame.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        ttk.Spinbox(count_frame, from_=1, to=10000, textvariable=self.product_count, width=10).pack(side=tk.LEFT)
        ttk.Label(count_frame, text="(1-10000)").pack(side=tk.LEFT, padx=(5, 0))
        
        export_frame = ttk.Frame(config_frame)
        export_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(export_frame, text="📤 Export formats:").pack(side=tk.LEFT)
        ttk.Checkbutton(export_frame, text="JSON", variable=self.export_json).pack(side=tk.LEFT, padx=(10, 5))
        ttk.Checkbutton(export_frame, text="CSV", variable=self.export_csv).pack(side=tk.LEFT, padx=(5, 0))
        
        category_frame = ttk.LabelFrame(main_frame, text="📂 Category Selection", padding="10")
        category_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        category_frame.columnconfigure(0, weight=1)
        
        ttk.Button(category_frame, text="🔍 Discover Categories", 
                  command=self.discover_categories).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.category_combo = ttk.Combobox(category_frame, textvariable=self.selected_category, 
                                          state="readonly", width=80)
        self.category_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        action_frame = ttk.LabelFrame(main_frame, text="🚀 Actions", padding="10")
        action_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        buttons_frame = ttk.Frame(action_frame)
        buttons_frame.pack(fill=tk.X)
        
        ttk.Button(buttons_frame, text="🛒 Start Scraping", 
                  command=self.start_scraping).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="📊 Show Product URLs", 
                  command=self.show_product_urls).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="📁 Open Export Folder", 
                  command=self.open_export_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="🔄 Reset", 
                  command=self.reset_scraper).pack(side=tk.LEFT, padx=(0, 5))
        
        progress_frame = ttk.LabelFrame(main_frame, text="📈 Progress", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.status_label = ttk.Label(progress_frame, textvariable=self.current_status)
        self.status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           mode='determinate', length=400)
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        log_frame = ttk.LabelFrame(main_frame, text="📝 Activity Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Button(log_frame, text="🗑️ Clear Log", 
                  command=self.clear_log).grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
    def log_message(self, message: str):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        
    def update_status(self, status: str):
        self.current_status.set(status)
        self.root.update_idletasks()
        
    def update_progress(self, value: float):
        self.progress_var.set(value)
        self.root.update_idletasks()
        
    def discover_categories(self):
        def worker():
            try:
                self.message_queue.put(("status", "🔍 Discovering categories..."))
                self.message_queue.put(("log", "Initializing scraper..."))
                
                config = ScraperConfig()
                config.HEADLESS = self.headless_mode.get()
                self.scraper = GjirafaScraper(config)
                
                self.message_queue.put(("log", "Discovering categories from website..."))
                categories = self.scraper.discover_category_urls()
                
                if categories:
                    self.categories = categories
                    category_display = []
                    for cat in categories:
                        name = cat.split('/')[-1].replace('-', ' ').title()
                        if not name:
                            name = cat.split('/')[-2].replace('-', ' ').title()
                        category_display.append(f"{name} - {cat}")
                    
                    self.message_queue.put(("categories", category_display))
                    self.message_queue.put(("status", f"✅ Found {len(categories)} categories"))
                    self.message_queue.put(("log", f"Successfully discovered {len(categories)} categories"))
                else:
                    self.message_queue.put(("error", "❌ No categories found"))
                    self.message_queue.put(("log", "ERROR: No categories found"))
                    
            except Exception as e:
                self.message_queue.put(("error", f"❌ Error: {str(e)}"))
                self.message_queue.put(("log", f"ERROR: {str(e)}"))
        
        threading.Thread(target=worker, daemon=True).start()
        
    def show_product_urls(self):
        if not self.selected_category.get():
            messagebox.showwarning("Warning", "Please select a category first")
            return
            
        def worker():
            try:
                selected_text = self.selected_category.get()
                category_url = selected_text.split(" - ")[-1]
                
                self.message_queue.put(("status", "🔍 Finding product URLs..."))
                self.message_queue.put(("log", f"Searching for products in: {category_url}"))
                
                if not self.scraper:
                    config = ScraperConfig()
                    config.HEADLESS = self.headless_mode.get()
                    self.scraper = GjirafaScraper(config)
                
                product_urls = self.scraper.discover_product_urls(category_url)
                
                if product_urls:
                    self.message_queue.put(("show_urls", (category_url, product_urls)))
                    self.message_queue.put(("status", f"✅ Found {len(product_urls)} product URLs"))
                    self.message_queue.put(("log", f"Found {len(product_urls)} product URLs"))
                else:
                    self.message_queue.put(("error", "❌ No product URLs found"))
                    self.message_queue.put(("log", "No product URLs found in this category"))
                    
            except Exception as e:
                self.message_queue.put(("error", f"❌ Error: {str(e)}"))
                self.message_queue.put(("log", f"ERROR: {str(e)}"))
        
        threading.Thread(target=worker, daemon=True).start()
        
    def start_scraping(self):
        if not self.selected_category.get():
            messagebox.showwarning("Warning", "Please select a category first")
            return
            
        if not (self.export_json.get() or self.export_csv.get()):
            messagebox.showwarning("Warning", "Please select at least one export format")
            return
            
        result = messagebox.askyesno("Confirm Scraping", 
                                   f"Start scraping {self.product_count.get()} products?\n"
                                   f"Category: {self.selected_category.get().split(' - ')[0]}")
        if not result:
            return
            
        def worker():
            try:
                selected_text = self.selected_category.get()
                category_url = selected_text.split(" - ")[-1]
                category_name = selected_text.split(" - ")[0]
                
                self.message_queue.put(("status", "🚀 Starting scraper..."))
                self.message_queue.put(("log", f"=== STARTING SCRAPING SESSION ==="))
                self.message_queue.put(("log", f"Category: {category_name}"))
                self.message_queue.put(("log", f"Target products: {self.product_count.get()}"))
                
                if not self.scraper:
                    config = ScraperConfig()
                    config.HEADLESS = self.headless_mode.get()
                    self.scraper = GjirafaScraper(config)
                self.message_queue.put(("status", "🔍 Finding products..."))
                self.message_queue.put(("log", "Discovering product URLs..."))
                target_count = self.product_count.get()
                product_urls = self.scraper.discover_product_urls(category_url, max_products=target_count)
                
                if not product_urls:
                    self.message_queue.put(("error", "❌ No products found"))
                    return
                
                self.message_queue.put(("log", f"Found {len(product_urls)} product URLs (target: {target_count})"))
                
                self.message_queue.put(("log", f"Will scrape {len(product_urls)} products"))
                
                self.scraper.products = []
                
                scraped_count = 0
                for i, url in enumerate(product_urls):
                    progress = (i / len(product_urls)) * 100
                    self.message_queue.put(("progress", progress))
                    self.message_queue.put(("status", f"📦 Scraping product {i+1}/{len(product_urls)}"))
                    self.message_queue.put(("log", f"Scraping product {i+1}: {url}"))
                    
                    product_data = self.scraper.extract_product_data(url)
                    
                    if product_data:
                        self.scraper.products.append(product_data)
                        scraped_count += 1
                        self.message_queue.put(("log", f"  ✅ {product_data.get('title', 'N/A')} - {product_data.get('price', 'N/A')}"))
                    else:
                        self.message_queue.put(("log", f"  ❌ Failed to extract data"))
                
                if self.scraper.products:
                    self.message_queue.put(("status", "📤 Exporting data..."))
                    self.message_queue.put(("log", f"Exporting {len(self.scraper.products)} products..."))
                    
                    export_formats = []
                    if self.export_json.get():
                        export_formats.append('json')
                    if self.export_csv.get():
                        export_formats.append('csv')
                    
                    filename_prefix = f"{category_name.lower().replace(' ', '_')}_scrape"
                    exported_files = self.scraper.export_data(
                        formats=export_formats,
                        filename_prefix=filename_prefix
                    )
                    
                    self.message_queue.put(("progress", 100))
                    self.message_queue.put(("status", "✅ Scraping completed!"))
                    self.message_queue.put(("log", f"=== SCRAPING COMPLETED ==="))
                    self.message_queue.put(("log", f"Total products scraped: {len(self.scraper.products)}"))
                    
                    for format_type, filepath in exported_files.items():
                        self.message_queue.put(("log", f"Exported {format_type.upper()}: {filepath}"))
                        if os.path.exists(filepath):
                            size = os.path.getsize(filepath) / (1024 * 1024)
                            self.message_queue.put(("log", f"  File size: {size:.2f} MB"))
                    
                    self.message_queue.put(("completion", {
                        'category': category_name,
                        'scraped': len(self.scraper.products),
                        'target': target_count,
                        'files': exported_files
                    }))
                    
                else:
                    self.message_queue.put(("error", "❌ No products were scraped successfully"))
                    
            except Exception as e:
                self.message_queue.put(("error", f"❌ Error: {str(e)}"))
                self.message_queue.put(("log", f"ERROR: {str(e)}"))
                import traceback
                self.message_queue.put(("log", traceback.format_exc()))
        
        threading.Thread(target=worker, daemon=True).start()
        
    def open_export_folder(self):
        """Open the export folder"""
        export_dir = os.path.join(os.getcwd(), "scraped_data")
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        if os.name == 'nt':
            os.startfile(export_dir)
        elif os.name == 'posix': 
            os.system(f'open "{export_dir}"' if sys.platform == 'darwin' else f'xdg-open "{export_dir}"')
            
    def reset_scraper(self):
        if self.scraper:
            self.scraper.close()
            self.scraper = None
        
        self.categories = []
        self.category_combo['values'] = []
        self.selected_category.set("")
        self.current_status.set("Ready")
        self.progress_var.set(0)
        self.log_message("🔄 Scraper reset")
        
    def show_urls_window(self, category_url: str, product_urls: List[str]):
        urls_window = tk.Toplevel(self.root)
        urls_window.title(f"Product URLs - {category_url}")
        urls_window.geometry("800x600")
        
        main_frame = ttk.Frame(urls_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text=f"📦 Found {len(product_urls)} Products", 
                 style='Heading.TLabel').pack(pady=(0, 10))
        
        url_frame = ttk.Frame(main_frame)
        url_frame.pack(fill=tk.BOTH, expand=True)
        
        url_text = scrolledtext.ScrolledText(url_frame, height=25)
        url_text.pack(fill=tk.BOTH, expand=True)
        
        for i, url in enumerate(product_urls, 1):
            url_text.insert(tk.END, f"{i:3d}. {url}\n")
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="📋 Copy All URLs", 
                  command=lambda: self.copy_to_clipboard('\n'.join(product_urls))).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="💾 Save to File", 
                  command=lambda: self.save_urls_to_file(product_urls)).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(button_frame, text="🌐 Open First URL", 
                  command=lambda: webbrowser.open(product_urls[0]) if product_urls else None).pack(side=tk.LEFT, padx=(5, 0))
        
    def copy_to_clipboard(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", "URLs copied to clipboard!")
        
    def save_urls_to_file(self, urls: List[str]):
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    for url in urls:
                        f.write(f"{url}\n")
                messagebox.showinfo("Saved", f"URLs saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")
                
    def show_completion_dialog(self, data: Dict[str, Any]):
        """Show scraping completion dialog"""
        message = f"""
🎉 Scraping Completed Successfully!

📊 Results:
   Category: {data['category']}
   Products Scraped: {data['scraped']}/{data['target']}
   Success Rate: {(data['scraped']/data['target']*100):.1f}%

📁 Exported Files:
"""
        for format_type, filepath in data['files'].items():
            filename = os.path.basename(filepath)
            message += f"   {format_type.upper()}: {filename}\n"
        
        result = messagebox.showinfo("Scraping Complete", message)
        
    def check_queue(self):
        try:
            while True:
                message_type, message_data = self.message_queue.get_nowait()
                
                if message_type == "status":
                    self.update_status(message_data)
                elif message_type == "log":
                    self.log_message(message_data)
                elif message_type == "progress":
                    self.update_progress(message_data)
                elif message_type == "categories":
                    self.category_combo['values'] = message_data
                elif message_type == "error":
                    self.update_status(message_data)
                    self.log_message(message_data)
                elif message_type == "show_urls":
                    category_url, product_urls = message_data
                    self.show_urls_window(category_url, product_urls)
                elif message_type == "completion":
                    self.show_completion_dialog(message_data)
                    
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_queue)
        
    def on_closing(self):
        if self.scraper:
            self.scraper.close()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ScraperGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
