import os
import tkinter as tk
from tkinter import (
    Tk, Label, Button, filedialog, Text, Scrollbar, Frame, Entry, BOTH, RIGHT,
    Y, LEFT, OptionMenu, StringVar, Checkbutton, IntVar
)
from tkinter import ttk  # Import for Treeview and Progressbar
import threading

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pypdf.errors import PdfReadError, PdfStreamError


def get_pdf_files(pdfs_dir):
    """Returns a list of PDF files in the specified directory."""
    return [f for f in os.listdir(pdfs_dir) if f.lower().endswith('.pdf')]

def sanitize_filename(filename):
    """Removes invalid characters from a filename."""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()

def ai_renamer_logic(api, model, pdfs_dir, naming_pattern, log_func, progress_callback):
    """
    Handles the AI logic for generating new filenames. This function is designed
    to be run in a separate thread to keep the UI responsive.
    """
    files = get_pdf_files(pdfs_dir)
    if not files:
        log_func("No PDF files found in the selected directory.", "error")
        return []

    try:
        if 'gemini' in model or 'o4' in model:
            llm = ChatGoogleGenerativeAI(model=model, google_api_key=api)
        elif 'gpt' in model:
            llm = ChatOpenAI(temperature=0, model_name=model, api_key=api)
        elif 'claude' in model:
            llm = ChatAnthropic(temperature=0, model=model, anthropic_api_key=api)
        elif 'mistral' in model:
            llm = ChatMistralAI(temperature=0, model=model, mistral_api_key=api)
        else:
            raise ValueError(f"Unsupported model: {model}")
    except Exception as e:
        log_func(f"Error initializing AI model: {e}", "error")
        return []

    user_input = (
        f"Based on the provided file, generate a file name in this format: {naming_pattern}"
        "Please do not give any response except for the file name. "
        "Do not include symbols like /, ~, !, @, #, or $ in the file name."
        "If a required piece of information (like title or year) cannot be found, please generate a suitable one based on the content."
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", 'You are a helpful assistant. Use the following context when responding:\n\n{context}.'),
        ("human", "{question}")
    ])

    output_parser = StrOutputParser()
    rag_chain = rag_prompt | llm | output_parser

    proposed_renames = []
    total_files = len(files)
    for i, f in enumerate(files):
        pdf_path = os.path.join(pdfs_dir, f)
        log_func(f"Processing file: {f}", "info")

        try:
            loader = PyPDFLoader(file_path=pdf_path)
            documents = loader.load_and_split()
            context = " ".join(page.page_content for page in documents)
            if len(context) > 128000:
                context = context[:128000]

            response = rag_chain.invoke({"question": user_input, "context": context})
            new_file_name = sanitize_filename(response.strip())
            if not new_file_name.lower().endswith('.pdf'):
                new_file_name += '.pdf'

            proposed_renames.append((f, new_file_name))
            log_func(f"Proposed name for {f}: {new_file_name}", "info")

        except (PdfReadError, PdfStreamError) as e:
            log_func(f"Error reading PDF {f}: {e}", "error")
            proposed_renames.append((f, "Error - Could not read PDF"))
        except Exception as e:
            log_func(f"An unexpected error occurred with {f}: {e}", "error")
            proposed_renames.append((f, f"Error - {e}"))
        
        # Update progress
        progress_callback((i + 1) / total_files * 100)
    
    return proposed_renames


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Renamer Master v2.0")
        self.root.geometry("800x700")
        
        self.pdfs_dir = ''
        self.proposed_renames = []

        # --- Main Layout Frames ---
        top_frame = Frame(root, padx=10, pady=10)
        top_frame.pack(fill=BOTH, expand=False)
        
        preview_frame = Frame(root, padx=10, pady=5)
        preview_frame.pack(fill=BOTH, expand=True)

        log_frame = Frame(root, padx=10, pady=5)
        log_frame.pack(fill=BOTH, expand=False)
        
        # --- Step 1: Configuration ---
        config_frame = ttk.LabelFrame(top_frame, text="Step 1: Configure AI Settings")
        config_frame.pack(fill=BOTH, expand=True, side=LEFT, padx=5)

        Label(config_frame, text="API Key:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.entry_api = Entry(config_frame, width=40, show="*")
        self.entry_api.grid(row=0, column=1, padx=5, pady=5)
        
        self.show_api_var = IntVar()
        self.show_api_check = Checkbutton(config_frame, text="Show", variable=self.show_api_var, command=self.toggle_api_visibility)
        self.show_api_check.grid(row=0, column=2, padx=5, pady=5)

        Label(config_frame, text="Select Model:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.model_var = StringVar(config_frame)
        models = ["gemini-2.5-flash", "gpt-4o-mini"]
        self.model_var.set(models[0])
        self.model_menu = OptionMenu(config_frame, self.model_var, *models)
        self.model_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        Label(config_frame, text="Naming Pattern:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.naming_pattern_entry = Entry(config_frame, width=40)
        self.naming_pattern_entry.insert(0, "[published year]_[title of the research].pdf")
        self.naming_pattern_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        # --- Step 2: Directory Selection ---
        dir_frame = ttk.LabelFrame(top_frame, text="Step 2: Select PDF Directory")
        dir_frame.pack(fill=BOTH, expand=True, side=LEFT, padx=5)

        self.select_button = Button(dir_frame, text="Select Directory", command=self.select_directory, font=("Helvetica", 10))
        self.select_button.pack(pady=10, padx=10)
        self.label_dir = Label(dir_frame, text="No directory selected.", wraplength=250)
        self.label_dir.pack(pady=5, padx=10)

        # --- Step 3: Preview and Rename ---
        preview_controls_frame = ttk.LabelFrame(preview_frame, text="Step 3: Preview and Rename Files")
        preview_controls_frame.pack(fill=BOTH, expand=True)

        controls = Frame(preview_controls_frame)
        controls.pack(fill='x', pady=5, padx=5)

        self.preview_button = Button(controls, text="Preview Changes", command=self.start_preview_thread, font=("Helvetica", 10, "bold"), bg="#007BFF", fg="white")
        self.preview_button.pack(side=LEFT, padx=5)
        
        self.run_button = Button(controls, text="Confirm and Rename", command=self.run_renamer, font=("Helvetica", 10, "bold"), bg="#28A745", fg="white", state="disabled")
        self.run_button.pack(side=LEFT, padx=5)

        self.progress = ttk.Progressbar(controls, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(side=LEFT, padx=10, fill='x', expand=True)
        
        # --- Preview Table ---
        tree_frame = Frame(preview_controls_frame)
        tree_frame.pack(fill=BOTH, expand=True, pady=5)

        self.tree = ttk.Treeview(tree_frame, columns=("Original", "New"), show="headings")
        self.tree.heading("Original", text="Original Filename")
        self.tree.heading("New", text="Proposed New Filename")
        self.tree.column("Original", width=300)
        self.tree.column("New", width=400)
        
        tree_scroll = Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        tree_scroll.pack(side=RIGHT, fill=Y)

        # --- Log Window ---
        log_outer_frame = ttk.LabelFrame(log_frame, text="Log")
        log_outer_frame.pack(fill=BOTH, expand=True)

        self.log_text = Text(log_outer_frame, height=8, wrap='word', font=("Helvetica", 9))
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)
        log_scroll = Scrollbar(log_outer_frame, command=self.log_text.yview)
        log_scroll.pack(side=RIGHT, fill=Y)
        self.log_text.config(yscrollcommand=log_scroll.set)
        
        # --- Log message colors ---
        self.log_text.tag_config("info", foreground="black")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("success", foreground="green")

    def toggle_api_visibility(self):
        if self.show_api_var.get():
            self.entry_api.config(show="")
        else:
            self.entry_api.config(show="*")

    def select_directory(self):
        self.pdfs_dir = filedialog.askdirectory()
        if self.pdfs_dir:
            self.label_dir.config(text=f"Selected: {self.pdfs_dir}")
            self.log(f"Selected Directory: {self.pdfs_dir}", "info")
            self.run_button.config(state="disabled")
            self.tree.delete(*self.tree.get_children()) # Clear previous results
        else:
            self.label_dir.config(text="No directory selected.")
            self.log("No directory selected.", "error")

    def start_preview_thread(self):
        """Starts the AI processing in a separate thread to avoid freezing the GUI."""
        api_key = self.entry_api.get().strip()
        model_name = self.model_var.get().strip()
        naming_pattern = self.naming_pattern_entry.get().strip()
        
        if not self.pdfs_dir:
            self.log("Please select a directory first.", "error")
            return
        if not api_key:
            self.log("Please enter your API key.", "error")
            return
        if not naming_pattern:
            self.log("Please enter a naming pattern.", "error")
            return

        self.preview_button.config(state="disabled")
        self.run_button.config(state="disabled")
        self.tree.delete(*self.tree.get_children()) # Clear previous preview
        self.progress['value'] = 0
        self.log("Starting AI renaming process... This may take a while.", "info")

        # Run the AI logic in a thread
        thread = threading.Thread(
            target=self.generate_previews,
            args=(api_key, model_name, self.pdfs_dir, naming_pattern)
        )
        thread.daemon = True
        thread.start()

    def generate_previews(self, api_key, model_name, pdfs_dir, naming_pattern):
        """Worker function that calls the AI logic and updates the UI from the thread."""
        self.proposed_renames = ai_renamer_logic(
            api_key, model_name, pdfs_dir, naming_pattern,
            lambda msg, tag: self.root.after(0, self.log, msg, tag),
            lambda value: self.root.after(0, self.update_progress, value)
        )
        
        # Schedule UI updates to run on the main thread
        self.root.after(0, self.populate_preview_table)

    def populate_preview_table(self):
        """Populates the preview table with the results from the AI."""
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for original, new in self.proposed_renames:
            self.tree.insert("", "end", values=(original, new))

        self.log("Preview generation complete. Review the changes above.", "success")
        self.preview_button.config(state="normal")
        if any("Error" not in new for _, new in self.proposed_renames):
            self.run_button.config(state="normal") # Enable renaming if there are valid proposals

    def update_progress(self, value):
        self.progress['value'] = value

    def run_renamer(self):
        """Executes the actual file renaming based on the previewed changes."""
        if not self.proposed_renames:
            self.log("No files to rename. Please generate a preview first.", "error")
            return
        
        self.log("Starting file renaming...", "info")
        for original_name, new_name in self.proposed_renames:
            if "Error" in new_name:
                self.log(f"Skipping {original_name} due to previous error.", "error")
                continue
            
            original_path = os.path.join(self.pdfs_dir, original_name)
            new_path = os.path.join(self.pdfs_dir, new_name)
            
            try:
                if os.path.exists(new_path):
                    base, ext = os.path.splitext(new_name)
                    counter = 1
                    while os.path.exists(new_path):
                        new_name_inc = f"{base}_{counter}{ext}"
                        new_path = os.path.join(self.pdfs_dir, new_name_inc)
                        counter += 1
                    self.log(f"File '{new_name}' already exists. Renaming to '{new_name_inc}'.", "info")
                
                os.rename(original_path, new_path)
                self.log(f"Successfully renamed '{original_name}' to '{os.path.basename(new_path)}'", "success")
            except Exception as e:
                self.log(f"Failed to rename '{original_name}': {e}", "error")

        self.log("Renaming process completed!", "success")
        self.run_button.config(state="disabled")

    def log(self, message, tag="info"):
        """Logs a message to the text widget with a specified color tag."""
        self.log_text.insert('end', message + '\n', tag)
        self.log_text.see('end')

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()