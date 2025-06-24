import os
import tkinter as tk
from tkinter import (
    Tk, Label, Button, filedialog, Text, Scrollbar, Frame, Entry, BOTH, RIGHT,
    Y, LEFT, OptionMenu, StringVar
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pypdf.errors import PdfReadError, PdfStreamError


def get_pdf_files(pdfs_dir):
    return [f for f in os.listdir(pdfs_dir) if f.lower().endswith('.pdf')]

def sanitize_filename(filename):
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()

def ai_reader(api, model, pdfs_dir, log_func):
    files = get_pdf_files(pdfs_dir)
    if not files:
        log_func("No PDF files found in the selected directory.")
        return

    # Initialize the language model
    if 'gemini' in model:
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api
        )
    elif 'gpt' in model:
        llm = ChatOpenAI(
            temperature=0,
            model_name=model,
            api_key=api
        )
    elif 'claude' in model:
        llm = ChatAnthropic(
            temperature=0,
            model=model,
            anthropic_api_key=api
        )
    elif 'mistral' in model:
        llm = ChatMistralAI(
            temperature=0,
            model=model,
            mistral_api_key=api
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    user_input = (
        "Based on the provided file, generate a file name in this format: "
        "[title of the research]_[published year].pdf"
        "Please do not give any response except for the file name. "
        "Do not include symbols like /, ~, !, @, #, or $ in the file name."
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", 'You are a helpful assistant. Use the following context when responding:\n\n{context}.'),
        ("human", "{question}")
    ])

    output_parser = StrOutputParser()
    rag_chain = rag_prompt | llm | output_parser

    for f in files:
        pdf_path = os.path.join(pdfs_dir, f)
        log_func(f"Processing file: {pdf_path}")

        try:
            loader = PyPDFLoader(file_path=pdf_path)
            documents = loader.load_and_split()
            context = " ".join(page.page_content for page in documents)
            if len(context) > 128000:
                context = context[:128000]
            response = rag_chain.invoke({
                "question": user_input,
                "context": context
            })
            new_file_name = sanitize_filename(response.strip())
            if not new_file_name.lower().endswith('.pdf'):
                new_file_name += '.pdf'
            new_file_path = os.path.join(pdfs_dir, new_file_name)
            if os.path.exists(new_file_path):
                base_name, extension = os.path.splitext(new_file_name)
                counter = 1
                while os.path.exists(new_file_path):
                    new_file_name = f"{base_name}_{counter}{extension}"
                    new_file_path = os.path.join(pdfs_dir, new_file_name)
                    counter += 1
            os.rename(pdf_path, new_file_path)
            log_func(f"Renamed to: {new_file_name}")

        except (PdfReadError, PdfStreamError) as e:
            log_func(f"Error reading PDF {pdf_path}: {e}")
            continue  # Skip this file and proceed to the next one
        except Exception as e:
            log_func(f"An unexpected error occurred with {pdf_path}: {e}")
            continue
                
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Renamer")
        self.root.geometry("600x500")

        api_frame = Frame(root)
        api_frame.pack(pady=10)
        self.label_api = Label(api_frame, text="API Key:", font=("Helvetica", 12))
        self.label_api.grid(row=0, column=0, padx=5)
        self.entry_api = Entry(api_frame, width=40)
        self.entry_api.grid(row=0, column=1, padx=5)

        model_frame = Frame(root)
        model_frame.pack(pady=10)
        self.label_model = Label(model_frame, text="Select Model:", font=("Helvetica", 12))
        self.label_model.grid(row=0, column=0, padx=5)
        self.model_var = StringVar(model_frame)
        self.model_var.set("gemini-2.0-flash-lite") #gemini-1.5-flash

        models = [
            "gpt-4o", "gpt-4o-mini", "gemini-1.5-pro", "emini-2.0-flash-lite", "gemini-1.5-flash", 
            "open-mistral-nemo-2407", "mistral-large-2407", "claude-3-5-sonnet-20240620"
        ]
        self.model_menu = OptionMenu(model_frame, self.model_var, *models)
        self.model_menu.grid(row=0, column=1, padx=5)

        dir_frame = Frame(root)
        dir_frame.pack(pady=10)
        self.label_dir = Label(dir_frame, text="Select PDF Directory:", font=("Helvetica", 12))
        self.label_dir.grid(row=0, column=0, padx=5)
        self.select_button = Button(dir_frame, text="Select Directory", command=self.select_directory, font=("Helvetica", 10))
        self.select_button.grid(row=0, column=1, padx=5)

        run_frame = Frame(root)
        run_frame.pack(pady=10)
        self.run_button = Button(
            run_frame, text="Run Renamer", command=self.run_renamer,
            font=("Helvetica", 10), bg="#4CAF50", fg="white"
        )
        self.run_button.pack()

        log_frame = Frame(root)
        log_frame.pack(pady=10, fill=BOTH, expand=True)
        self.log_text = Text(
            log_frame, height=15, width=70, wrap='word',
            font=("Helvetica", 10)
        )
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar = Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        self.pdfs_dir = ''  # Initialize pdfs_dir

    def select_directory(self):
        self.pdfs_dir = filedialog.askdirectory()
        if self.pdfs_dir:
            self.log(f"Selected Directory: {self.pdfs_dir}")
        else:
            self.log("No directory selected.")

    def run_renamer(self):
        if self.pdfs_dir:
            os.chdir(self.pdfs_dir)
            api_key = self.entry_api.get().strip()
            model_name = self.model_var.get().strip()
            if not api_key:
                self.log("Please enter your API key.")
                return
            try:
                ai_reader(api_key, model_name, self.pdfs_dir, self.log)
                self.log("Renaming Completed")
            except Exception as e:
                self.log(f"An error occurred: {e}")
        else:
            self.log("Please select a directory first.")

    def log(self, message):
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
