import os
from tkinter import Tk, Label, Button, filedialog, Text, Scrollbar, Frame, Entry, BOTH, RIGHT, Y, LEFT, OptionMenu, StringVar
import tkinter as tk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def numbering(pdfs_dir):
    origin_files = os.listdir(pdfs_dir)
    i = 0
    for file in origin_files:
        old_file_path = os.path.join(pdfs_dir, file)
        new_file_path = os.path.join(pdfs_dir, f"{str(i)}.pdf")
        print(new_file_path)
        try:
            os.rename(old_file_path, new_file_path)
            i += 1
        except FileExistsError:
            continue
    files = os.listdir(pdfs_dir)
    return files

def rename(pdfs_dir, names):
    origin_files = os.listdir(pdfs_dir)
    for old_name, new_name in zip(origin_files, names):
        old_file_path = os.path.join(pdfs_dir, old_name)
        new_file_path = os.path.join(pdfs_dir, new_name)
    try:
        os.rename(old_file_path, new_file_path)

    except FileExistsError:
        new_name = '_' + new_name
        new_file_path = os.path.join(pdfs_dir, new_name)
        os.rename(old_file_path, new_file_path)



def ai_reader(api, model, pdfs_dir):
    files = numbering(pdfs_dir)

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

    res = []
    user_input = '''Based on file provided, generate a file name in this format:[year published]_[aspect of the technology]_[main topic]_[primary application].pdf.
    Please do not give any response except for the file name. Do not include symbol like /, \, ~, !, @, #, or $ in the file name.'''

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", 'You are a helpful assistant. Use the following context when responding:\n\n{context}.'),
        ("human", "{question}")
    ])

    output_parser = StrOutputParser()
    rag_chain = rag_prompt | llm | StrOutputParser()

    for f in files:
        pdf_path = os.path.join(pdfs_dir, f)
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load_and_split()
        context = " ".join(page.page_content for page in documents)
        if len(context) > 128000:
            context = context[:128000]
        response = rag_chain.invoke({
            "question": user_input,
            "context": context
        })
        res.append(response.strip())
        print(response)

    rename(pdfs_dir, res)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Renamer")
        self.root.geometry("400x350")

        api_frame = Frame(root)
        api_frame.pack(pady=10)
        self.label_api = Label(api_frame, text="API Key:", font=("Helvetica", 12))
        self.label_api.grid(row=0, column=0, padx=5)
        self.entry_api = Entry(api_frame, width=30)
        self.entry_api.grid(row=0, column=1, padx=5)

        model_frame = Frame(root)
        model_frame.pack(pady=10)
        self.label_model = Label(model_frame, text="Select Model:", font=("Helvetica", 12))
        self.label_model.grid(row=0, column=0, padx=5)
        self.model_var = StringVar(model_frame)
        self.model_var.set("gemini-1.5-flash")

        models = [
            "gpt-4o", "gpt-4o-mini", "gemini-1.5-pro", "gemini-1.5-flash",
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
        self.run_button = Button(run_frame, text="Run Renamer", command=self.run_renamer, font=("Helvetica", 10), bg="#4CAF50", fg="white")
        self.run_button.pack()

        log_frame = Frame(root)
        log_frame.pack(pady=10, fill=BOTH, expand=True)
        self.log_text = Text(log_frame, height=10, width=50, wrap='word', font=("Helvetica", 10))
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar = Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

    def select_directory(self):
        self.pdfs_dir = filedialog.askdirectory()
        self.log("Selected Directory: " + self.pdfs_dir)

    def run_renamer(self):
        if hasattr(self, 'pdfs_dir'):
            os.chdir(self.pdfs_dir)
            ai_reader(self.entry_api.get(), self.model_var.get(), self.pdfs_dir)
            self.log("Renaming Completed")
        else:
            self.log("Please select a directory first")

    def log(self, message):
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()