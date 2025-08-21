# PDF Renamer Master

<img src="./an_AI_robot_organizes_documents.png" alt="Icon" width="250"/>

A desktop application that automatically renames PDF files based on their content using large language models.

## Features

*   Supports multiple AI models from Google and OpenAI.
*   Simple and intuitive graphical user interface.
*   Process a directory of PDF files in batch.
*   View a log of the renaming process.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/PDF-Renamer-Master.git
    cd PDF-Renamer-Master
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python app_v6.py
    ```

## Usage

1.  **Get an API Key:**
    *   You will need an API key from a supported AI provider (e.g., Google AI, OpenAI).

2.  **Using the Application:**
    *   Launch the application by running `python app_v6.py`.
    *   Enter your API key in the "API Key" field.
    *   Select the AI model you want to use from the dropdown menu.
    *   Click the "Select Directory" button to choose the folder containing your PDF files.
    *   Click the "Run Renamer" button to start the process.
    *   The application will display the progress in the log window.

## Supported Models

The following AI models are available:

*   `gemini-2.5-flash`
*   `gtp-o4-mini`

## Dependencies

This project relies on the following Python libraries:

* pydantic
* langchain_core
* fastapi
* langchain
* langchain-community
* langchain-text-splitters
* langchain-google-genai
* langchain-openai
* langchain-anthropic
* langchain-mistralai
* pypdf

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
