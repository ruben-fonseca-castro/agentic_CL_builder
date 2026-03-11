# Agentic Context-Aware Cover Letter Builder

My first agent project, which involved the development and refinement of an agentic framework and workflow using Langchain and Langgraph
that **end-to-end takes a url of a specific job posting, and outputs a rendered, polished, and formatted pdf of a cover letter for that
specific job posting**. The framework scrapes the url for the raw job contents, searches the web for company information and cultural/idealistics values, formats and extracts key information for downstream tasks, and also prepares 2-3 tailored experiences from personal context to use
when writing. The writing agent then takes in all of this information, along with a writing sample in a personal style, and creates a 
formatted letter that achieves primary objectives, relates to personal context, is relevant to the position, all while maintaining the original
writing style provided. This then gets converted to a pdf under a predefined css style, and eventually exported to output_letters.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ruben-fonseca-castro/agentic_CL_builder.git
   cd agentic_CL_builder
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   This project can be installed using `uv`:
   ```bash
   uv pip install -r requirements.txt
   ```
   *(Note: You may also need to run `playwright install` to initialize browsers for `crawl4ai`).*

4. **Environment Variables:**
   Create a `.env` file in the root directory and add necessary API keys for the LLM and search tool:
   ```env
   ANTHROPIC_API_KEY=your_anthropic_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
   *Make sure these are loaded into your environment.*

5. **Personal Context Setup:**
   The script relies on your personal context to tailor the letter:
   - **Resume Database:** Update `./personal_context/context.md` with your own resume or list of experiences. Maintain the Markdown header format `## Project: ...` so the splitter can chunk it correctly (`Use either Project: or Experience:`, followed by a title of the revevant entry). Right under, include a `Details: ...` section that details a given project, followed by a new line separating different entries.
   - **Writing Sample:** Update `./personal_context/samples/sample_cover_letter.txt` with a template of your cover letter writing style, using bracketed placeholders like `[COMPANY NAME]` for the LLM to understand style and structure. Ensure that the template itself originally fits in a 1 page pdf format. 

## How to Run

1. **Execute the Agentic Builder:**
   Start the application using Python:
   ```bash
   python agentic_cl_builder.py
   ```

2. **First-time Initialization:**
   On the very first run, the script will automatically:
   - Download the BGE embeddings model (`BAAI/bge-m3`, ~2GB).
   - Chunk your personal context markdown and initialize the Chroma vector database in the `./chroma_db` folder.
   - NOTE: If you would like to update your context, please update the `context.md` file, then delete the `chroma_db` directory so that builder can regenerate the database.

3. **Provide the Job URL:**
   Once initialized, the CLI will prompt you:
   ```text
   Enter the job posting URL: 
   ```
   Paste the link to the targeted job description and press **Enter**.

4. **Wait for Agent Generation:**
   The LangGraph agent will show a progress spinner while it:
   - Scrapes the job description using *crawl4ai*.
   - Researches the company's core values using *Tavily*.
   - Retrieves your most relevant personal experiences from the vector database.
   - Drafts the cover letter using *Anthropic's Claude* mimicking your writing style.
   - Converts the finalized draft into a formatted PDF.

5. **Retrieve the Output:**
   Upon successful completion, your tailored cover letter will be saved as a styled PDF in the `./output_letters/` directory. You can type `q`, `quit`, or press `Ctrl+C` to exit the builder.
