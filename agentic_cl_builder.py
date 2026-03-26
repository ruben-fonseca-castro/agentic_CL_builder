### Main agentic script used to run the agentic cover letter builder
## Ruben Fonseca

## Imports ##

import warnings
import logging

# 1. Broadly ignore the dependency warnings from requests
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
# 2. Specifically target the message if the category filter fails
warnings.filterwarnings("ignore", message=".*urllib3.*")
# 3. Optional: Silence the logger if it's being sent to logging instead of stdout
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import Tool, tool
from langchain_tavily import TavilySearch
from crawl4ai import AsyncWebCrawler
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import List
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage
from markdown_pdf import MarkdownPdf, Section
from langgraph.graph import StateGraph, START, END
from datetime import datetime
import asyncio
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, TextColumn, SpinnerColumn

## Main Function ##

async def main():

    print("Initializing Agentic Cover Letter Builder... \n")

    ## Initialize Model ##

    print("Initializing Model... \n")

    model = init_chat_model("claude-haiku-4-5-20251001") # change to sonnet once stable

    ## Initialize the state class ##

    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage],operator.add]
        llm_calls: int
        job_url: str
        job_desc: str
        company_research: str
        resume_context: List[str]
        writing_sample: str
        selected_experiences: List[str]
        draft_letter: str
        critique: str
        iteration_count: int
        exception_triggered: int
        current_date: str
        cover_letter_name: str

    ## RAG Database Setup ##

    print("Initializing RAG Database... \n")

    # persisent directory of the database
    persist_directory = "./chroma_db"
    source_file = "./personal_context/context.md" # location of the personal context file (markdown labeled)
    
    # Initialize BGE-M3

    # This will download the model (~2GB) on the first run
    model_name = "BAAI/bge-m3"
    model_kwargs = {'device': 'cpu'} # Change to 'cuda' if you have a GPU
    encode_kwargs = {'normalize_embeddings': True}

    bge_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )

    # Check if DB exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("--- Loading existing Vector Store ---")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=bge_embeddings
        )
    else:
        print("--- Creating new Vector Store ---")
        with open(source_file, "r") as f:
            markdown_document = f.read()
        
        headers_to_split_on = [("##", "Experience_Name")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_document)
        
        vectorstore = Chroma.from_documents(
            documents=md_header_splits, 
            embedding=bge_embeddings,
            persist_directory=persist_directory
        )

    # Initialize the retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15,
            "lambda_mult": 0.5
        }
    )

    ## Node Definitions ##

    print("Initializing Node Definitions... \n")

    # Define tools

    tavily_tool = TavilySearch(
    max_results=3,
    )

    @tool
    async def web_crawler_tool(url: str) -> str:
        """
        Crawls a website and returns its clean markdown content. 
        Use this for extracting detailed information from job postings or company pages.
        """
        async with AsyncWebCrawler(verbose=False) as crawler:
            # arun fetches the page and converts it to LLM-ready markdown
            result = await crawler.arun(url=url)
            
            if result.success:
                return result.markdown
            else:
                return f"Failed to crawl {url}: {result.error_message}"

            
    # Augment the LLM with tools

    tools = [tavily_tool, web_crawler_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)

    ## Define researcher node ##

    async def llm_call(state: AgentState):
        """LLM decides whether to call a tool or not"""

        prompt = [
            SystemMessage(
                content="""You are a helpful researcher tasked with using your tools to take in a given request, and 
                perform the necessary tool calls to fill out the company research and job description fields in the State. 
                Take the URL that is given to you, and perform a Crawl scrape on it, which gets saved to job_desc. Based on 
                whatever company that job description is for (ensure that you have the correct company name, not a subteam or 
                division of the company), then perform a Tavily search to find out general company values, current company 
                headquarters location (city and state), mission statements, and beliefs that characterize the work and style of that company, 
                this will then be saved into company_research. You do not need to know the specific information extracted from 
                either tool, that is handled and saved to the state by the tool itself. Proceed once both tasks are complete. Do 
                not summarize or discuss what was extracted, simply call the necessary tools, and move on."""
            )
        ] + state["messages"]

        # Use ainvoke for the async call
        response = await model_with_tools.ainvoke(prompt)

        return {
            "messages": [response],
            "llm_calls": state.get('llm_calls', 0) + 1
        }

    ## Define tool node ##

    async def tool_node(state: AgentState):
        """Unified tool node that extracts and summarizes results directly into the state."""
        last_message = state["messages"][-1]
        tool_messages = []
        
        # Use local variables to build the update dictionary
        updated_company_research = state.get("company_research", "")
        updated_job_desc = state.get("job_desc", "")
        updated_job_url = state.get("job_url", "")

        # --- THE FIX: Initialize variable from state with a safe default ---
        cover_letter_name = state.get("cover_letter_name", "Generated_Cover_Letter.pdf")



        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            
            # 1. HANDLE COMPANY RESEARCH (TAVILY)
            if "tavily" in tool_name.lower():
                query = tool_call["args"].get("query", "")
                raw_result = await tavily_tool.ainvoke({"query": query})
                
                summary_prompt = f"Summarize the company values, mission, and location information from this research in 3-5 sentences. CONTENT:{raw_result}"
                summary = await model.ainvoke(summary_prompt)
                
                updated_company_research = summary.content
                tool_messages.append(ToolMessage(
                    content="Company research summarized and saved.", 
                    tool_call_id=tool_call["id"]
                ))

            # 2. HANDLE JOB DESCRIPTION (CRAWL4AI)
            elif tool_name == "web_crawler_tool":

                url = tool_call["args"].get("url", "")
                raw_markdown = await web_crawler_tool.ainvoke({"url": url})
                
                summary_prompt = f"""
                    Extract only the company name, job title, job responsibilities, qualifications, and requirements from the following CONTENT.
                    Ignore navigation menus, footer links, and application form fields (dropdowns, questions).
                    
                    CONTENT: {raw_markdown}
                    """
                summary = await model.ainvoke(summary_prompt)

                # TO DO: PROMPT AGAIN TO GENERATE AN APPROPRIATELY FORMATTED COVER LETTER NAME SO THAT IT USES IT UPON OUTPUT!!!!!
                pdf_name_prompt = f"""
                Based on the following job description CONTENT, create a title for a cover letter under the following format.
                Do not deviate from the following format, and do not include any other explanative or confirming text. Output
                ONLY the correctly formatted string:

                STRING FORMAT: "[COMPANY_NAME]_[JOB_TITLE]_Cover_Letter.pdf"

                CRITICAL INSTRUCTIONS:
                1. Output ONLY the correctly formatted string.
                2. Do NOT include any conversational text, apologies, or explanations.
                3. If the CONTENT is empty or you cannot find the company/job title, output EXACTLY: "Generated_Cover_Letter.pdf"

                CONTENT: {raw_markdown}
                """

                pdf_name_result = await model.ainvoke(pdf_name_prompt)
                
                updated_job_desc = summary.content
                cover_letter_name = pdf_name_result.content
                updated_job_url = url
                tool_messages.append(ToolMessage(
                    content="Job description summarized and saved.", 
                    tool_call_id=tool_call["id"]
                ))
                
            # 3. SAFETY FALLBACK (Prevents the 400 error)
            else:
                print("\n Entering Else Statement!")
                tool_messages.append(ToolMessage(
                    content="Tool executed but result not summarized.", 
                    tool_call_id=tool_call["id"]
                ))

        return {
            "company_research": updated_company_research,
            "job_desc": updated_job_desc,
            "job_url": updated_job_url,
            "cover_letter_name": cover_letter_name,
            "messages": tool_messages
        }

    ## Define context engine node ##

    async def context_engine_node(state: AgentState):
        """
        Combined node: Retrieves context from RAG & MCP, then selects the best 
        experiences to highlight based on the job description.
        """
        job_requirements = state["job_desc"]
        
        # --- STEP 1: RAG RETRIEVAL ---

        query_prompt = f"""
            Extract the most critical technical requirements from this job description to use as a dense database search query.
            
            You must return exactly 12 to 20 highly specific technical keywords or short phrases. 
            Focus heavily on identifying hard skills across the relevant engineering disciplines (Mechanical, Software, Electrical, Robotics, etc.):
            1. Software & Tools: Programming languages, CAD/design software, and simulation tools (e.g., Python, SolidWorks, MATLAB, Altium).
            2. Frameworks & Environments: Libraries, operating systems, and middleware (e.g., ROS, Linux, RTOS, PyTorch).
            3. Core Engineering & Analytical Skills: Methodologies, mathematical models, and domain expertise (e.g., Finite Element Analysis, control theory, thermodynamics, machine learning).
            4. Hardware & Physical Systems: Manufacturing processes, testing equipment, circuitry, and physical components (e.g., CNC machining, PCB layout, CAN bus, sensors).
            
            IGNORE soft skills (e.g., "team player", "cross-functional communication"), generic company culture/perks, and basic educational requirements.
            
            Return ONLY a single comma-separated list of these 12-20 technical terms. Do not include bullet points, category labels, or any other text.
            
            Job Description: {job_requirements}
            """

        keyword_response = await model.ainvoke(query_prompt)
        dense_query = keyword_response.content

        # Find experiences in your resumes that match the job description
        relevant_docs = await retriever.ainvoke(dense_query)

        # FORCE UNIQUENESS: Use a set to track content we've already seen
        unique_content = []
        seen = set()
        for doc in relevant_docs:
            # Use a hash or just the text to check for duplicates
            content = doc.page_content.strip()
            if content not in seen:
                unique_content.append(content)
                seen.add(content)
        
        # Ensure we only present the unique ones to the LLM
        raw_blocks = unique_content[:5]
        resume_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                
        # --- STEP 2: EXTRACT SAMPLE LETTER ---

        try:
            sample_path = os.path.join("./personal_context/samples", "sample_cover_letter.txt")
            with open(sample_path, "r") as f:
                writing_style = f.read()
        except Exception as e:
            writing_style = f"Fallback style: Professional and technical. Error: {str(e)}"

        # --- STEP 3: EXACT EXPERIENCE SELECTION ---
        # Create an enumerated list for the LLM to choose from
        formatted_options = "\n\n".join([f"--- ID: {i} ---\n{content}" for i, content in enumerate(raw_blocks)])

        selection_prompt = f"""
            You are a Career Strategist.
            
            TARGET JOB REQUIREMENTS:
            {job_requirements}
            
            AVAILABLE PERSONAL EXPERIENCES:
            {formatted_options}
            
            TASK:
            Identify the IDs of the 2-3 most relevant experiences and projects that best match and are closely 
            related to the job requirements and skills that the job is asking for. 
            Return ONLY the IDs as a comma-separated list (e.g., 0, 2). Do not include any text.
            """

        id_response = await model.ainvoke(selection_prompt)

        try:
            # Parse the IDs and strip any whitespace/extra characters
            selected_ids = [int(i.strip()) for i in id_response.content.split(",") if i.strip().isdigit()]
            
            # Map the IDs back to the EXACT raw text from RAG
            selected_experiences = [raw_blocks[i] for i in selected_ids if i < len(raw_blocks)]
            exception_trig = 0
        except Exception as e:
            # Fallback: if LLM fails formatting, take the first 2 blocks
            print("LLM Failed formatting!!")
            exception_trig = 1
            selected_experiences = raw_blocks[:2]


        return {
            "resume_context": [resume_text],
            "writing_sample": writing_style,
            "selected_experiences": selected_experiences,
            "exception_triggered": exception_trig,
            "messages": [AIMessage(content=f"Strategist: Selected {len(selected_experiences)} key experiences.")]
        }

    ## Define cover letter writer node ##

    async def cl_writer_node(state: AgentState):
        """Few shot prompt node that takes in a bunch of information from the state, 
        and returns an initial draft of the cover letter in my writing style, using
        the selected experiences from the context engine, and writing sample from the 
        filesystem. 
        """
        job_desc = state["job_desc"]
        company_research = state["company_research"] 
        resume_context = state["resume_context"]
        style_sample = state["writing_sample"]
        experiences = "\n\n".join(state["selected_experiences"])
        current_date = state["current_date"]


        writer_prompt = f"""
        You are Ruben Fonseca, a robotics engineer with an MS from UMich and a BS from Harvard. 
        Your task is to write a highly tailored cover letter for a given job description/posting
        based on the resume context, selected experiences, and writing sample in Ruben's style. You
        are to mimic his writing style, and copy the overall paragraph structure of the style sample.
        The style sample includes [] brackets with descriptive information that must be filled it based
        on the information provided to you. For the first and fourth body paragraphs, besides those bracketed fill in
        sections, do not change the wording/sentences. For the second and third body paragraphs, treat the style sample as
        purely a style sample, and instead replace them with similarly lengthed paragraphs that discuss the 2-3 specific
        experiences that are highlighted, and tie them to the job requirements and/or overall company values as best as possible,
        again using Ruben's writing style.

        STRICT STYLE GUIDE (Mimic this):
        {style_sample}

        TARGET JOB REQUIREMENTS:
        {job_desc}

        TARGET COMPANY RESEARCH (Incorporate these values):
        {company_research}

        SPECIFIC EXPERIENCES TO HIGHLIGHT (Use these exact technical details and do not extrapolate):
        {experiences}

        INSTRUCTIONS:
        1. Match the technical depth and professional yet accessible tone of the style sample.
        2. Bridge the 'Specific Experiences' directly to the 'Job Requirements' and/or 'Company Research.'
        3. Keep it to one page (approx 300-400 words), as in stay around the same length as the style sample.
        4. Maintain the style sample's paragraph structure, and the preamble with date and company name.
        5. Use '{current_date}' as the date in the cover letter header.
        6. ONLY include the letter content. Do NOT include any additional text, or confirmation messages of the task being complete.

        Based on the information provided and further instructions above, output the final cover letter draft in this 
        exact format:

        OUTPUT STRUCTURE TEMPLATE (YOU MUST FOLLOW THIS EXACTLY, DON"T FORGET TO INCLUDE THE # MARKDOWN ELEMENTS):
        # [today's  date]
        # [Company Name]
        # [Company City, State]
        
        ## Dear [COMPANY NAME] Recruitment Team,

        [Paragraph 1: reference the writing sample for specific content formatting]

        [Paragraph 2: reference the writing sample for specific content formatting]

        [Paragraph 3: reference the writing sample for specific content formatting]

        [Paragraph 4: reference the writing sample for specific content formatting]
        
        ### Sincerely,
        ### Ruben Fonseca
        """

        cover_letter = await model.ainvoke(writer_prompt)

        return {
            "draft_letter": cover_letter.content,
            "messages": [AIMessage(content=f"Cover Letter Draft created and saved to appropriate state.")]
        }

    ## Define pdf generator node ##

    async def pdf_generator_node(state: AgentState):
        """
        Converts a cover letter to a PDF file.
        """

        output_dir = "./output_letters"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Format the letter
        draft = state['draft_letter']

        css_style = """
            @page { margin: 1in; }
            body {
                font-family: "Times New Roman", Times, serif;
                font-size: 12pt;
                line-height: 1.4;
                text-align: justify;
            }
            
            /* 1. Address Block (h1) */
            h1 {
                font-size: 12pt;
                font-weight: normal;
                margin: 0; /* Keeps the address lines tight */
                padding: 0;
                text-indent: 0px !important;
                text-align: left;
            }
            
            /* 2. Salutation (h2) */
            h2 {
                font-size: 12pt;
                font-weight: normal;
                margin-top: 20px; /* GAP BETWEEN ADDRESS AND DEAR */
                margin-bottom: 10px; /* Gap before the first paragraph */
                text-indent: 0px !important;
                text-align: left;
            }
            
            /* 3. Body Paragraphs */
            p {
                text-indent: 35px; /* Only indents the actual body */
                margin-top: 8px;
                margin-bottom: 0px;
            }
            
            /* 4. Signature Block (h3) */
            h3 {
                font-size: 12pt;
                font-weight: normal;
                margin: 0; /* Keeps 'Sincerely' and your Name tight */
                text-indent: 0px !important;
                text-align: left;
            }
            h3:first-of-type {
                margin-top: 25px; /* GAP BETWEEN LAST PARAGRAPH AND SINCERELY */
            }
        """

        # 4. Initialize PDF Generator
        pdf = MarkdownPdf(toc_level=0)
        pdf.add_section(
            Section(draft), 
            user_css=css_style # Pass CSS here, NOT in Section()
        )

        # 5. Save the PDF
        file_name = state['cover_letter_name']
        # file_name = f"Cover_Letter_test.pdf"
        file_path = os.path.join(output_dir, file_name)
        pdf.save(file_path)

        return {
            "messages": [AIMessage(content=f"Cover Letter converted and saved as PDF. Objective Completed.")]
        }

    ## Build LangGraph state graph ##

    print("Building LangGraph state graph... \n")

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return "context_engine"
        return "tool_node"

    agent_builder = StateGraph(AgentState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)  # Single unified node
    agent_builder.add_node("context_engine",context_engine_node) # with RAG and FileSystem MCP
    agent_builder.add_node("cl_writer",cl_writer_node)
    agent_builder.add_node("pdf_generator",pdf_generator_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", "context_engine"]
    )
    agent_builder.add_edge("tool_node", "llm_call")
    agent_builder.add_edge("context_engine","cl_writer")
    agent_builder.add_edge("cl_writer","pdf_generator")
    agent_builder.add_edge("pdf_generator",END)

    agent = agent_builder.compile()

    # Setup complete, moving onto main loop

    # Mapping technical node names to professional status updates

    # checklist function
    # def generate_checklist(node_statuses):
    #     """Creates a table showing the progress of each agent node."""
    #     table = Table.grid(padding=(0, 1))
    #     table.add_column("Status", justify="center", width=3)
    #     table.add_column("Task", justify="left")

    #     for task, status in node_statuses.items():
    #         if status == "working":
    #             icon = "[bold blue]⚙[/bold blue]"
    #             text = f"[bold blue]{task}...[/bold blue]"
    #         elif status == "done":
    #             icon = "[bold green]✔[/bold green]"
    #             text = f"[green]{task}[/green]"
    #         else:
    #             icon = "[dim]○[/dim]"
    #             text = f"[dim]{task}[/dim]"
            
    #         table.add_row(icon, text)
        
    #     return Panel(table, title="[bold]Build Progress[/bold]", border_style="blue", expand=False)

    
    task_map = {
        "llm_call": "🔍 Researcher: Analyzing job requirements",
        "tool_node": "🌐 Researcher: Scraping job site and company data",
        "context_engine": "🧠 Strategist: Selecting best project matches",
        "cl_writer": "✍️ Writer: Drafting your tailored cover letter",
        "pdf_generator": "📄 Generator: Formatting and saving PDF"
    }


    console = Console()
    console.print("\n[bold green]✓ Agentic Cover Letter Builder Initialized![/bold green]")
    console.print("[dim]Type 'q', 'quit', or press Ctrl+C to exit.[/dim]\n")

    while True:
        try:
            # 1. Get user input
            url = console.input("[bold cyan]Enter the job posting URL:[/bold cyan] ").strip()
            
            # 2. Check for exit commands
            if url.lower() in ['q', 'quit', 'exit']:
                console.print("[yellow]Exiting cover letter builder. Goodbye![/yellow]")
                break
            
            if not url:
                continue

            console.print("\n" * 6)

            # 3. Setup the initial state 
            today = datetime.now().strftime("%B %d, %Y")
            request = [HumanMessage(content=f"Scrape the listing from this website: {url}. Also extract the company information of that listing")]
            
            initial_state = {
                "messages": request,
                "current_date": today  
            }

            # # node status for checklist display in CLI
            # node_statuses = {
            #     "🔍 Researcher: Analyzing job requirements": "pending",
            #     "🌐 Researcher: Scraping job site and company data": "pending",
            #     "🧠 Strategist: Selecting best project matches": "pending",
            #     "✍️ Writer: Drafting your tailored cover letter": "pending",
            #     "📄 Generator: Formatting and saving PDF": "pending"
            # }
            
            current_cover_letter_name = "Cover_Letter_Fallback.pdf"
            

            # 4. Execute the graph inside a rich status spinner
            # with console.status("[bold blue]Agent is researching, strategizing, and writing...[/bold blue]", spinner="dots"):
            #     final_state = await agent.ainvoke(initial_state)

            # with console.status("[bold blue]Initializing agent...[/bold blue]", spinner="dots") as status:
            #     # Use astream to get updates as each node finishes
            #     async for event in agent.astream(initial_state, stream_mode="updates"):
            #         # 'event' is a dict: { "node_name": { "state_key": value } }
            #         for node_name in event.keys():
            #             if node_name in NODE_LABELS:
            #                 # Dynamically update the spinner text based on the active node
            #                 status.update(f"[bold blue]{NODE_LABELS[node_name]}[/bold blue]")


            # # 2. Use Live to keep the checklist on screen
            # with Live(generate_checklist(node_statuses), console=console, auto_refresh=False) as live:
            #     async for event in agent.astream(initial_state, stream_mode="updates"):
            #         # event looks like: {"node_name": {"state_key": value}}
            #         for node_name, node_update in event.items():
                        
            #             # --- FILENAME CATCHER ---
            #             # If the tool_node updates the filename, we catch it here
            #             if "cover_letter_name" in node_update:
            #                 current_cover_letter_name = node_update["cover_letter_name"]

            #             # --- UI UPDATER ---
            #             task_name = task_map.get(node_name)
            #             if task_name:
            #                 # Mark any previously 'working' tasks as 'done'
            #                 for k in node_statuses:
            #                     if node_statuses[k] == "working":
            #                         node_statuses[k] = "done"
                            
            #                 # Set current task to working
            #                 node_statuses[task_name] = "working"
                            
            #             # Re-render the table
            #             live.update(generate_checklist(node_statuses), refresh=True)
                
            #     # Final set: Ensure the last task is marked done at the end
            #     for k in node_statuses:
            #         node_statuses[k] = "done"
            #     live.update(generate_checklist(node_statuses), refresh=True)

            #     final_state = event


            # 1. Initialize Progress with just Text, turning off transient so it stays on screen
            with Progress(
                TextColumn("{task.description}"),
                console=console,
                transient=False,
                redirect_stdout=True,
                redirect_stderr=True 
            ) as progress:
                
                # 2. Add all tasks upfront so the whole list is visible immediately
                task_ids = {}
                for node_key, task_name in task_map.items():
                    # Initialize them with a dim empty circle
                    task_ids[node_key] = progress.add_task(f"[dim]○ {task_name}[/dim]", total=None)

                active_node = None

                completed_nodes = set()

                async for event in agent.astream(initial_state, stream_mode="updates"):
                    for node_name, node_update in event.items():
                        
                        if "cover_letter_name" in node_update:
                            current_cover_letter_name = node_update["cover_letter_name"]

                        if node_name in task_map:
                            # 3. Mark the PREVIOUS task as done (green check)
                            if active_node and active_node != node_name:
                                completed_nodes.add(active_node)
                                progress.update(
                                    task_ids[active_node], 
                                    description=f"[bold green]✔ {task_map[active_node]}[/bold green]"
                                )

                            if node_name not in completed_nodes:
                                active_node = node_name
                                progress.update(
                                    task_ids[active_node], 
                                    description=f"[bold blue]⚙ {task_map[active_node]}...[/bold blue]"
                                )

                            # # 4. Set the CURRENT task to working (blue gear)
                            # active_node = node_name
                            # progress.update(
                            #     task_ids[active_node], 
                            #     description=f"[bold blue]⚙ {task_map[active_node]}...[/bold blue]"
                            # )

                # 5. When the loop finishes, mark the final task as done
                if active_node:
                    progress.update(
                        task_ids[active_node], 
                        description=f"[bold green]✔ {task_map[active_node]}[/bold green]"
                    )

                # final_state = event


            # 5. Extract the generated filename and print success
            # output_name = final_state.get('cover_letter_name', 'Cover_Letter.pdf')
            console.print(f"[bold green]✓ Process Complete![/bold green] Cover letter saved as [bold white]{current_cover_letter_name}[/bold white] in ./output_letters\n")

        except KeyboardInterrupt:
            # Handles Ctrl+C gracefully
            console.print("\n[yellow]Process interrupted by user. Exiting...[/yellow]")
            break
        except Exception as e:
            # Catches and prints any workflow errors without crashing the whole loop
            console.print(f"[bold red]An error occurred during generation:[/bold red] {e}\n")


if __name__ == "__main__":
    asyncio.run(main())