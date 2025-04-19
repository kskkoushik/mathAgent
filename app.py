# main.py
import os
import chainlit as cl
import matplotlib.pyplot as plt
import tempfile # For creating temporary files for plots
import uuid # For unique filenames
import io # To handle image data in memory temporarily
import base64 # For potential future use or alternative display methods

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool # To create custom tools
import numpy as np
import pandas as pd # For data manipulation if needed in plotting
# --- Environment Variables ---
# Make sure to set these environment variables before running the script:
# export OPENAI_API_KEY="your_openai_api_key"
# export TAVILY_API_KEY="your_tavily_api_key"

# --- Constants ---
MODEL_NAME = "gpt-4o" # Or "gpt-3.5-turbo"
MEMORY_WINDOW_SIZE = 5 # Number of past interactions to remember
VECTORSTORE_DIR = "math_chroma_db" # Directory to store ChromaDB data
COLLECTION_NAME = "math_concepts"
TEMP_PLOT_DIR = "temp_plots" # Directory to temporarily store plots

# Create temporary plot directory if it doesn't exist
os.makedirs(TEMP_PLOT_DIR, exist_ok=True)

# --- RAG Setup ---
# (Identical to the previous version - Creating/Loading ChromaDB)
embeddings = OpenAIEmbeddings()
sample_docs = [
    ("Pythagorean Theorem", "In a right-angled triangle, the square of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the other two sides. Formula: a^2 + b^2 = c^2"),
    ("Quadratic Formula", "The solutions (roots) of a quadratic equation ax^2 + bx + c = 0 are given by the formula: x = [-b ± sqrt(b^2 - 4ac)] / 2a"),
    ("Derivative of x^n", "The derivative of x^n with respect to x is nx^(n-1). This is a fundamental rule in calculus."),
    ("Integral of x^n", "The indefinite integral of x^n with respect to x is (x^(n+1))/(n+1) + C, where C is the constant of integration (for n != -1)."),
    ("Definition of a Limit", "In calculus, the limit of a function describes the value that the function approaches as the input approaches some value. Limits are essential for defining continuity, derivatives, and integrals."),
    ("Basic Trigonometric Identities", "sin^2(x) + cos^2(x) = 1; tan(x) = sin(x)/cos(x); sec(x) = 1/cos(x); csc(x) = 1/sin(x); cot(x) = 1/tan(x)"),
    ("Plotting Sine Wave", "To plot a sine wave using matplotlib: import matplotlib.pyplot as plt; import numpy as np; x = np.linspace(0, 2*np.pi, 100); y = np.sin(x); plt.plot(x, y); plt.title('Sine Wave'); plt.xlabel('x'); plt.ylabel('sin(x)'); plt.grid(True); plt.show() # In a script, plt.savefig('sine_wave.png') would save it."),
]
if not os.path.exists(VECTORSTORE_DIR):
    print(f"Creating new vector store in {VECTORSTORE_DIR}")
    vectorstore = Chroma.from_texts(
        [doc[1] for doc in sample_docs],
        embedding=embeddings,
        metadatas=[{"source": doc[0]} for doc in sample_docs],
        persist_directory=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME
    )
    vectorstore.persist()
else:
    print(f"Loading vector store from {VECTORSTORE_DIR}")
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    "math_concepts_retriever",
    "Searches and returns relevant information about mathematical concepts, theorems, formulas, and definitions. Use this when asked about specific math topics or need background info.",
)

# --- Tools ---

# Initialize Tavily Search Tool
tavily_tool = TavilySearchResults(max_results=3)
tavily_tool.description = (
    "A search engine useful for finding real-time information, specific formulas not in the knowledge base, "
    "or external context relevant to the math problem. Use this for up-to-date information or broader queries. "
    "When presenting results from this tool, explicitly mention the source URLs if available in the search results." # Added instruction for sources
)

# --- NEW: Matplotlib Plotting Tool ---
def generate_plot(code: str) -> str:
    """
    Executes Python code using matplotlib to generate a plot and saves it
    to a temporary file.

    Args:
        code: A string containing Python code to generate a matplotlib plot.
              The code MUST include plt.savefig(filepath) to save the figure.

    Returns:
        A string containing the path to the saved plot image file,
        or an error message if execution fails.
    """
    # Security Note: Executing arbitrary code is risky.
    # In a production environment, consider sandboxing or safer alternatives.
    # For this example, we proceed with exec, assuming controlled input.

    # Generate a unique filename for the plot
    plot_filename = f"plot_{uuid.uuid4()}.png"
    plot_filepath = os.path.join(TEMP_PLOT_DIR, plot_filename)

    # Prepare the code: ensure matplotlib is imported and saving is handled
    full_code = f"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Add other common libraries if needed by the agent
import io
import os

# Isolate the plot generation
fig, ax = plt.subplots() # Create a figure and axes for the plot code to use

# User's code will modify 'ax' or use 'plt' directly
# Make the filepath available within the executed code's scope
_plot_filepath = "{plot_filepath}"

# --- User Provided Code Start ---
{code}
# --- User Provided Code End ---

# Ensure the plot is saved
try:
    # Check if savefig was already called in user code
    # This is tricky, so we might just always call savefig
    plt.savefig(_plot_filepath)
    plt.close(fig) # Close the figure to free memory
    _result = _plot_filepath
except Exception as e:
    # Try to capture any error during plotting or saving
    plt.close(fig) # Ensure figure is closed even on error
    _result = f"Error during plot saving: {{e}}"

# The result needs to be captured from the exec scope
# We'll rely on the _result variable set within the exec'd code.
"""
    local_namespace = {}
    try:
        # Execute the code
        exec(full_code, {'plt': plt, 'np': np, 'pd': pd}, local_namespace) # Pass common libraries

        # Retrieve the result (filepath or error message)
        execution_result = local_namespace.get('_result', "Error: Plot saving did not complete.")

        # Check if the file was actually created (if no error message)
        if execution_result == plot_filepath and not os.path.exists(plot_filepath):
             return f"Error: Code executed but plot file '{plot_filepath}' was not created. Make sure the code generates a plot."
        elif "Error:" in execution_result:
             return execution_result # Return error message from exec scope
        else:
             return plot_filepath # Return the successful path

    except Exception as e:
        plt.close() # Ensure any dangling plots are closed
        # More specific error handling can be added here
        # Clean up the potentially empty file if creation started but failed
        if os.path.exists(plot_filepath):
             os.remove(plot_filepath)
        return f"Error executing plotting code: {e}"


matplotlib_tool = Tool(
    name="matplotlib_plotter",
    func=generate_plot,
    description="""
    Generates a plot using Matplotlib based on provided Python code.
    Use this tool when the user asks for a visualization, graph, or plot of a function or data.
    Input MUST be valid Python code that uses matplotlib.pyplot (as plt) to generate a plot.
    The code MUST generate a plot and implicitly save it; do NOT include plt.show().
    Example input for a sine wave:
    'import numpy as np\\nx = np.linspace(0, 2 * np.pi, 100)\\ny = np.sin(x)\\nplt.plot(x, y)\\nplt.title(\"Sine Wave\")\\nplt.xlabel(\"x\")\\nplt.ylabel(\"sin(x)\")\\nplt.grid(True)'
    The tool will return the filepath to the generated plot image or an error message.
    """,
)
# --- End Matplotlib Tool ---


tools = [tavily_tool, retriever_tool, matplotlib_tool] # Add the new tool

# --- Agent Prompt ---
# Updated system message
system_message_content = """
You are a highly intelligent and helpful AI Math Agent. Your goal is to assist users with their math questions and problems, including generating visualizations when helpful.

Follow these steps carefully:
1.  **Understand the Question:** Read the user's query thoroughly. Identify the core mathematical concepts involved and whether a visualization is requested or would be beneficial.
2.  **Plan Your Approach:** Break down the problem into smaller, manageable steps. Think step-by-step. Briefly outline your plan before executing. If planning to plot, mention it.
3.  **Utilize Tools:**
    * If the question involves specific math concepts, formulas, or theorems, first use the `math_concepts_retriever` tool to check the knowledge base.
    * If the knowledge base doesn't have the answer, or if you need real-time information, external context, or verification, use the `tavily_search` tool. **When reporting results from Tavily, explicitly mention the source URLs if they are available in the search result snippets.**
    * If the user asks for a plot, graph, or visualization, OR if you determine that a plot would significantly clarify the explanation (e.g., plotting a function discussed), use the `matplotlib_plotter` tool. Provide the necessary Python code (using libraries like numpy 'np' and matplotlib.pyplot 'plt') to generate the plot. Ensure the code only contains plotting logic and does not include `plt.show()`.
    * Clearly state which tool you are using and why.
4.  **Execute and Explain:** Perform the necessary calculations or reasoning based on your plan and tool results. Explain each step clearly and concisely. Show your work where applicable. If a plot was generated by the tool, mention that the plot is being displayed.
5.  **Provide Awesome Solutions:** Aim for clarity, accuracy, and insight. Offer explanations that are easy to understand, even for complex topics. Use Markdown for formatting (e.g., bold text, bullet points, code blocks for formulas/steps).
6.  **Memory:** Remember the context of the conversation to provide relevant follow-up assistance.
7.  **Be Polite and Encouraging:** Maintain a friendly and supportive tone.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_message_content),
        MessagesPlaceholder(variable_name="chat_history", optional=True), # For memory
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # For agent's internal steps
    ]
)


# --- Agent Initialization ---
@cl.on_chat_start
async def start_chat():
    """Initializes the agent and memory when a new chat starts."""
    # (Setup is mostly the same as before)
    cl.user_session.set("description", "AI Math Agent")
    # Removed markdown template as output structure is more dynamic now

    print("Initializing Math Agent...")
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2, streaming=True)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=MEMORY_WINDOW_SIZE, return_messages=True, output_key="output"
    )
    cl.user_session.set("memory", memory)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, memory=memory, verbose=True,
        handle_parsing_errors="Check your output and make sure it conforms!", # More informative error
        max_iterations=7, # Slightly increased for potential plotting steps
    )
    cl.user_session.set("agent_executor", agent_executor)
    print("Math Agent Initialized.")
    await cl.Message(content="Hello! I'm your AI Math Assistant. I can help solve problems and even generate plots. How can I assist you today?").send()

# --- Message Handling ---
@cl.on_message
async def handle_message(message: cl.Message):
    """Handles incoming user messages and runs the agent."""
    agent_executor = cl.user_session.get("agent_executor")
    memory = cl.user_session.get("memory") # Retrieve memory

    if not agent_executor:
        await cl.Message(content="Agent not initialized. Please restart the chat.").send()
        return

    # Create the response message placeholder
    msg = cl.Message(content="")
    await msg.send()

    plot_paths_generated = [] # Keep track of generated plots in this turn

    # Stream the response
    async for chunk in agent_executor.astream(
        {"input": message.content},
        config={"callbacks": [cl.LangchainCallbackHandler(stream_final_answer=True)]}
    ):
        # Agent Action/Observation Steps (Intermediate)
        if "actions" in chunk:
            for action in chunk["actions"]:
                # Optional: Stream intermediate actions to the UI for clarity
                # await msg.stream_token(f"\n*Thinking: Using tool {action.tool} with input {action.tool_input}*")
                pass # Avoid cluttering UI, keep logs verbose if needed
        elif "steps" in chunk:
             # This part receives the observation (output) from the tool
             for step in chunk["steps"]:
                 # Check if the matplotlib tool was used and returned a valid path
                 # We infer the tool used based on the likely output format (a file path)
                 # A more robust way might involve checking action logs if available in stream
                 observation = str(step.observation) # Ensure it's a string
                 if TEMP_PLOT_DIR in observation and observation.endswith(".png") and os.path.exists(observation):
                     if observation not in plot_paths_generated: # Avoid duplicates if stream sends observation multiple times
                        print(f"Detected plot path: {observation}")
                        plot_paths_generated.append(observation)
                        # Send the image immediately
                        await cl.Message(
                            content=f"Generated plot:", # Simple label
                            elements=[
                                cl.Image(path=observation, name=os.path.basename(observation), display="inline")
                            ]
                        ).send()
                 # Optional: Stream intermediate observations for debugging
                 # await msg.stream_token(f"\n*Observation: {step.observation}*")

        # Final Answer Chunk
        elif "output" in chunk:
            # Stream the final text output from the LLM
            await msg.stream_token(chunk["output"])

    # Update the message with the final accumulated text content
    # The plots are sent as separate messages above
    await msg.update()

    # Clean up temporary plot files generated in this turn
    # for plot_path in plot_paths_generated:
    #     if os.path.exists(plot_path):
    #         try:
    #             os.remove(plot_path)
    #             print(f"Cleaned up temporary plot: {plot_path}")
    #         except OSError as e:
    #             print(f"Error cleaning up plot {plot_path}: {e}")

    # Note: Cleaning up immediately might cause issues if the UI needs the file later.
    # A better strategy might involve periodic cleanup or cleanup on session end.
    # For simplicity here, we'll leave the cleanup commented out.


# --- To Run This Code ---
# 1. Install required libraries:
#    pip install chainlit langchain langchain-openai langchain-community tavily-python chromadb tiktoken matplotlib numpy pandas
# 2. Set environment variables:
#    export OPENAI_API_KEY="your_openai_api_key"
#    export TAVILY_API_KEY="your_tavily_api_key"
# 3. Save the code as main.py
# 4. Run from terminal:
#    chainlit run main.py -w
