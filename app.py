# -*- coding: utf-8 -*-
"""
JEEnius - An AI-powered JEE Math Tutor using LangChain, OpenAI, and Chainlit.

This script sets up a conversational agent capable of solving JEE math problems,
explaining concepts, visualizing functions, querying a knowledge base,
and analyzing student performance.
"""

import os
import json
import time
import io
import base64
import re
import logging
from typing import Dict, List, Any, Optional

# Third-party libraries
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_core.tools import BaseTool, tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
# Updated import for ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import latex
from PIL import Image
import chainlit as cl

# --- Configuration & Initialization ---

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # Tavily is not used in the tools, using DuckDuckGo instead via Langchain

# Directories
TEMP_DIR = "temp"
KNOWLEDGE_BASE_DIR = "knowledge_base"
CHROMA_DB_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "chroma_db")
STUDENT_PROFILES_DIR = "user_data/student_profiles"
DOCUMENTS_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "documents") # Directory to load documents from

# Create necessary directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(STUDENT_PROFILES_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True) # Chroma PersistentClient needs the dir

# --- LLM and Embeddings ---
try:
    # Check if API key is loaded
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1, api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    # Setup embedding function for ChromaDB compatible with OpenAIEmbeddings
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-ada-002" # Or newer models if available/preferred
    )
except Exception as e:
    logging.error(f"Failed to initialize OpenAI models: {e}")
    # Exit or handle gracefully if core components fail
    print(f"Fatal Error: Failed to initialize OpenAI models. Please check your API key and environment setup. Error: {e}")
    exit(1)


# --- Knowledge Base Setup (ChromaDB) ---
CHROMA_COLLECTION_NAME = "jee_knowledge"

def initialize_knowledge_base() -> Optional[chromadb.Collection]:
    """
    Initializes or loads the ChromaDB vector store with JEE documents.

    Returns:
        Optional[chromadb.Collection]: The ChromaDB collection object or None if setup fails.
    """
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        # Check if the collection already exists
        existing_collections = [col.name for col in chroma_client.list_collections()]
        if CHROMA_COLLECTION_NAME in existing_collections:
            logging.info(f"Loading existing ChromaDB collection: {CHROMA_COLLECTION_NAME}")
            # Ensure embedding function is passed when getting the collection if it wasn't set during creation or differs
            collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=openai_ef)
            # Verify collection has data (optional check)
            if collection.count() == 0:
                 logging.warning("Existing collection found but is empty. Consider re-indexing.")
                 # Optionally trigger re-indexing here
                 # return index_documents(chroma_client) # Example call
            return collection
        else:
            logging.info(f"Creating new ChromaDB collection: {CHROMA_COLLECTION_NAME}")
            # If collection doesn't exist, create it and index documents
            return index_documents(chroma_client)

    except Exception as e:
        logging.error(f"Error initializing ChromaDB: {e}", exc_info=True)
        return None

def index_documents(client: chromadb.PersistentClient) -> Optional[chromadb.Collection]:
    """
    Loads documents, splits them, and indexes them into a new ChromaDB collection.

    Args:
        client (chromadb.PersistentClient): The ChromaDB client instance.

    Returns:
        Optional[chromadb.Collection]: The newly created collection or None if indexing fails.
    """
    try:
        # Create sample files if DOCUMENTS_DIR is empty (for demo purposes)
        if not os.listdir(DOCUMENTS_DIR):
            logging.info(f"No documents found in {DOCUMENTS_DIR}. Creating sample files.")
            sample_files = {
                "jee_math_formulas.txt": "JEE Mathematics Formulas:\n\n1. Quadratic Formula: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$\n2. Integration by Parts: $\\int u \\, dv = uv - \\int v \\, du$\n3. Differentiation: $\\frac{d}{dx}(x^n) = nx^{n-1}$\n",
                "jee_physics_concepts.txt": "JEE Physics Concepts:\n\n1. Newton's Laws of Motion\n2. Conservation of Energy\n3. Electromagnetism principles\n",
                "jee_chemistry_notes.txt": "JEE Chemistry Notes:\n\n1. Periodic Table properties\n2. Chemical Bonding\n3. Organic Chemistry reactions\n"
            }
            for filename, content in sample_files.items():
                with open(os.path.join(DOCUMENTS_DIR, filename), 'w', encoding='utf-8') as f:
                    f.write(content)

        # Load documents from the directory
        logging.info(f"Loading documents from: {DOCUMENTS_DIR}")
        # Using DirectoryLoader to handle various file types potentially
        # Defaulting to TextLoader, assuming most content is text-based.
        # Add specific loaders to loader_map if needed for PDF, CSV etc.
        loader = DirectoryLoader(
            DOCUMENTS_DIR,
            glob="**/*.*", # Load all files
            show_progress=True,
            use_multithreading=True,
            loader_cls=TextLoader, # Default loader
            # Example: loader_map={".pdf": PyPDFLoader, ".csv": CSVLoader, ".txt": TextLoader}
            silent_errors=True # Prevent crashing on single file load error
        )

        documents = loader.load()

        if not documents:
            logging.warning(f"No documents successfully loaded from {DOCUMENTS_DIR}. Knowledge base will be empty.")
            # Still create the collection, but it will be empty
            collection = client.create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=openai_ef)
            return collection

        logging.info(f"Successfully loaded {len(documents)} documents.")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        if not chunks:
            logging.warning("No chunks created from documents. Knowledge base will be empty.")
            collection = client.create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=openai_ef)
            return collection

        # Prepare data for ChromaDB
        # Ensure IDs are unique strings
        ids = [f"doc_{i}_{hash(chunk.page_content)[:8]}" for i, chunk in enumerate(chunks)] # Create more robust IDs
        contents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks] # Ensure metadata is serializable

        # Clean metadata (ensure values are str, int, float, or bool)
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned_meta[k] = v
                else:
                    cleaned_meta[k] = str(v) # Convert other types to string
            cleaned_metadatas.append(cleaned_meta)


        # Create and populate the collection
        logging.info(f"Creating and populating collection: {CHROMA_COLLECTION_NAME}")
        # Use get_or_create_collection for robustness
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=openai_ef)

        # Add documents in batches if necessary (for very large datasets)
        batch_size = 100 # Example batch size
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_contents = contents[i:i+batch_size]
            batch_metadatas = cleaned_metadatas[i:i+batch_size]
            try:
                 collection.add(
                     ids=batch_ids,
                     documents=batch_contents,
                     metadatas=batch_metadatas
                 )
                 logging.info(f"Added batch {i//batch_size + 1} to ChromaDB collection.")
            except Exception as batch_e:
                 logging.error(f"Error adding batch {i//batch_size + 1} to ChromaDB: {batch_e}")
                 # Decide whether to continue or raise the error

        logging.info("Knowledge base indexing completed.")
        return collection

    except Exception as e:
        logging.error(f"Error indexing documents: {e}", exc_info=True)
        return None

# Initialize or load the knowledge base collection
knowledge_collection = initialize_knowledge_base()
if knowledge_collection is None:
     logging.warning("Knowledge base could not be initialized. Querying knowledge base will not work.")


# --- Student Profile Manager ---
class StudentProfileManager:
    """Manages loading, saving, and updating student profiles."""
    def __init__(self, profiles_dir=STUDENT_PROFILES_DIR):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)
        logging.info(f"Student profiles will be stored in: {profiles_dir}")

    def get_profile_path(self, student_id: str) -> str:
        """Constructs the file path for a student's profile."""
        # Sanitize student_id to prevent path traversal issues
        safe_student_id = re.sub(r'[^a-zA-Z0-9_-]', '_', student_id)
        return os.path.join(self.profiles_dir, f"{safe_student_id}.json")

    def load_profile(self, student_id: str) -> Dict[str, Any]:
        """Loads a student profile, creating one if it doesn't exist."""
        profile_path = self.get_profile_path(student_id)
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    # Ensure all keys exist, add defaults if missing
                    profile = self._ensure_profile_keys(student_id, profile)
                    logging.info(f"Loaded profile for student: {student_id}")
                    return profile
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON for student {student_id} at {profile_path}. Creating new profile.")
                return self._create_new_profile(student_id)
            except Exception as e:
                 logging.error(f"Error loading profile for student {student_id} from {profile_path}: {e}. Creating new profile.")
                 return self._create_new_profile(student_id)
        else:
            logging.info(f"No profile found for student {student_id}. Creating new profile.")
            return self._create_new_profile(student_id)

    def _create_new_profile(self, student_id: str) -> Dict[str, Any]:
        """Creates a default profile structure."""
        new_profile = {
            "student_id": student_id,
            "name": "Student", # Default name
            "topics_strengths": {}, # Track correct attempts per topic
            "topics_weaknesses": {}, # Track incorrect attempts per topic
            "interaction_log": [], # Log of topics interacted with
            "questions_solved": 0, # Total interactions logged with feedback
            "correct_solutions": 0, # Requires feedback mechanism
            "recent_topics": [],
            "difficulty_preference": "medium", # Default preference
            "last_active": time.time()
        }
        # Don't save here, let the caller save after potential modifications
        # self.save_profile(student_id, new_profile)
        return new_profile

    def _ensure_profile_keys(self, student_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures all expected keys are present in a loaded profile."""
        defaults = self._create_new_profile(student_id) # Get default structure
        updated = False
        for key, default_value in defaults.items():
            if key not in profile:
                profile[key] = default_value
                updated = True
        # Only save if structure was actually modified
        # if updated:
        #     logging.info(f"Updated profile structure for student: {student_id}")
        #     self.save_profile(student_id, profile) # Save the updated structure
        return profile


    def save_profile(self, student_id: str, profile_data: Dict[str, Any]):
        """Saves the student profile data to a JSON file."""
        profile_path = self.get_profile_path(student_id)
        try:
            # Ensure last_active is updated before saving
            profile_data['last_active'] = time.time()
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile_data, f, indent=4) # Use indent for readability
            logging.debug(f"Saved profile for student: {student_id}")
        except Exception as e:
            logging.error(f"Error saving profile for student {student_id} to {profile_path}: {e}")

    def log_interaction(self, student_id: str, topic: str, success: Optional[bool] = None):
        """Logs an interaction with a topic and optionally updates performance."""
        # Load the latest profile state
        profile = self.load_profile(student_id)

        # Log the interaction
        profile["interaction_log"].append({"topic": topic, "timestamp": time.time(), "success": success})
        # Limit log size if needed
        profile["interaction_log"] = profile["interaction_log"][-100:] # Keep last 100 interactions

        # Update recent topics
        # Remove topic if it exists, then insert at the beginning
        if topic in profile["recent_topics"]:
            profile["recent_topics"].remove(topic)
        profile["recent_topics"].insert(0, topic)
        profile["recent_topics"] = profile["recent_topics"][:5] # Keep top 5

        # Update counts (only if success feedback is provided)
        if success is not None:
            profile["questions_solved"] = profile.get("questions_solved", 0) + 1
            # Initialize topic counts if they don't exist
            profile["topics_strengths"] = profile.get("topics_strengths", {})
            profile["topics_weaknesses"] = profile.get("topics_weaknesses", {})
            profile["topics_strengths"][topic] = profile["topics_strengths"].get(topic, 0)
            profile["topics_weaknesses"][topic] = profile["topics_weaknesses"].get(topic, 0)

            if success:
                profile["topics_strengths"][topic] += 1
                profile["correct_solutions"] = profile.get("correct_solutions", 0) + 1
            else:
                profile["topics_weaknesses"][topic] += 1

        # Save the updated profile
        self.save_profile(student_id, profile)
        logging.info(f"Logged interaction for student {student_id} on topic '{topic}'. Success: {success}")
        # Return the updated profile (optional)
        # return profile

# Initialize student profile manager
student_manager = StudentProfileManager()


# --- Agent Tools ---

# Tool: Search (Using DuckDuckGo via Langchain Community)
# Switched to DuckDuckGo for simplicity as Tavily requires an API key.
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    web_search_tool = DuckDuckGoSearchRun()
    logging.info("Using DuckDuckGoSearchRun for web search.")
except ImportError:
    logging.warning("DuckDuckGoSearchRun not available (pip install duckduckgo-search?). Web search tool will be a placeholder.")
    # Define placeholder tool if import fails
    @tool
    def web_search_tool(query: str) -> str:
        """
        Placeholder web search tool. Performs a web search for the given query.
        Install 'duckduckgo-search' for actual functionality.
        """
        return f"Search tool not available. Cannot search for: {query}"

@tool
def web_search(query: str) -> str:
    """
    Performs a web search for the given query using DuckDuckGo. Use this to find general information,
    current events, or resources not present in the knowledge base.
    For specific JEE problems or concepts, prefer 'query_knowledge_base' or 'solve_math_problem'.
    Example Input: "Latest advancements in calculus teaching methods"
    """
    logging.info(f"Tool 'web_search' called for: {query}")
    try:
        return web_search_tool.run(query)
    except Exception as e:
         logging.error(f"Web search failed: {e}")
         return f"Error during web search: {e}"


@tool
def solve_math_problem(problem: str) -> str:
    """
    Solves a mathematical problem step-by-step using an LLM.
    Use this for solving specific math questions, especially those requiring detailed derivation.
    Input should be the full text of the math problem.
    Provides reasoning and uses LaTeX for mathematical expressions.
    """
    logging.info(f"Tool 'solve_math_problem' called for: {problem[:50]}...")
    solve_prompt = PromptTemplate(
        template="""
        You are an expert mathematics tutor specializing in JEE (Joint Entrance Examination) problems.
        Solve the following JEE math problem step-by-step:

        Problem: {problem}

        Follow these steps meticulously:
        1.  **Understand the Problem:** Clearly state what the problem is asking and identify the core mathematical concepts involved.
        2.  **Identify Key Information:** List the given data, variables, and constraints.
        3.  **Formulate a Plan:** Outline the approach or strategy you will use (e.g., which formulas, theorems, or techniques are applicable).
        4.  **Execute the Plan (Step-by-Step):** Break down the solution into logical steps. Show all calculations clearly. Use LaTeX notation for all mathematical expressions, formulas, and variables (e.g., $E = mc^2$, $\int_a^b f(x) \\, dx$). Explain the reasoning behind each step.
        5.  **Verification (Optional but Recommended):** If possible, briefly check if the answer is reasonable or verify it using an alternative method.
        6.  **Final Answer:** State the final answer clearly and concisely, again using LaTeX if it involves mathematical notation.
        7.  **Learning Points:** Briefly summarize the key concepts, formulas, or techniques used in solving this problem that the student should remember.

        Format your response using Markdown:
        ```markdown
        ## Problem Understanding
        [Explanation]

        ## Key Information
        [Given data, variables]

        ## Plan/Strategy
        [Outline of the solution approach]

        ## Step-by-Step Solution

        ### Step 1: [Descriptive Step Name]
        [Detailed explanation with LaTeX math: $...$]

        ### Step 2: [Descriptive Step Name]
        [Detailed explanation with LaTeX math: $...$]

        ... (continue steps as needed)

        ## Verification (Optional)
        [Check or verification steps]

        ## Final Answer
        The final answer is: [Answer using LaTeX if needed, e.g., $x = 5$]

        ## Learning Points
        - [Key concept 1]
        - [Formula used]
        - [Technique applied]
        ```
        Ensure all mathematical parts are enclosed in $...$ for inline math or $$...$$ for display math. Use standard LaTeX commands.
        """,
        input_variables=["problem"],
    )
    try:
        # Ensure the LLM object is available
        if llm is None:
            return "Error: LLM not initialized."
        chain = solve_prompt | llm | StrOutputParser()
        solution = chain.invoke({"problem": problem})
        logging.info("Successfully generated solution using 'solve_math_problem'.")
        return solution
    except Exception as e:
        logging.error(f"Error in solve_math_problem tool: {e}", exc_info=True)
        return f"An error occurred while trying to solve the problem: {e}"


@tool
def query_knowledge_base(query: str) -> str:
    """
    Queries the internal knowledge base (vector store) for JEE math concepts, formulas, definitions, or explanations stored in documents.
    Use this before trying a general web search for specific JEE-related information.
    Input should be a question or topic related to JEE Math.
    Example Input: "Explain integration by parts" or "Formula for the area of an ellipse"
    """
    logging.info(f"Tool 'query_knowledge_base' called for: {query}")
    if knowledge_collection is None:
        logging.warning("query_knowledge_base called but collection is None.")
        return "Knowledge base is not available. Cannot query."
    try:
        results = knowledge_collection.query(
            query_texts=[query],
            n_results=3, # Retrieve top 3 relevant chunks
            include=['documents', 'metadatas'] # Include content and source info
        )

        # Check results structure carefully
        if not results or not results.get('ids') or not results['ids'][0]:
            logging.warning(f"No relevant information found in knowledge base for query: {query}")
            return "No relevant information found in the knowledge base for this query."

        result_str = "### Relevant Information from Knowledge Base:\n\n"
        num_results = len(results['ids'][0])
        for i in range(num_results):
            doc = results['documents'][0][i] if results.get('documents') and results['documents'][0] else "N/A"
            metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}
            source = metadata.get('source', 'Unknown source')
            # page = metadata.get('page', 'N/A') # If using PyPDFLoader and page info exists

            result_str += f"**Source Document {i+1}:** `{os.path.basename(source)}`\n"
            # if page != 'N/A': result_str += f"Page: {page}\n"
            result_str += f"Content Snippet:\n```\n{doc}\n```\n---\n"

        logging.info(f"Found {num_results} relevant snippets in knowledge base.")
        return result_str

    except Exception as e:
        logging.error(f"Error querying knowledge base: {e}", exc_info=True)
        return f"An error occurred while querying the knowledge base: {e}"


@tool
def generate_visualization(math_expression_description: str) -> str:
    """
    Generates a visualization (plot) for a mathematical function or a simple geometric shape.
    Input should describe what to plot. Include the function/equation and optionally the range or details.
    Use LaTeX notation for the expression itself within the description if possible.
    Format: "Plot the function [function in LaTeX, e.g., $f(x) = x^2 - 3x + 2$] from x = -2 to 5"
    Format: "Draw a circle with radius 5 centered at (2, -1)"
    Format: "Plot the curve $y^2 = 4ax$"
    Returns a Markdown string with a base64 encoded image or an error message.
    """
    logging.info(f"Tool 'generate_visualization' called for: {math_expression_description}")
    # Ensure LLM is available
    if llm is None:
        return "Error: LLM not initialized, cannot parse visualization request."
    try:
        # Use LLM to parse the request and generate parameters for plotting
        parsing_prompt = PromptTemplate(
            template="""
            Parse the following request to extract information needed for plotting. Identify the type of plot (function, curve, geometric) and the necessary parameters (equation/expression, variable(s), range, center, radius, vertices, etc.). Output ONLY a JSON object with the extracted information. Handle ranges like "from a to b" or "between a and b". Use standard python math syntax for expressions (e.g., x**2, np.sin(x)). If a parameter like range is missing for a function, use a default like -10 to 10.

            Request: "{request}"

            Example Outputs:
            Request: "Plot the function $f(x) = x^2 - 3x + 2$ from x = -2 to 5"
            Output: {{"type": "function", "expression": "x**2 - 3*x + 2", "variable": "x", "range_min": -2.0, "range_max": 5.0, "latex_expression": "x^2 - 3x + 2"}}

            Request: "Draw a circle with radius 5 centered at (2, -1)"
            Output: {{"type": "geometric", "shape": "circle", "radius": 5.0, "center": [2.0, -1.0]}}

            Request: "Plot the curve $y^2 = 4ax$. Assume a=1."
            Output: {{"type": "curve", "equation": "y**2 = 4*x", "variables": ["x", "y"], "latex_expression": "y^2 = 4x"}}

            Request: "Visualize a triangle with vertices (0,0), (1,1), (0,2)"
            Output: {{"type": "geometric", "shape": "triangle", "vertices": [[0.0,0.0], [1.0,1.0], [0.0,2.0]]}}

            Request: "Graph $y = \sin(x)$"
            Output: {{"type": "function", "expression": "sin(x)", "variable": "x", "range_min": -10.0, "range_max": 10.0, "latex_expression": "\\\\sin(x)"}}

            Now parse this request:
            Request: "{request}"
            Output:
            """,
            input_variables=["request"]
        )
        parser_chain = parsing_prompt | llm | StrOutputParser()
        params_json_str = parser_chain.invoke({"request": math_expression_description})

        logging.debug(f"LLM Parsing Output for visualization: {params_json_str}")

        # Clean potential markdown code block fences and extraneous text
        params_json_str = re.sub(r"^.*?```json\s*", "", params_json_str.strip(), flags=re.DOTALL)
        params_json_str = re.sub(r"\s*```.*?$", "", params_json_str.strip(), flags=re.DOTALL)

        try:
             params = json.loads(params_json_str)
        except json.JSONDecodeError as json_err:
             logging.error(f"Failed to decode JSON from LLM parser: {json_err}")
             logging.error(f"Invalid JSON string received: {params_json_str}")
             # Ask LLM to reformat its own output might be an option here, but for now, return error
             return f"Error: Could not understand the plotting request structure after parsing. Parser output: ```{params_json_str}```"


        plot_type = params.get("type")
        # Ensure matplotlib uses a non-interactive backend suitable for saving files
        plt.switch_backend('Agg')
        plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
        fig, ax = plt.subplots(figsize=(8, 6)) # Create figure and axes explicitly

        # --- Function Plotting ---
        if plot_type == "function":
            expr_str = params.get("expression")
            var_str = params.get("variable", "x")
            x_min = float(params.get("range_min", -10.0)) # Ensure float
            x_max = float(params.get("range_max", 10.0)) # Ensure float
            latex_expr = params.get("latex_expression", expr_str) # Fallback to expression

            if not expr_str: return "Error: Missing function expression in parsed request."

            try:
                var = sp.symbols(var_str)
                # Define allowed symbols/functions for sympify
                allowed_locals = {
                    var_str: var,
                    'e': sp.E, 'pi': sp.pi,
                    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
                    'log': sp.log, 'ln': sp.ln, 'sqrt': sp.sqrt,
                    'abs': sp.Abs, 'exp': sp.exp
                 }
                func = sp.sympify(expr_str, locals=allowed_locals)

                # Generate points
                x_vals_np = np.linspace(x_min, x_max, 400)
                y_vals_np = np.full_like(x_vals_np, np.nan) # Initialize with NaN

                # Use lambdify for faster numerical evaluation if possible
                try:
                    func_lambdified = sp.lambdify(var, func, modules=['numpy', {'Abs': np.abs}]) # Use numpy functions where possible
                    y_vals_np = func_lambdified(x_vals_np)
                    # Replace potential complex numbers resulting from evaluation with NaN
                    if np.iscomplexobj(y_vals_np):
                        y_vals_np = np.where(np.isreal(y_vals_np), y_vals_np.real, np.nan)

                except (SyntaxError, NameError, TypeError) as lambdify_err:
                     logging.warning(f"Lambdify failed for '{expr_str}': {lambdify_err}. Falling back to subs.")
                     # Fallback to slower subs evaluation
                     for i, x_val in enumerate(x_vals_np):
                         try:
                             y_val_sympy = func.subs(var, x_val).evalf()
                             if y_val_sympy.is_real:
                                 y_vals_np[i] = float(y_val_sympy)
                             # else leave as NaN
                         except (TypeError, ValueError, AttributeError):
                             pass # leave as NaN

                # Plot
                ax.plot(x_vals_np, y_vals_np, label=f"${latex_expr}$")
                ax.set_xlabel(f"${var_str}$")
                ax.set_ylabel(f"$f({var_str})$")
                ax.set_title(f"Plot of ${latex_expr}$")
                ax.legend()
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.axhline(0, color='black', linewidth=0.5)
                ax.axvline(0, color='black', linewidth=0.5)
                # Add reasonable y-limits to avoid issues with asymptotes
                valid_y = y_vals_np[~np.isnan(y_vals_np)]
                if len(valid_y) > 0:
                    y_median = np.median(valid_y)
                    y_std = np.std(valid_y) if len(valid_y) > 1 else 1.0
                    y_lim_lower = y_median - 5 * y_std
                    y_lim_upper = y_median + 5 * y_std
                     # Ensure limits are reasonable, prevent excessively large ranges
                    if y_lim_upper - y_lim_lower > 1e4:
                         y_lim_lower = np.percentile(valid_y, 5) - 1
                         y_lim_upper = np.percentile(valid_y, 95) + 1

                    ax.set_ylim(y_lim_lower, y_lim_upper)


            except (sp.SympifyError, TypeError, ValueError) as e:
                logging.error(f"SymPy or plotting error for function '{expr_str}': {e}")
                return f"Error processing function '{expr_str}': {e}. Please ensure it's a valid mathematical expression using standard functions (sin, cos, log, sqrt, etc.)."

        # --- Geometric Shape Plotting ---
        elif plot_type == "geometric":
            shape = params.get("shape")
            if shape == "circle":
                radius = float(params.get("radius", 1.0))
                center = tuple(map(float, params.get("center", [0.0, 0.0])))
                circle = plt.Circle(center, radius, fill=False, color='blue', linewidth=2)
                ax.add_patch(circle)
                ax.set_title(f"Circle: radius={radius}, center={center}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                # Adjust limits to show the circle nicely
                lim_pad = radius * 0.2
                ax.set_xlim(center[0] - radius - lim_pad, center[0] + radius + lim_pad)
                ax.set_ylim(center[1] - radius - lim_pad, center[1] + radius + lim_pad)
                ax.set_aspect('equal', adjustable='box') # Ensure it looks like a circle

            elif shape == "triangle":
                vertices = params.get("vertices")
                if vertices and len(vertices) == 3 and all(len(v) == 2 for v in vertices):
                    try:
                        verts_np = np.array(vertices, dtype=float)
                        triangle = plt.Polygon(verts_np, closed=True, fill=True, alpha=0.3, edgecolor='green', linewidth=2)
                        ax.add_patch(triangle)
                        ax.set_title(f"Triangle: vertices={vertices}")
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                         # Adjust limits
                        min_coords = verts_np.min(axis=0)
                        max_coords = verts_np.max(axis=0)
                        range_coords = max_coords - min_coords
                        pad = max(range_coords.max() * 0.1, 0.5) # Add padding, ensure minimum padding
                        ax.set_xlim(min_coords[0] - pad, max_coords[0] + pad)
                        ax.set_ylim(min_coords[1] - pad, max_coords[1] + pad)
                        ax.set_aspect('equal', adjustable='box')
                    except ValueError as e:
                         return f"Error: Invalid vertex format for triangle: {vertices}. Ensure vertices are pairs of numbers. Error: {e}"
                else:
                    return "Error: Invalid or missing vertices for triangle. Need 3 pairs of coordinates like [[x1,y1], [x2,y2], [x3,y3]]."
            else:
                return f"Error: Unsupported geometric shape '{shape}'."

            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)

        # --- Implicit Curve Plotting (Placeholder) ---
        elif plot_type == "curve":
             # Placeholder - full implementation is complex
             eq_str = params.get("equation", "N/A")
             logging.warning(f"Implicit curve plotting requested ('{eq_str}'), which is complex. Returning placeholder.")
             return f"Plotting implicit curves like '{eq_str}' is currently not fully supported. You can try plotting it as a function if you can solve for one variable (e.g., solve $y^2=4x$ for $y$ and plot $y = \\sqrt4x$ and $y = -\\sqrt4x$)."

        else:
            return f"Error: Unsupported plot type '{plot_type}' identified by parser."

        # --- Save plot to buffer and encode ---
        buf = io.BytesIO()
        # Use the figure object to save
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig) # Close the specific figure

        logging.info("Successfully generated visualization.")
        # Return Markdown image tag
        return f"![Visualization](data:image/png;base64,{img_base64})"

    except json.JSONDecodeError as json_err:
        # This error is caught above, but as a fallback
        logging.error(f"JSON parsing error in generate_visualization: {json_err}")
        return f"Error: Could not understand the plotting request structure. {json_err}"
    except Exception as e:
        logging.error(f"Error generating visualization: {e}", exc_info=True) # Log full traceback
        # Attempt to return a more informative error if possible
        if isinstance(e, (sp.SympifyError, TypeError, ValueError)):
             plt.close(plt.gcf()) # Ensure plot is closed on error
             return f"Error processing mathematical expression: {e}. Please check the syntax."
        plt.close(plt.gcf()) # Ensure plot is closed on error
        return f"An unexpected error occurred while generating the visualization: {e}"


@tool
def analyze_student_performance(student_id: str) -> str:
    """
    Analyzes the performance of a student based on their interaction history stored in their profile.
    Provides insights into strengths, weaknesses, recent activity, and overall accuracy (if feedback is logged).
    Input should be the student's ID (which is usually managed by the system).
    """
    logging.info(f"Tool 'analyze_student_performance' called for student: {student_id}")
    try:
        profile = student_manager.load_profile(student_id)

        # Calculate strengths (topics with most correct interactions)
        strengths = sorted(
            [(topic, score) for topic, score in profile.get("topics_strengths", {}).items() if score > 0],
            key=lambda x: x[1],
            reverse=True
        )[:3] # Top 3

        # Calculate weaknesses (topics with most incorrect interactions)
        weaknesses = sorted(
            [(topic, score) for topic, score in profile.get("topics_weaknesses", {}).items() if score > 0],
            key=lambda x: x[1],
            reverse=True
        )[:3] # Top 3

        # Calculate accuracy (requires success logging)
        total_interactions_with_feedback = profile.get("questions_solved", 0)
        correct_solutions = profile.get("correct_solutions", 0)
        accuracy = (correct_solutions / total_interactions_with_feedback * 100) if total_interactions_with_feedback > 0 else 0

        # Generate the analysis report
        analysis = f"""## Student Performance Analysis for {profile.get('name', student_id)}

**Overall Activity:**
- Total interactions logged: {len(profile.get('interaction_log', []))}
- Interactions with feedback provided: {total_interactions_with_feedback}
- Correct solutions (based on feedback): {correct_solutions}
- Accuracy (based on feedback): {accuracy:.1f}%

**Identified Strengths (Topics with most correct attempts):**
"""
        if strengths:
            for topic, score in strengths:
                analysis += f"- {topic.capitalize()} ({score} correct)\n"
        else:
            analysis += "- Not enough data with positive feedback to determine strengths. Try solving more problems and providing feedback!\n"

        analysis += "\n**Potential Areas for Improvement (Topics with most incorrect attempts):**\n"
        if weaknesses:
            for topic, score in weaknesses:
                analysis += f"- {topic.capitalize()} ({score} incorrect)\n"
        else:
            analysis += "- Not enough data with negative feedback to determine specific weaknesses. Keep practicing!\n"

        analysis += "\n**Recently Covered Topics:**\n"
        if profile.get("recent_topics"):
             analysis += "- " + "\n- ".join([topic.capitalize() for topic in profile["recent_topics"]]) + "\n"
        else:
            analysis += "- No recent topics recorded.\n"

        analysis += "\n**Recommendations:**\n"
        if weaknesses:
            analysis += f"- Focus practice on: {', '.join([w[0].capitalize() for w in weaknesses])}.\n"
            analysis += "- You can ask me to `explain_concept` for these topics or `suggest_practice_problems`.\n"
        elif total_interactions_with_feedback < 5: # Arbitrary threshold
             analysis += "- Keep interacting and solving problems! The more you practice, the better I can understand your needs.\n"
             analysis += "- Ask for explanations (`explain_concept`) if you're unsure about any topic.\n"
        else:
            analysis += "- Great job practicing! Continue exploring different topics.\n"
            analysis += "- Consider trying slightly harder problems (`suggest_practice_problems` with 'hard' difficulty) in your strong areas like {strengths[0][0].capitalize()}.\n" if strengths else ""

        logging.info(f"Successfully generated performance analysis for student: {student_id}")
        return analysis

    except Exception as e:
        logging.error(f"Error analyzing student performance for {student_id}: {e}", exc_info=True)
        return f"An error occurred while analyzing performance: {e}"


@tool
def suggest_practice_problems(topic: str, difficulty: str = "medium") -> str:
    """
    Suggests relevant JEE-level practice problems for a given mathematics topic and optional difficulty level.
    Input should be the topic name (e.g., "calculus", "complex numbers", "probability").
    Optionally specify difficulty: "easy", "medium", or "hard".
    Provides problem statements and hints, but not solutions.
    """
    logging.info(f"Tool 'suggest_practice_problems' called for topic: {topic}, difficulty: {difficulty}")
    # Validate difficulty
    valid_difficulties = ["easy", "medium", "hard"]
    if difficulty.lower() not in valid_difficulties:
        logging.warning(f"Invalid difficulty '{difficulty}' provided. Defaulting to 'medium'.")
        difficulty = "medium"

    # Use LLM to generate problems based on topic and difficulty
    suggest_prompt = PromptTemplate(
        template="""
        You are a JEE Mathematics question generator. Generate 3 distinct practice problems relevant to the topic '{topic}' at a '{difficulty}' difficulty level suitable for JEE preparation.

        For each problem:
        1. Provide a clear and concise problem statement using LaTeX notation for mathematical expressions ($...$). Ensure the problem is solvable and well-posed.
        2. Indicate the difficulty level ({difficulty}).
        3. Give a single, non-revealing hint to guide the student towards the correct approach or concept. Do NOT provide the solution or final answer.

        Format the output strictly using Markdown as follows:

        ```markdown
        ## Problem 1: [Brief Title Related to the Problem Concept]
        **Difficulty:** {difficulty}

        **Problem:** [Problem statement using LaTeX, e.g., Find the derivative of $f(x) = \sin(x^2)$ with respect to $x$.]

        **Hint:** [A single guiding hint, e.g., Remember the chain rule for differentiation.]

        ---

        ## Problem 2: [Brief Title Related to the Problem Concept]
        **Difficulty:** {difficulty}

        **Problem:** [Problem statement using LaTeX...]

        **Hint:** [A single guiding hint...]

        ---

        ## Problem 3: [Brief Title Related to the Problem Concept]
        **Difficulty:** {difficulty}

        **Problem:** [Problem statement using LaTeX...]

        **Hint:** [A single guiding hint...]
        ```
        Ensure the problems are appropriate for JEE level and the specified topic and difficulty. Make sure hints are genuinely helpful but don't give away the answer.
        """,
        input_variables=["topic", "difficulty"],
    )
    try:
        # Ensure LLM is available
        if llm is None:
            return "Error: LLM not initialized."
        chain = suggest_prompt | llm | StrOutputParser()
        problems = chain.invoke({"topic": topic, "difficulty": difficulty})
        logging.info("Successfully generated practice problems.")
        return problems
    except Exception as e:
        logging.error(f"Error suggesting practice problems for topic '{topic}': {e}", exc_info=True)
        return f"An error occurred while generating practice problems: {e}"


@tool
def explain_concept(concept: str) -> str:
    """
    Provides a detailed explanation of a mathematical concept relevant to JEE preparation.
    Input should be the name of the concept (e.g., "Integration by Parts", "Binomial Theorem", "Complex Numbers").
    Uses LaTeX for mathematical notation and aims for clarity and relevance to JEE syllabus.
    """
    logging.info(f"Tool 'explain_concept' called for concept: {concept}")
    explain_prompt = PromptTemplate(
        template="""
        You are an expert JEE mathematics teacher. Explain the following concept clearly and comprehensively for a JEE aspirant:

        **Concept:** {concept}

        Structure your explanation using Markdown as follows:
        1.  **Introduction/Overview:** Briefly define the concept and state its significance in the context of JEE mathematics.
        2.  **Core Definition(s)/Formula(s):** Clearly state the primary definition(s) or formula(s). Use precise mathematical language and LaTeX notation ($...$ for inline, $$...$$ for display math).
        3.  **Explanation & Intuition:** Elaborate on the definition/formula. Provide intuition or analogies where helpful. Explain the 'why' behind it, if possible.
        4.  **Illustrative Examples:** Provide 1-2 clear, step-by-step examples demonstrating the application of the concept. Show calculations using LaTeX.
        5.  **Key Properties/Theorems (if applicable):** List important properties, related theorems, or special cases. Use LaTeX.
        6.  **Applications in JEE Problems:** Describe common scenarios or types of JEE questions where this concept is frequently tested.
        7.  **Tips for Success & Common Mistakes:** Offer practical tips for mastering the concept and point out common errors or misunderstandings students encounter.

        Ensure all mathematical notation is correctly formatted using LaTeX. The explanation should be thorough yet concise, focusing on aspects most relevant to JEE preparation.
        ```markdown
        ## Explanation: {concept}

        ### 1. Introduction/Overview
        [Your explanation here]

        ### 2. Core Definition(s)/Formula(s)
        [Your explanation here, using LaTeX e.g., The formula is $a^2 + b^2 = c^2$.]

        ### 3. Explanation & Intuition
        [Your explanation here]

        ### 4. Illustrative Examples
        **Example 1:**
        *Problem:* [Problem statement]
        *Solution:*
        [Step-by-step solution using LaTeX]

        **Example 2:**
        *Problem:* [Problem statement]
        *Solution:*
        [Step-by-step solution using LaTeX]

        ### 5. Key Properties/Theorems
        - Property 1: [Description with LaTeX]
        - Theorem 1: [Description with LaTeX]

        ### 6. Applications in JEE Problems
        - [Application scenario 1]
        - [Application scenario 2]

        ### 7. Tips for Success & Common Mistakes
        **Tips:**
        - [Tip 1]
        - [Tip 2]
        **Common Mistakes:**
        - [Mistake 1: Explanation]
        - [Mistake 2: Explanation]
        ```
        """,
        input_variables=["concept"],
    )
    try:
         # Ensure LLM is available
        if llm is None:
            return "Error: LLM not initialized."
        chain = explain_prompt | llm | StrOutputParser()
        explanation = chain.invoke({"concept": concept})
        logging.info("Successfully generated concept explanation.")
        return explanation
    except Exception as e:
        logging.error(f"Error explaining concept '{concept}': {e}", exc_info=True)
        return f"An error occurred while explaining the concept: {e}"


# --- Agent Setup ---

# List of tools available to the agent
tools = [
    query_knowledge_base,
    solve_math_problem,
    generate_visualization,
    suggest_practice_problems,
    explain_concept,
    analyze_student_performance,
    web_search, # Use the DuckDuckGo tool instance
]

# Define the ReAct agent prompt
# Note: Improved instructions for tool usage and student interaction
react_prompt_template = """
You are JEEnius, an expert AI assistant and tutor specializing in JEE (Joint Entrance Examination) Mathematics. Your primary goal is to help students understand concepts, solve problems, and prepare effectively for the exam in a friendly, encouraging, and step-by-step manner.

You have access to the following tools:
{tools}

**Interaction Flow:**
1.  **Understand the Request:** Carefully analyze the student's query ({input}). Consider the conversation history ({chat_history}) for context. Is it a problem to solve, a concept to explain, a request for practice, a visualization, performance analysis, or something else?
2.  **Prioritize Knowledge Base:** For JEE-specific concepts, formulas, or definitions, ALWAYS try `query_knowledge_base` first.
3.  **Problem Solving:**
    * If the request is a specific math problem, use `solve_math_problem`. Ensure the output is detailed and uses LaTeX correctly ($...$ or $$...$$).
    * If the problem seems standard, you might briefly use `web_search` to see if common solutions exist online, but prefer `solve_math_problem` for generating a tailored, step-by-step explanation suitable for learning.
4.  **Concept Explanation:** Use `explain_concept` for detailed explanations of mathematical topics.
5.  **Practice:** Use `suggest_practice_problems` when asked for practice questions. Ask for topic and difficulty if not provided.
6.  **Visualization:** Use `generate_visualization` if the student asks to see a graph or geometric shape related to a mathematical expression. The tool expects a description (e.g., "plot sin(x) from -pi to pi").
7.  **Performance:** Use `analyze_student_performance` ONLY when explicitly asked by the user (e.g., "analyze my performance", "how am I doing?"). The student's ID ({student_id}) is automatically available to the tool.
8.  **General Queries:** Use `web_search` for information outside the scope of JEE math or the internal knowledge base (e.g., study tips, exam dates, general knowledge, non-math topics).
9.  **Be Conversational:** Respond in a helpful, tutor-like tone. Acknowledge the student's request. If a tool fails or returns an error, explain politely what happened and suggest an alternative or ask for clarification. Don't just output the error message.
10. **Use LaTeX:** ALWAYS use LaTeX notation ($...$ for inline, $$...$$ for display) for all mathematical symbols, variables, and expressions in your final answers and thoughts where appropriate. Double-check LaTeX formatting.

**Output Format:**
Follow the ReAct framework strictly:

Question: The input question or request from the student.
Thought: Your reasoning about the request, the best tool to use, and the plan. Check if you need information from chat history. Explicitly state which tool you are choosing and why.
Action: The action to take, MUST be one of [{tool_names}].
Action Input: The input string or dictionary for the selected tool. Ensure the input format matches the tool's requirements.
Observation: The result returned by the tool.
... (Repeat Thought/Action/Action Input/Observation N times as needed. If a tool fails, think about why and try a different approach or ask the user for clarification.)
Thought: I have gathered enough information and can now formulate the final response to the student. I will ensure the response is clear, addresses the original question ({input}) directly, uses Markdown for structure, and includes correct LaTeX formatting ($...$) for all math. If a visualization was generated, I will mention it and expect the system to display it.
Final Answer: Your comprehensive and well-formatted response to the student.

**Student Context:**
The current student's ID is {student_id}. The conversation history is provided in {chat_history}.

Let's begin!

Question: {input}
Thought: {agent_scratchpad}
"""

react_prompt = PromptTemplate.from_template(react_prompt_template)

# Create the agent
try:
    # Ensure LLM is available
    if llm is None:
        raise ValueError("LLM not initialized before agent creation.")
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # Logs agent steps (useful for debugging)
        handle_parsing_errors="I encountered an issue formatting my response. Please try again or rephrase your request.", # User-friendly message
        max_iterations=8, # Limit iterations to prevent loops
        max_execution_time=120.0, # Limit execution time in seconds (float)
        early_stopping_method="generate", # Stop if agent outputs Final Answer
    )
    logging.info("React Agent created successfully.")
except Exception as e:
    logging.error(f"Failed to create React agent: {e}", exc_info=True)
    print(f"Fatal Error: Failed to create React agent: {e}")
    exit(1)

# --- Memory and Runnable ---

# Store message histories keyed by session ID (simple in-memory store)
message_history_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieves or creates a chat message history for a given session ID."""
    if session_id not in message_history_store:
        message_history_store[session_id] = ChatMessageHistory()
        logging.info(f"Created new chat history for session: {session_id}")
    return message_history_store[session_id]

# Create the runnable with message history
try:
    agent_with_chat_history = RunnableWithMessageHistory(
        runnable=agent_executor, # Use 'runnable' keyword argument
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        # Define how session ID is extracted from the config
        history_factory_config=[
            ConfigurableFieldSpec(
                id="session_id", # This is the key expected in config['configurable']
                annotation=str,  # Type hint for the session ID
                name="Session ID", # Optional: User-friendly name
                description="Unique identifier for the chat session.", # Optional: Description
            ),
        ],
    )
    logging.info("RunnableWithMessageHistory created successfully.")
except Exception as e:
    logging.error(f"Failed to create RunnableWithMessageHistory: {e}", exc_info=True)
    print(f"Fatal Error: Failed to create RunnableWithMessageHistory: {e}")
    exit(1)


# --- Formatting and Utility ---

def format_final_answer(answer: str) -> str:
    """
    Cleans up the final answer from the agent.
    (Currently minimal, can be expanded to handle specific formatting quirks).
    """
    # Basic trimming
    formatted_answer = answer.strip()
    # Potentially add more complex regex or parsing if needed
    # Example: Replace escaped dollar signs if needed: formatted_answer = formatted_answer.replace('\\$', '$')
    return formatted_answer

# --- Chainlit Integration ---

# Custom handler to stream agent thoughts to Chainlit
class ChainlitCallbackHandler(BaseCallbackHandler):
    """Callback handler to stream agent thoughts and actions to Chainlit."""
    def __init__(self, msg: cl.Message):
        self.msg = msg
        # Initialize intermediate steps message here, potentially hidden initially
        self.steps_msg = cl.Message(content="", author="JEEnius (Thinking)", parent_id=self.msg.id)

    async def start_step(self, step_type: str, content: str):
        """Helper to update the intermediate steps message."""
        # Append new step content
        # Use Streaming Threshold to control updates if needed (cl.sleep)
        await self.steps_msg.stream_token(f"**{step_type}:**\n```\n{content}\n```\n\n")
        # Ensure message is visible
        # await self.steps_msg.update() # Might cause flickering, stream_token is often enough

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Called when LLM starts processing for thought/action plan."""
        # Indicate that the LLM is thinking about the next step
        # await self.steps_msg.stream_token("**Thinking...**\n")
        pass # Avoid too much noise, focus on actions/observations

    async def on_agent_action(self, action: Any, **kwargs: Any) -> Any:
        """Display agent actions clearly."""
        # Use action.log which contains the thought process leading to the action
        thought = action.log.strip().split("Action:")[0].strip() # Extract thought before action
        if thought.startswith("Thought:"): thought = thought[len("Thought:"):].strip()

        await self.steps_msg.stream_token(f"**Thought:**\n{thought}\n\n")
        await self.steps_msg.stream_token(f"**Action:** `{action.tool}`\n**Input:** `{action.tool_input}`\n\n")

    async def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Display tool outputs concisely."""
        # Shorten long outputs for display in the steps
        max_len = 300
        display_output = output[:max_len] + "..." if len(output) > max_len else output
        await self.steps_msg.stream_token(f"**Observation:**\n```\n{display_output}\n```\n\n")

    async def on_agent_finish(self, finish: Any, **kwargs: Any) -> Any:
         """Called when the agent finishes."""
         # Final answer is handled separately by the main on_message function
         # Optionally clear or update the steps message here
         await self.steps_msg.stream_token("**Finalizing Answer...**")
         # await self.steps_msg.update() # Send final update to steps message
         # Or delete it if preferred: await self.steps_msg.delete()
         pass


@cl.on_chat_start
async def on_chat_start():
    """Initializes the chat session."""
    # Generate a unique student ID for the session
    # Using a more robust method for uniqueness if multiple users run simultaneously
    student_id = f"student_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
    cl.user_session.set("student_id", student_id)
    logging.info(f"Chat started. Assigned student ID: {student_id}")

    # Ensure profile exists (or is created) but don't save it yet
    student_manager.load_profile(student_id)

    # Send welcome message
    await cl.Message(
        content=f"""# Welcome to JEEnius - Your JEE Math Assistant! 👋

I'm here to help you with your JEE Mathematics preparation. You can ask me to:

✅ Solve specific JEE math problems step-by-step.
✅ Explain mathematical concepts (like Calculus, Algebra, Geometry).
✅ Visualize functions or shapes (e.g., "Plot $y=x^2$").
✅ Get practice problems for a topic (e.g., "Suggest practice problems for Probability").
✅ Analyze your performance (Just ask "Analyze my performance").

*(Session ID: {student_id})*

How can I help you today?
""",
        author="JEEnius"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handles incoming user messages."""
    query = message.content
    student_id = cl.user_session.get("student_id")

    if not query:
        await cl.Message(content="Please enter a question or topic.", author="JEEnius").send()
        return

    logging.info(f"Received message from {student_id}: '{query}'")

    # Create the main response message placeholder
    msg = cl.Message(content="", author="JEEnius")
    await msg.send()

    # --- Agent Invocation with History and Callbacks ---
    final_answer = ""
    try:
        # Create callback handler for this message to show intermediate steps
        callback_handler = ChainlitCallbackHandler(msg)

        # Ensure agent_with_chat_history is not None
        if agent_with_chat_history is None:
             raise ValueError("Agent Runnable is not initialized.")

        # Invoke the agent asynchronously
        response = await agent_with_chat_history.ainvoke(
            {"input": query, "student_id": student_id}, # Pass necessary inputs
            config={
                "configurable": {"session_id": student_id}, # Pass session_id here
                "callbacks": [callback_handler], # Stream steps via callback
            }
        )
        final_answer = format_final_answer(response.get("output", "Sorry, I couldn't generate a response."))

        # --- Log Interaction (Simple Topic Extraction) ---
        # Basic keyword-based topic detection (can be improved with LLM)
        topic_keywords = {
            "calculus": ["integral", "derivative", "limit", "differential", "calculus"],
            "algebra": ["equation", "algebra", "polynomial", "matrix", "determinant", "complex number", "quadratic"],
            "geometry": ["triangle", "circle", "geometry", "coordinate", "vector", "conic", "3d"],
            "trigonometry": ["sin", "cos", "tan", "trigonometry", "angle", "inverse trig"],
            "probability": ["probability", "random", "distribution", "bayes", "permutation", "combination"],
            "vectors": ["vector", "scalar product", "cross product"],
        }
        detected_topic = "general" # Default topic
        query_lower = query.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_topic = topic
                break
        # Log interaction without assuming success/failure yet
        # Success/failure requires user feedback
        student_manager.log_interaction(student_id, detected_topic, success=None)
        logging.info(f"Automatically detected topic: {detected_topic} for query.")
        # TODO: Implement a feedback mechanism (e.g., buttons 👍/👎) to capture 'success' for performance tracking

    except Exception as e:
        error_msg = f"Sorry, I encountered an error processing your request: {str(e)}\n\nPlease try rephrasing your question or asking something different."
        logging.error(f"Agent execution failed for query '{query}' from {student_id}: {e}", exc_info=True)
        final_answer = error_msg # Set final answer to the error message


    # --- Update Final Message & Handle Visualizations ---
    # Update the main message with the final answer
    await msg.update(content=final_answer)

    # Check for and display base64 images generated by the visualization tool
    # This needs to happen *after* the final answer text is set
    img_pattern = r'!\[(.*?)\]\(data:image\/png;base64,([a-zA-Z0-9+/=]+)\)' # More specific base64 pattern
    matches = re.findall(img_pattern, final_answer)

    if matches:
        # Remove the markdown image tag(s) from the text message to avoid duplication
        text_without_image = re.sub(img_pattern, "\n_[Visualization displayed below]_\n", final_answer).strip()
        await msg.update(content=text_without_image) # Update message content without the image tag

        image_elements = []
        for alt_text, base64_data in matches:
            try:
                # Decode base64 data safely
                image_bytes = base64.b64decode(base64_data)
                img_element = cl.Image(
                    content=image_bytes,
                    name=f"{alt_text or 'visualization'}.png",
                    display="inline", # Display image below the message
                    size="large"
                )
                image_elements.append(img_element)
                logging.info(f"Prepared visualization element: {alt_text or 'visualization'}.png")

            except base64.binascii.Error as b64_err:
                 logging.error(f"Base64 decoding error for image: {b64_err}")
                 await cl.Message(content=f"(Error decoding visualization data)", parent_id=msg.id, author="JEEnius").send()
            except Exception as img_err:
                logging.error(f"Error processing or creating image element: {img_err}")
                await cl.Message(content=f"(Error preparing visualization: {img_err})", parent_id=msg.id, author="JEEnius").send()

        # Send images if any were successfully created
        if image_elements:
             # Send a new message containing only the image elements
             await cl.Message(
                 content="", # No text content needed here
                 elements=image_elements,
                 parent_id=msg.id, # Link it visually to the text response
                 author="JEEnius"
             ).send()


# --- Main block for testing without Chainlit ---
if __name__ == "__main__":
    # This block is for testing the agent logic directly if needed.
    # Chainlit provides the interactive environment.
    print("--- Starting JEEnius Agent Test (Command Line Interface) ---")
    print(f"Knowledge Base Status: {'Initialized' if knowledge_collection else 'Failed/Not Initialized'}")
    print(f"LLM Status: {'Initialized' if llm else 'Failed/Not Initialized'}")
    print(f"Agent Runnable Status: {'Initialized' if agent_with_chat_history else 'Failed/Not Initialized'}")
    print("-" * 60)

    if not all([knowledge_collection, llm, agent_with_chat_history]):
        print("One or more critical components failed to initialize. Exiting CLI test.")
        exit(1)

    test_student_id = "test_student_cli_001"
    print(f"Using Test Student ID: {test_student_id}")
    student_manager.load_profile(test_student_id) # Ensure profile exists

    # Example usage:
    test_queries = [
        "Hi JEEnius!",
        "Explain the concept of limits in calculus.",
        r"Solve the integral $\int x e^x dx$", # Use raw string or double backslash for LaTeX
        "Suggest some medium difficulty practice problems on matrices.",
        r"Plot the function $f(x) = \sin(x)$ from $x = -2\pi$ to $2\pi$", # Use raw string
        "Plot a circle centered at the origin with radius 3",
        "what is the capital of France?", # Test web search
        "Analyze my performance" # Will use test_student_id
    ]

    # Get history object for the test session
    cli_history = get_session_history(test_student_id)
    print(f"Initial History for {test_student_id}: {cli_history.messages}")

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        try:
            # Invoke directly using the agent_with_chat_history runnable
            # Note: .invoke is synchronous, .ainvoke is asynchronous (used in Chainlit)
            result = agent_with_chat_history.invoke(
                 {"input": query, "student_id": test_student_id}, # Pass necessary context
                 config={"configurable": {"session_id": test_student_id}}
            )
            output = result.get("output", "No output received.")
            print("\nAgent Output:")
            print(output)

            # Basic topic detection and logging for CLI test
            topic_keywords = { "calculus": ["integral", "derivative", "limit", "calculus"], "algebra": ["matrix", "algebra"], "geometry": ["circle", "triangle"], "general": []}
            detected_topic = "general"
            query_lower = query.lower()
            for topic, keywords in topic_keywords.items():
                 if any(keyword in query_lower for keyword in keywords):
                     detected_topic = topic
                     break
            student_manager.log_interaction(test_student_id, detected_topic)
            print(f"(Logged interaction on topic: {detected_topic})")

            # Print history after each turn
            # print(f"History after Query {i}: {cli_history.messages}\n")

        except Exception as e:
            print(f"\n--- ERROR during CLI test execution ---")
            print(f"Query: {query}")
            print(f"Error: {e}")
            logging.error(f"CLI Test Error for query '{query}': {e}", exc_info=True)
            print("-" * 40)

    print("\n--- Test Complete ---")

# To run with Chainlit:
# 1. Save this file (e.g., as `app.py`).
# 2. Make sure you have your .env file with OPENAI_API_KEY.
# 3. Create the directories: knowledge_base/documents, user_data/student_profiles, temp
# 4. (Optional) Place your PDF, TXT, CSV files in knowledge_base/documents. Sample files will be created if empty.
# 5. Install requirements: pip install -U langchain langchain-openai langchain-community chromadb python-dotenv matplotlib numpy sympy Pillow chainlit tiktoken duckduckgo-search openai
# 6. Run from terminal: `chainlit run app.py -w` (the -w flag enables auto-reload on code changes)
