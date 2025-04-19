import os
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
from langchain_core.tools import BaseTool, tool
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import re

# Load environment variables
load_dotenv()

# Initialize Tavily Search API
tavily_api_key = os.getenv("TAVILY_API_KEY")
search =TavilySearch(
    max_results=5,
    topic="general",)

# Initialize OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)

# Define search tool
@tool
def search_jee_question(query: str) -> str:
    """Search for JEE math questions and solutions."""
    results = search.results(query + " JEE math problem solution step by step")
    
    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append(
            f"Title: {result['title']}\n"
            f"Content: {result['content']}\n"
            f"URL: {result['url']}\n"
        )
    
    return "\n\n".join(formatted_results) if formatted_results else "No results found."

# Define math solution tool
@tool
def solve_math_problem(problem: str) -> str:
    """Solve a JEE math problem step by step."""
    solve_prompt = PromptTemplate(
        template="""
        You are an expert mathematics tutor specializing in JEE (Joint Entrance Examination) problems.


      
        
        Solve the following JEE math problem step by step:
        
        {problem}
        
        Follow these steps:
        1. Understand the problem and identify the relevant mathematical concepts
        2. Break down the problem into manageable steps
        3. Solve each step systematically, showing all work
        4. Explain your reasoning clearly, using LaTeX notation for mathematical expressions
        5. Verify the solution
        6. Provide a concise final answer
        
        Format your solution as follows:
        ```
        ## Problem Understanding
        [Explain what the problem is asking and identify key concepts]
        
        ## Step 1: [Step Name]
        [Detailed explanation with LaTeX equations]
        
        ## Step 2: [Step Name]
        [Detailed explanation with LaTeX equations]
        
        ... (additional steps as needed)
        
        ## Final Answer
        [Concise final answer with LaTeX notation if needed]
        ```
        
        Make sure all equations are properly formatted using LaTeX notation (e.g., $x^2 + y^2 = z^2$).
        
        """,
        input_variables=["problem"],
    )
    
    return llm.invoke(solve_prompt.format(problem=problem)).content

# Define ReACT agent prompt
react_prompt = PromptTemplate(
    template="""You are an expert JEE Math Problem Solver agent. You help students solve complex JEE (Joint Entrance Examination) mathematics problems.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

When providing your Final Answer:
1. Format all mathematical equations using LaTeX notation
2. Present the solution in a structured format with clear steps
3. Explain each step thoroughly so students can learn
4. Include a brief summary of the solution approach
5. Include links to relevant learning resources when available

Begin!
use {agent_scratchpad} to reason about the tools and how to use them. 

Question: {input}
Thought: """,
    input_variables=["input", "tools", "tool_names"],
)

# Create agent
tools = [search_jee_question, solve_math_problem]
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6
)

# Format the output
def format_solution(solution: str) -> str:
    """Format the solution to make it more readable."""
    # Extract sources/links
    links = re.findall(r'https?://\S+', solution)
    
    # Clean up the solution
    formatted_solution = solution
    
    # Add sources section if links were found
    if links:
        formatted_solution += "\n\n## Related Resources\n"
        for i, link in enumerate(links, 1):
            formatted_solution += f"{i}. [{link}]({link})\n"
    
    return formatted_solution

# Main function to use the agent
def solve_jee_math_problem(question: str) -> str:
    """
    Solves a JEE math problem by first searching for existing solutions,
    then solving step by step if needed.
    
    Args:
        question: The JEE math question to solve
        
    Returns:
        Formatted solution with step-by-step explanation
    """
    try:
        result = agent_executor.invoke({"input": question})
        return format_solution(result["output"])
    except Exception as e:
        return f"Error solving the problem: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example JEE math problems
    questions = [
        "Find the value of the integral ∫(1 to 2) (x^2 + 3x) dx",
        "If f(x) = x^3 - 3x^2 + 2x - 5, find the critical points of the function",
        "The probability that a student solves a problem correctly is 0.7. If 5 students attempt the same problem independently, what is the probability that at least 3 students solve it correctly?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*50}\nExample {i}: {question}\n{'='*50}\n")
        solution = solve_jee_math_problem(question)
        print(solution)
        print("\n")