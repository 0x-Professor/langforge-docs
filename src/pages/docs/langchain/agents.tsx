import { DocLayout } from '@/components/docs/DocLayout';
import { CodeBlock } from '@/components/CodeBlock';
import { Callout } from '@/components/docs/DocHeader';

export default function AgentsDocumentation() {
  const toc = [
    { id: 'overview', title: 'Overview', level: 2 },
    { id: 'agent-types', title: 'Agent Types', level: 2 },
    { id: 'tools', title: 'Tools', level: 3 },
    { id: 'custom-agents', title: 'Custom Agents', level: 2 },
    { id: 'multi-agent-systems', title: 'Multi-Agent Systems', level: 2 },
    { id: 'best-practices', title: 'Best Practices', level: 2 },
  ];

  const basicAgentExample = `from langchain.agents import initialize_agent, Tool
from langchain_community.llms import OpenAI
from langchain_community.utilities import GoogleSearchAPIWrapper

# Initialize the language model
llm = OpenAI(temperature=0)

# Set up Google Search as a tool
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events"
    )
]

# Initialize the agent
agent = initialize_agent(
    tools, 
    llm, 
    agent="zero-shot-react-description",
    verbose=True
)

# Run the agent
agent.run("What's the latest news about AI?")`;

  const customToolExample = `from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "Useful for when you need to perform mathematical calculations"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, a: int, b: int) -> str:
        """Add two numbers together."""
        return str(a + b)
    
    async def _arun(self, a: int, b: int) -> str:
        """Async version of the tool."""
        return self._run(a, b)

# Create an instance of the tool
calculator = CustomCalculatorTool()

# Use the tool
result = calculator.run({"a": 5, "b": 3})
print(f"5 + 3 = {result}")`;

  const customAgentExample = `from typing import List, Tuple, Any, Optional
from langchain.agents import BaseSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain

# 1. Define a custom prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Format the tools into a string
        tools_string = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # Format the prompt
        return self.template.format(
            tools=tools_string,
            **kwargs
        )

# 2. Create a custom output parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentAction | AgentFinish:
        # Parse the LLM output to determine the next action
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output
            )
        
        # Parse the action and action input
        action, action_input = llm_output.split("Action Input:")
        action = action.replace("Action:", "").strip()
        action_input = action_input.strip()
        
        return AgentAction(
            tool=action, 
            tool_input=action_input.strip("\""), 
            log=llm_output
        )

# 3. Create a custom agent
class CustomAgent(BaseSingleActionAgent):
    llm_chain: LLMChain
    output_parser: AgentOutputParser
    stop: List[str]
    
    @property
    def input_keys(self):
        return ["input"]
    
    def plan(self, intermediate_steps, **kwargs):
        # Get the output from the LLM
        output = self.llm_chain.run(**kwargs)
        
        # Parse the output
        return self.output_parser.parse(output)
    
    async def aplan(self, intermediate_steps, **kwargs):
        # Async version of plan
        output = await self.llm_chain.arun(**kwargs)
        return self.output_parser.parse(output)

# 4. Initialize the custom agent
def initialize_custom_agent(llm, tools, verbose=False):
    # Create the prompt template
    template = """
    You are a helpful AI assistant. You have access to the following tools:
    
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
    
    Begin!
    
    Question: {input}
    {agent_scratchpad}"""
    
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "agent_scratchpad"]
    )
    
    # Create the LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Create the output parser
    output_parser = CustomOutputParser()
    
    # Create and return the agent
    return CustomAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        verbose=verbose
    )`;

  const multiAgentExample = `from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI

# Define tools for the agents
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Useful for performing mathematical calculations"
    )
]

# Create a memory object
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the agents
researcher = initialize_agent(
    tools, 
    OpenAI(temperature=0), 
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

analyst = initialize_agent(
    tools, 
    OpenAI(temperature=0), 
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Simulate a conversation between agents
def simulate_conversation(question):
    # Researcher finds information
    research = researcher.run(f"Research information about: {question}")
    
    # Analyst processes the information
    analysis = analyst.run(f"Analyze this information: {research}")
    
    return analysis

# Run the simulation
result = simulate_conversation("latest advancements in renewable energy")
print(result)`;

  return (
    <DocLayout 
      title="LangChain Agents" 
      description="Learn how to build autonomous agents with LangChain that can use tools, make decisions, and solve complex tasks."
      toc={toc}
    >
      <section id="overview" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Overview</h2>
        <p className="mb-4">
          Agents in LangChain are systems that use a language model to determine a sequence of actions to take. 
          They can use tools, access memory, and make decisions based on the current state of the environment.
        </p>
        
        <Callout type="tip">
          <p>
            Agents are particularly useful for tasks that require dynamic decision-making and the ability to 
            use external tools or APIs. They can handle complex workflows that would be difficult to implement 
            with simple chains.
          </p>
        </Callout>
      </section>

      <section id="agent-types" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Agent Types</h2>
        <p className="mb-4">
          LangChain provides several built-in agent types, each designed for different use cases:
        </p>
        
        <div className="grid gap-4 md:grid-cols-2 mt-6">
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Zero-shot ReAct</h3>
            <p className="text-sm text-muted-foreground mb-2">
              Uses the ReAct framework to decide which tool to use based on the tool's description.
            </p>
            <pre className="text-xs bg-muted p-2 rounded">agent="zero-shot-react-description"</pre>
          </div>
          
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Self-ask with Search</h3>
            <p className="text-sm text-muted-foreground mb-2">
              Uses a single tool (typically a search tool) to find information and answer questions.
            </p>
            <pre className="text-xs bg-muted p-2 rounded">agent="self-ask-with-search"</pre>
          </div>
          
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Conversational</h3>
            <p className="text-sm text-muted-foreground mb-2">
              Designed for conversational agents that need to maintain context across multiple turns.
            </p>
            <pre className="text-xs bg-muted p-2 rounded">agent="conversational-react-description"</pre>
          </div>
          
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-2">Structured Chat</h3>
            <p className="text-sm text-muted-foreground mb-2">
              Handles multi-input tools and structured output better than other agent types.
            </p>
            <pre className="text-xs bg-muted p-2 rounded">agent="structured-chat-zero-shot-react-description"</pre>
          </div>
        </div>
        
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-2">Basic Agent Example</h3>
          <CodeBlock 
            code={basicAgentExample} 
            language="python" 
            title="Creating a Basic Agent with Tools"
          />
        </div>
      </section>

      <section id="tools" className="mb-8 ml-6">
        <h3 className="text-xl font-semibold mb-4">Tools</h3>
        <p className="mb-4">
          Tools are functions that agents can use to interact with the world. They can be anything from 
          search engines to calculators to custom functions.
        </p>
        
        <div className="mb-6">
          <h4 className="font-medium mb-2">Built-in Tools</h4>
          <ul className="list-disc pl-6 space-y-1 text-sm">
            <li><code>GoogleSearchAPIWrapper</code>: Perform web searches</li>
            <li><code>WolframAlphaQueryRun</code>: Access computational knowledge</li>
            <li><code>PythonREPLTool</code>: Execute Python code</li>
            <li><code>RequestsGetTool</code>: Make HTTP GET requests</li>
            <li><code>VectorDBQA</code>: Query a vector database</li>
          </ul>
        </div>
        
        <div>
          <h4 className="font-medium mb-2">Creating Custom Tools</h4>
          <CodeBlock 
            code={customToolExample} 
            language="python" 
            title="Creating a Custom Tool"
          />
        </div>
      </section>

      <section id="custom-agents" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Custom Agents</h2>
        <p className="mb-4">
          For more complex use cases, you can create custom agents by subclassing the base agent class. 
          This gives you full control over the agent's behavior.
        </p>
        
        <CodeBlock 
          code={customAgentExample} 
          language="python" 
          title="Creating a Custom Agent from Scratch"
        />
      </section>

      <section id="multi-agent-systems" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Multi-Agent Systems</h2>
        <p className="mb-4">
          You can create systems with multiple agents that work together to solve complex problems. 
          Each agent can have its own role, tools, and memory.
        </p>
        
        <CodeBlock 
          code={multiAgentExample} 
          language="python" 
          title="Creating a Multi-Agent System"
        />
      </section>

      <section id="best-practices" className="mb-8">
        <h2 className="text-2xl font-bold mb-4">Best Practices</h2>
        
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold mb-2">1. Choose the Right Agent Type</h3>
            <p className="text-muted-foreground">
              Select an agent type that matches your use case. For simple tool use, a zero-shot agent might be 
              sufficient, while complex workflows might require a custom agent.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">2. Provide Clear Tool Descriptions</h3>
            <p className="text-muted-foreground">
              Write clear and descriptive tool descriptions. The agent uses these descriptions to decide which 
              tool to use, so be specific about what each tool does and when it should be used.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">3. Handle Errors Gracefully</h3>
            <p className="text-muted-foreground">
              Implement error handling in your tools and agents to manage cases where tools fail or return 
              unexpected results. This makes your agent more robust in production.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">4. Use Memory Effectively</h3>
            <p className="text-muted-foreground">
              For conversational agents, use memory to maintain context across multiple turns. This allows 
              the agent to reference previous parts of the conversation.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">5. Monitor and Evaluate</h3>
            <p className="text-muted-foreground">
              Track how your agent performs in production. Monitor metrics like tool usage, success rates, 
              and user satisfaction to identify areas for improvement.
            </p>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2">6. Limit Tool Access</h3>
            <p className="text-muted-foreground">
              Only give your agent access to the tools it needs. This reduces the complexity of the agent's 
              decision-making and improves security.
            </p>
          </div>
        </div>
      </section>
    </DocLayout>
  );
}
