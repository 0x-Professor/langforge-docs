import { DocSection, FeatureCard, QuickStart } from '@/components/DocSection';
import { CodeBlock } from '@/components/CodeBlock';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Bot, Users, MessageSquare, Network, Zap, Shield, Database } from 'lucide-react';

export const AgentArchitectureSection = () => {
  const supervisorCode = `from langgraph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List
import asyncio

class MultiAgentState(TypedDict):
    messages: List[dict]
    task: str
    assigned_agent: str
    results: dict
    final_answer: str

# Define specialized agents
class ResearchAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.name = "researcher"
    
    async def execute(self, task: str) -> str:
        prompt = f"""You are a research specialist. Your task is: {task}
        
        Provide comprehensive research findings with sources and data."""
        
        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        return response.content

class AnalysisAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4", temperature=0.2)
        self.name = "analyst"
    
    async def execute(self, data: str) -> str:
        prompt = f"""You are an analysis specialist. Analyze this data: {data}
        
        Provide insights, patterns, and conclusions."""
        
        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        return response.content

class SupervisorAgent:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4", temperature=0)
        self.agents = {
            "researcher": ResearchAgent(),
            "analyst": AnalysisAgent()
        }
    
    async def route_task(self, state: MultiAgentState) -> MultiAgentState:
        task = state["task"]
        
        # Determine which agent should handle the task
        routing_prompt = f"""Given this task: {task}
        
        Which agent should handle this?
        - researcher: for gathering information and data
        - analyst: for analyzing data and drawing conclusions
        
        Respond with just the agent name."""
        
        response = await self.model.ainvoke([HumanMessage(content=routing_prompt)])
        assigned_agent = response.content.strip().lower()
        
        return {**state, "assigned_agent": assigned_agent}
    
    async def execute_task(self, state: MultiAgentState) -> MultiAgentState:
        agent_name = state["assigned_agent"]
        task = state["task"]
        
        if agent_name in self.agents:
            result = await self.agents[agent_name].execute(task)
            results = state.get("results", {})
            results[agent_name] = result
            
            return {**state, "results": results}
        
        return state
    
    async def synthesize_results(self, state: MultiAgentState) -> MultiAgentState:
        results = state["results"]
        task = state["task"]
        
        synthesis_prompt = f"""Original task: {task}
        
        Agent results:
        {results}
        
        Synthesize these results into a comprehensive final answer."""
        
        response = await self.model.ainvoke([HumanMessage(content=synthesis_prompt)])
        
        return {**state, "final_answer": response.content}

# Build the supervisor workflow
def create_supervisor_workflow():
    supervisor = SupervisorAgent()
    
    workflow = StateGraph(MultiAgentState)
    
    # Add nodes
    workflow.add_node("route", supervisor.route_task)
    workflow.add_node("execute", supervisor.execute_task)
    workflow.add_node("synthesize", supervisor.synthesize_results)
    
    # Add edges
    workflow.add_edge(START, "route")
    workflow.add_edge("route", "execute")
    workflow.add_edge("execute", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()

# Usage
async def main():
    workflow = create_supervisor_workflow()
    
    result = await workflow.ainvoke({
        "task": "Analyze the impact of AI on job markets",
        "messages": [],
        "results": {}
    })
    
    print(result["final_answer"])

if __name__ == "__main__":
    asyncio.run(main())`;

  const communicationCode = `from langgraph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List, Dict
import json
import asyncio

class AgentMessage(TypedDict):
    sender: str
    recipient: str
    content: str
    message_type: str  # "request", "response", "broadcast"
    timestamp: float

class SharedState(TypedDict):
    messages: List[dict]
    agent_messages: List[AgentMessage]
    shared_memory: Dict[str, any]
    active_agents: List[str]

class CommunicatingAgent:
    def __init__(self, name: str, specialization: str):
        self.name = name
        self.specialization = specialization
        self.model = ChatOpenAI(model="gpt-4")
        self.message_queue = []
    
    async def send_message(self, recipient: str, content: str, 
                          message_type: str = "request") -> AgentMessage:
        """Send a message to another agent."""
        import time
        
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=message_type,
            timestamp=time.time()
        )
        
        return message
    
    async def receive_message(self, message: AgentMessage) -> str:
        """Process incoming message and generate response."""
        prompt = f"""You are {self.name}, specialized in {self.specialization}.
        
        You received this message from {message['sender']}:
        {message['content']}
        
        Generate an appropriate response based on your specialization."""
        
        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        return response.content
    
    async def process_task(self, task: str, shared_memory: Dict) -> str:
        """Process a task with access to shared memory."""
        context = ""
        if shared_memory:
            context = f"\\nShared context: {json.dumps(shared_memory, indent=2)}"
        
        prompt = f"""You are {self.name}, specialized in {self.specialization}.
        
        Task: {task}{context}
        
        Complete this task and indicate if you need help from other agents."""
        
        response = await self.model.ainvoke([HumanMessage(content=prompt)])
        return response.content

# Multi-agent communication system
class AgentCommunicationSystem:
    def __init__(self):
        self.agents = {}
        self.message_history = []
        self.shared_memory = {}
    
    def add_agent(self, agent: CommunicatingAgent):
        self.agents[agent.name] = agent
    
    async def route_message(self, message: AgentMessage, state: SharedState) -> SharedState:
        """Route message to recipient agent."""
        if message["recipient"] in self.agents:
            response_content = await self.agents[message["recipient"]].receive_message(message)
            
            # Create response message
            response = await self.agents[message["recipient"]].send_message(
                recipient=message["sender"],
                content=response_content,
                message_type="response"
            )
            
            # Update state
            agent_messages = state["agent_messages"] + [message, response]
            return {**state, "agent_messages": agent_messages}
        
        return state
    
    async def coordinate_agents(self, task: str) -> str:
        """Coordinate multiple agents to complete a complex task."""
        # Initialize workflow
        workflow = StateGraph(SharedState)
        
        # Agent coordination node
        async def agent_coordination(state: SharedState) -> SharedState:
            task = "Analyze market trends and provide investment recommendations"
            
            # Assign tasks to specialized agents
            if "market_analyst" in self.agents:
                analysis = await self.agents["market_analyst"].process_task(
                    "Analyze current market trends", state["shared_memory"]
                )
                state["shared_memory"]["market_analysis"] = analysis
            
            if "risk_assessor" in self.agents:
                risk_assessment = await self.agents["risk_assessor"].process_task(
                    "Assess investment risks based on market analysis", 
                    state["shared_memory"]
                )
                state["shared_memory"]["risk_assessment"] = risk_assessment
            
            if "financial_advisor" in self.agents:
                recommendations = await self.agents["financial_advisor"].process_task(
                    "Provide investment recommendations", state["shared_memory"]
                )
                state["shared_memory"]["final_recommendations"] = recommendations
            
            return state
        
        workflow.add_node("coordinate", agent_coordination)
        workflow.add_edge(START, "coordinate")
        workflow.add_edge("coordinate", END)
        
        compiled_workflow = workflow.compile()
        
        result = await compiled_workflow.ainvoke({
            "messages": [],
            "agent_messages": [],
            "shared_memory": {"initial_task": task},
            "active_agents": list(self.agents.keys())
        })
        
        return result["shared_memory"].get("final_recommendations", "No recommendations generated")

# Example usage
async def main():
    # Create communication system
    comm_system = AgentCommunicationSystem()
    
    # Create specialized agents
    market_analyst = CommunicatingAgent("market_analyst", "financial market analysis")
    risk_assessor = CommunicatingAgent("risk_assessor", "investment risk assessment")
    financial_advisor = CommunicatingAgent("financial_advisor", "investment recommendations")
    
    # Add agents to system
    comm_system.add_agent(market_analyst)
    comm_system.add_agent(risk_assessor)
    comm_system.add_agent(financial_advisor)
    
    # Coordinate agents
    result = await comm_system.coordinate_agents(
        "Provide comprehensive investment analysis and recommendations"
    )
    
    print("Final Recommendations:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())`;

  const distributedCode = `from langgraph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import asyncio
import json
from typing import TypedDict, List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class Task:
    id: str
    description: str
    priority: int
    assigned_agent: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None

class DistributedState(TypedDict):
    task_queue: List[Task]
    active_tasks: Dict[str, Task]
    completed_tasks: List[Task]
    agent_workloads: Dict[str, int]
    shared_memory: Dict[str, any]

class DistributedAgent:
    def __init__(self, agent_id: str, capabilities: List[str], max_concurrent_tasks: int = 3):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.max_concurrent_tasks = max_concurrent_tasks
        self.model = ChatOpenAI(model="gpt-4")
        self.current_tasks = {}
    
    async def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle the task."""
        if len(self.current_tasks) >= self.max_concurrent_tasks:
            return False
        
        # Check if task description matches capabilities
        task_lower = task.description.lower()
        return any(capability in task_lower for capability in self.capabilities)
    
    async def execute_task(self, task: Task, shared_context: Dict) -> Task:
        """Execute a task with shared context."""
        try:
            task.status = "in_progress"
            task.assigned_agent = self.agent_id
            self.current_tasks[task.id] = task
            
            # Prepare context
            context = f"Shared context: {json.dumps(shared_context, indent=2)}" if shared_context else ""
            
            prompt = f"""You are Agent {self.agent_id} with capabilities: {', '.join(self.capabilities)}
            
            Task: {task.description}
            Priority: {task.priority}
            
            {context}
            
            Execute this task and provide detailed results."""
            
            response = await self.model.ainvoke([
                {"role": "user", "content": prompt}
            ])
            
            task.result = response.content
            task.status = "completed"
            
        except Exception as e:
            task.status = "failed"
            task.result = f"Error: {str(e)}"
        
        finally:
            if task.id in self.current_tasks:
                del self.current_tasks[task.id]
        
        return task

class DistributedAgentOrchestrator:
    def __init__(self):
        self.agents: Dict[str, DistributedAgent] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def add_agent(self, agent: DistributedAgent):
        """Add an agent to the system."""
        self.agents[agent.agent_id] = agent
    
    async def distribute_tasks(self, state: DistributedState) -> DistributedState:
        """Distribute tasks to available agents."""
        task_queue = state["task_queue"].copy()
        active_tasks = state["active_tasks"].copy()
        agent_workloads = state["agent_workloads"].copy()
        
        # Sort tasks by priority
        task_queue.sort(key=lambda x: x.priority, reverse=True)
        
        for task in task_queue[:]:
            # Find best agent for task
            best_agent = None
            for agent_id, agent in self.agents.items():
                if await agent.can_handle_task(task):
                    current_workload = agent_workloads.get(agent_id, 0)
                    if best_agent is None or current_workload < agent_workloads.get(best_agent, float('inf')):
                        best_agent = agent_id
            
            if best_agent:
                # Assign task
                task.assigned_agent = best_agent
                active_tasks[task.id] = task
                agent_workloads[best_agent] = agent_workloads.get(best_agent, 0) + 1
                task_queue.remove(task)
        
        return {
            **state,
            "task_queue": task_queue,
            "active_tasks": active_tasks,
            "agent_workloads": agent_workloads
        }
    
    async def execute_distributed_tasks(self, state: DistributedState) -> DistributedState:
        """Execute tasks in parallel across agents."""
        active_tasks = state["active_tasks"]
        completed_tasks = state["completed_tasks"].copy()
        agent_workloads = state["agent_workloads"].copy()
        shared_memory = state["shared_memory"]
        
        # Execute tasks concurrently
        execution_futures = []
        
        for task_id, task in active_tasks.items():
            if task.assigned_agent and task.status == "pending":
                agent = self.agents[task.assigned_agent]
                future = asyncio.create_task(agent.execute_task(task, shared_memory))
                execution_futures.append((task_id, future))
        
        # Wait for task completion
        completed_task_ids = []
        for task_id, future in execution_futures:
            try:
                completed_task = await future
                completed_tasks.append(completed_task)
                completed_task_ids.append(task_id)
                
                # Update shared memory with results
                shared_memory[f"task_result_{task_id}"] = completed_task.result
                
                # Update workload
                if completed_task.assigned_agent:
                    agent_workloads[completed_task.assigned_agent] -= 1
                
            except Exception as e:
                print(f"Task {task_id} failed: {e}")
        
        # Remove completed tasks from active
        for task_id in completed_task_ids:
            if task_id in active_tasks:
                del active_tasks[task_id]
        
        return {
            **state,
            "active_tasks": active_tasks,
            "completed_tasks": completed_tasks,
            "agent_workloads": agent_workloads,
            "shared_memory": shared_memory
        }

# Example usage
async def main():
    # Create orchestrator
    orchestrator = DistributedAgentOrchestrator()
    
    # Create specialized agents
    research_agent = DistributedAgent("researcher_001", ["research", "data", "analysis"])
    writing_agent = DistributedAgent("writer_001", ["writing", "content", "summary"])
    coding_agent = DistributedAgent("coder_001", ["programming", "code", "development"])
    
    # Add agents
    orchestrator.add_agent(research_agent)
    orchestrator.add_agent(writing_agent)
    orchestrator.add_agent(coding_agent)
    
    # Create tasks
    tasks = [
        Task("1", "Research latest AI trends", 5),
        Task("2", "Write summary of research findings", 3),
        Task("3", "Code a simple ML model", 4),
        Task("4", "Analyze market data for AI companies", 5),
        Task("5", "Write technical documentation", 2)
    ]
    
    # Create workflow
    workflow = StateGraph(DistributedState)
    
    workflow.add_node("distribute", orchestrator.distribute_tasks)
    workflow.add_node("execute", orchestrator.execute_distributed_tasks)
    
    workflow.add_edge(START, "distribute")
    workflow.add_edge("distribute", "execute")
    workflow.add_edge("execute", END)
    
    compiled_workflow = workflow.compile()
    
    # Execute distributed processing
    result = await compiled_workflow.ainvoke({
        "task_queue": tasks,
        "active_tasks": {},
        "completed_tasks": [],
        "agent_workloads": {},
        "shared_memory": {}
    })
    
    print("Completed Tasks:")
    for task in result["completed_tasks"]:
        print(f"Task {task.id}: {task.status}")
        print(f"Result: {task.result[:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())`;

  return (
    <DocSection
      id="agent-architecture"
      title="Agent-to-Agent Architecture"
      description="Build sophisticated multi-agent systems with communication, coordination, and distributed processing capabilities."
      badges={["Advanced", "Multi-Agent", "Distributed"]}
      externalLinks={[
        { title: "Multi-Agent Papers", url: "https://arxiv.org/search/?query=multi-agent+systems" },
        { title: "LangGraph Agents", url: "https://langchain-ai.github.io/langgraph/tutorials/" }
      ]}
    >
      <div className="space-y-8">
        {/* Core Concepts */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Multi-Agent Patterns</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <FeatureCard
              icon={<Users className="w-6 h-6" />}
              title="Supervisor Pattern"
              description="Central coordinator that manages and delegates tasks to specialized sub-agents."
              features={[
                "Task routing and delegation",
                "Result aggregation",
                "Error handling",
                "Progress monitoring"
              ]}
            />
            <FeatureCard
              icon={<MessageSquare className="w-6 h-6" />}
              title="Peer-to-Peer Communication"
              description="Agents communicate directly with each other to share information and coordinate."
              features={[
                "Direct messaging",
                "Shared memory",
                "Event broadcasting",
                "Conflict resolution"
              ]}
            />
            <FeatureCard
              icon={<Network className="w-6 h-6" />}
              title="Distributed Processing"
              description="Parallel execution of tasks across multiple agents with load balancing."
              features={[
                "Load distribution",
                "Parallel execution",
                "Fault tolerance",
                "Dynamic scaling"
              ]}
            />
            <FeatureCard
              icon={<Bot className="w-6 h-6" />}
              title="Specialized Agents"
              description="Domain-specific agents with focused capabilities and expertise."
              features={[
                "Domain expertise",
                "Skill specialization",
                "Tool integration",
                "Context awareness"
              ]}
            />
            <FeatureCard
              icon={<Database className="w-6 h-6" />}
              title="Shared Memory"
              description="Common knowledge base accessible by all agents for coordination."
              features={[
                "Global state management",
                "Knowledge sharing",
                "Persistent memory",
                "Consistency guarantees"
              ]}
            />
            <FeatureCard
              icon={<Shield className="w-6 h-6" />}
              title="Security & Isolation"
              description="Secure agent interactions with proper access controls and sandboxing."
              features={[
                "Agent authentication",
                "Permission controls",
                "Resource isolation",
                "Audit logging"
              ]}
            />
          </div>
        </div>

        {/* Implementation Examples */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Implementation Patterns</h2>
          <Tabs defaultValue="supervisor" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="supervisor">Supervisor Pattern</TabsTrigger>
              <TabsTrigger value="communication">Agent Communication</TabsTrigger>
              <TabsTrigger value="distributed">Distributed Processing</TabsTrigger>
            </TabsList>
            
            <TabsContent value="supervisor" className="space-y-4">
              <CodeBlock
                title="Supervisor-Based Multi-Agent System"
                language="python"
                code={supervisorCode}
              />
            </TabsContent>
            
            <TabsContent value="communication" className="space-y-4">
              <CodeBlock
                title="Agent-to-Agent Communication"
                language="python"
                code={communicationCode}
              />
            </TabsContent>
            
            <TabsContent value="distributed" className="space-y-4">
              <CodeBlock
                title="Distributed Agent Processing"
                language="python"
                code={distributedCode}
              />
            </TabsContent>
          </Tabs>
        </div>

        {/* Architecture Patterns */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">System Architectures</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Hierarchical Architecture</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="bg-gradient-section p-4 rounded-lg">
                  <div className="text-center space-y-2">
                    <div className="px-3 py-1 bg-primary/20 rounded">Supervisor Agent</div>
                    <div className="flex justify-center">↓</div>
                    <div className="flex justify-center space-x-2">
                      <div className="px-2 py-1 bg-secondary rounded text-sm">Agent A</div>
                      <div className="px-2 py-1 bg-secondary rounded text-sm">Agent B</div>
                      <div className="px-2 py-1 bg-secondary rounded text-sm">Agent C</div>
                    </div>
                  </div>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium">Benefits</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Clear command structure</li>
                    <li>• Centralized coordination</li>
                    <li>• Easy to monitor and debug</li>
                    <li>• Simple error handling</li>
                  </ul>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Mesh Architecture</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="bg-gradient-section p-4 rounded-lg">
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="px-2 py-1 bg-primary/20 rounded text-sm">Agent A</div>
                    <div className="px-2 py-1 bg-primary/20 rounded text-sm">Agent B</div>
                    <div className="px-2 py-1 bg-primary/20 rounded text-sm">Agent C</div>
                  </div>
                  <div className="text-center text-sm text-muted-foreground mt-2">
                    ↕ All agents communicate directly ↕
                  </div>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium">Benefits</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Flexible communication</li>
                    <li>• No single point of failure</li>
                    <li>• Dynamic collaboration</li>
                    <li>• Emergent behaviors</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Use Cases */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Common Use Cases</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <QuickStart
              title="Research & Analysis Pipeline"
              description="Multi-agent system for comprehensive research and analysis workflows."
              steps={[
                "Research Agent gathers information from multiple sources",
                "Analysis Agent processes and identifies patterns",
                "Summary Agent creates structured reports",
                "Review Agent validates findings and recommendations"
              ]}
            />
            <QuickStart
              title="Customer Service Automation"
              description="Specialized agents handling different aspects of customer support."
              steps={[
                "Routing Agent classifies and routes inquiries",
                "Support Agent handles common questions",
                "Technical Agent resolves complex issues",
                "Escalation Agent manages human handoffs"
              ]}
            />
          </div>
        </div>

        {/* Best Practices */}
        <Card className="shadow-card border-l-4 border-l-primary">
          <CardHeader>
            <CardTitle>Multi-Agent Best Practices</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="font-medium">Design Principles</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Keep agents focused and specialized</li>
                  <li>• Design clear communication protocols</li>
                  <li>• Implement proper error handling and recovery</li>
                  <li>• Plan for scalability and load distribution</li>
                </ul>
              </div>
              <div className="space-y-3">
                <h4 className="font-medium">Operational</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Monitor agent performance and health</li>
                  <li>• Implement proper logging and observability</li>
                  <li>• Use circuit breakers for fault tolerance</li>
                  <li>• Ensure consistent state management</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </DocSection>
  );
};