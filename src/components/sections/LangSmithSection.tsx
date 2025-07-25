import { DocSection, FeatureCard, QuickStart } from '@/components/DocSection';
import { CodeBlock } from '@/components/CodeBlock';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Settings, BarChart3, Eye, AlertTriangle, Zap, Target, Clock, CheckCircle, AlertCircle, GitBranch, Code2, Server, BookOpen, Terminal, BarChart, AlertOctagon, Info } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

export const LangSmithSection = () => {
  const setupCode = `import os
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain.callbacks.tracers import LangChainTracer

# Set up LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "your-project-name"

# Initialize client
client = Client()

# Create chat model with tracing
model = ChatOpenAI(
    model="gpt-4",
    callbacks=[LangChainTracer()]
)

# Your traces will automatically appear in LangSmith
response = model.invoke("Hello, world!")`;

  const evaluationCode = `# First, install required packages
# pip install langsmith langchain-openai python-dotenv numpy scikit-learn

import os
from typing import Dict, List, Any, Optional
from langsmith import Client, RunEvaluator, EvaluationResult
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import StringEvaluator
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load environment variables
load_dotenv()

# Initialize clients
client = Client()
llm = ChatOpenAI(model="gpt-4", temperature=0)
embeddings = OpenAIEmbeddings()

class CustomEvaluator(RunEvaluator):
    """Custom evaluator that combines multiple evaluation metrics."""
    
    def __init__(self):
        # Initialize evaluators
        self.embedding_evaluator = load_evaluator("embedding_distance", embeddings=embeddings)
        self.criteria_evaluator = load_evaluator("criteria", llm=llm)
    
    def evaluate_run(
        self, run: Any, example: Optional[Dict] = None
    ) -> EvaluationResult:
        """Evaluate an individual run."""
        if run.error is not None:
            return EvaluationResult(key="error", score=0, comment=f"Run failed: {run.error}")
        
        try:
            # Get prediction and reference
            prediction = run.outputs.get("output", "")
            reference = example.outputs.get("expected", "") if example else ""
            
            # Calculate embedding similarity
            embedding_result = self.embedding_evaluator.evaluate_strings(
                prediction=prediction,
                reference=reference
            })
            
            # Evaluate against criteria
            criteria_result = self.criteria_evaluator.evaluate_strings(
                prediction=prediction,
                input=example.inputs.get("question", "") if example else "",
                criteria={
                    "relevance": "The response should directly address the question",
                    "correctness": "The response should be factually accurate",
                    "completeness": "The response should fully answer the question"
                }
            })
            
            # Calculate scores (normalize to 0-1 range)
            embedding_score = 1 - embedding_result["score"]  # Convert distance to similarity
            criteria_scores = criteria_result.get("results", {})
            
            # Calculate final score (weighted average)
            final_score = (
                0.4 * embedding_score +
                0.2 * criteria_scores.get("relevance", 0.5) +
                0.2 * criteria_scores.get("correctness", 0.5) +
                0.2 * criteria_scores.get("completeness", 0.5)
            )
            
            return EvaluationResult(
                key="custom_eval",
                score=final_score,
                comment=(
                    f"Embedding similarity: {embedding_score:.2f}\n"
                    f"Relevance: {criteria_scores.get('relevance', 0.5):.2f}\n"
                    f"Correctness: {criteria_scores.get('correctness', 0.5):.2f}\n"
                    f"Completeness: {criteria_scores.get('completeness', 0.5):.2f}"
                )
            )
            
        except Exception as e:
            return EvaluationResult(
                key="error",
                score=0,
                comment=f"Evaluation failed: {str(e)}"
            )

def evaluate_chatbot(
    dataset_name: str = "chatbot-eval",
    model_name: str = "gpt-4",
    max_examples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate a chatbot model on a dataset.
    
    Args:
        dataset_name: Name of the dataset to evaluate on
        model_name: Name of the model to evaluate
        max_examples: Maximum number of examples to evaluate (None for all)
        
    Returns:
        Dictionary containing evaluation results
    """
    try:
        # Initialize model
        model = ChatOpenAI(model=model_name, temperature=0.7)
        
        # Define the chatbot function
        def chatbot(inputs: Dict) -> Dict:
            """Simple chatbot implementation."""
            try:
                response = model.invoke(inputs["question"])
                return {"output": response.content}
            except Exception as e:
                return {"output": f"Error: {str(e)}", "error": True}
        
        # Configure evaluation
        eval_config = RunEvalConfig(
            evaluators=[
                # Built-in evaluators
                RunEvalConfig.LabeledScoreString(
                    criteria={
                        "relevance": "How relevant is the response to the question?",
                        "correctness": "Is the response factually correct?",
                        "completeness": "Does the response fully answer the question?"
                    },
                    llm=llm
                ),
                # Custom evaluator
                CustomEvaluator()
            ],
            eval_llm=llm,
            reference_key="expected"
        )
        
        # Run evaluation
        results = run_on_dataset(
            client=client,
            dataset_name=dataset_name,
            llm_or_chain_factory=chatbot,
            evaluation=eval_config,
            max_examples=max_examples,
            project_name=f"{dataset_name}-eval-{model_name}",
            verbose=True
        )
        
        # Calculate aggregate metrics
        metrics = {
            "model": model_name,
            "dataset": dataset_name,
            "total_examples": results.get("total_examples", 0),
            "success_rate": results.get("success_rate", 0),
            "avg_score": results.get("avg_score", 0),
            "evaluation_metrics": {}
        }
        
        # Add detailed metrics from evaluators
        for eval_name, eval_results in results.get("evaluator_results", {}).items():
            if hasattr(eval_results, "scores"):
                scores = [r.get("score", 0) for r in eval_results.scores if r.get("score") is not None]
                if scores:
                    metrics["evaluation_metrics"][eval_name] = {
                        "mean_score": np.mean(scores),
                        "median_score": np.median(scores),
                        "min_score": min(scores),
                        "max_score": max(scores),
                        "std_dev": np.std(scores) if len(scores) > 1 else 0
                    }
        
        return metrics
        
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}

# Example usage
if __name__ == "__main__":
    # Configuration
    DATASET_NAME = "chatbot-eval"
    MODEL_NAME = "gpt-4"
    MAX_EXAMPLES = 10  # Set to None to evaluate all examples
    
    print(f"Starting evaluation of {MODEL_NAME} on {DATASET_NAME} dataset...")
    
    # Run evaluation
    results = evaluate_chatbot(
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        max_examples=MAX_EXAMPLES
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Model: {results.get('model')}")
    print(f"Dataset: {results.get('dataset')}")
    print(f"Examples evaluated: {results.get('total_examples')}")
    print(f"Success rate: {results.get('success_rate', 0):.1%}")
    print(f"Average score: {results.get('avg_score', 0):.2f}")
    
    # Print detailed metrics
    print("\nDetailed Metrics:")
    for eval_name, metrics in results.get("evaluation_metrics", {}).items():
        print(f"\n{eval_name}:")
        print(f"  Mean Score: {metrics.get('mean_score', 0):.3f}")
        print(f"  Score Range: {metrics.get('min_score', 0):.3f} - {metrics.get('max_score', 0):.3f}")
        print(f"  Std Dev: {metrics.get('std_dev', 0):.3f}")
    
    print("\nEvaluation complete!")`;

  const monitoringCode = `# First, install required packages
# pip install langsmith langchain-openai python-dotenv

import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from langsmith import Client
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LangSmith client
client = Client()

class ProductionMonitor:
    def __init__(self, project_name: str):
        """Initialize the production monitor with a project name."""
        self.client = Client()
        self.project_name = project_name
        
    def log_run(
        self, 
        inputs: Dict[str, Any], 
        outputs: Dict[str, Any], 
        run_type: str = "llm",
        **metadata
    ) -> str:
        """
        Log a production run to LangSmith.
        
        Args:
            inputs: Dictionary of input parameters
            outputs: Dictionary of output values
            run_type: Type of run (e.g., 'llm', 'chain', 'agent')
            **metadata: Additional metadata to log
            
        Returns:
            str: The ID of the created run
        """
        try:
            run = self.client.create_run(
                name=f"production-{run_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                project_name=self.project_name,
                run_type=run_type,
                inputs=inputs,
                outputs=outputs,
                extra={
                    **metadata,
                    "environment": os.getenv("ENV", "development"),
                    "version": os.getenv("APP_VERSION", "1.0.0")
                }
            )
            return str(run.id)
        except Exception as e:
            print(f"Error logging run: {str(e)}")
            raise
    
    def log_feedback(
        self, 
        run_id: str, 
        score: float, 
        feedback: str = "",
        key: str = "user_satisfaction"
    ) -> None:
        """
        Log user feedback for a run.
        
        Args:
            run_id: The ID of the run to log feedback for
            score: Numeric score (typically 0-1)
            feedback: Optional text feedback
            key: The feedback key/name
        """
        try:
            self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=feedback
            )
        except Exception as e:
            print(f"Error logging feedback: {str(e)}")
            raise
    
    def get_project_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get statistics for the project.
        
        Args:
            days: Number of days to look back for statistics
            
        Returns:
            Dictionary containing project statistics
        """
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Get runs for the time period
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp()
            ))
            
            if not runs:
                return {
                    "total_runs": 0,
                    "error_rate": 0,
                    "avg_latency": 0,
                    "success_rate": 0,
                    "total_tokens": 0
                }
            
            # Calculate statistics
            successful_runs = [r for r in runs if not getattr(r, 'error', None)]
            error_rate = 1 - (len(successful_runs) / len(runs)) if runs else 0
            
            latencies = [r.latency for r in runs if hasattr(r, 'latency') and r.latency is not None]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            # Estimate token usage (this is a simplification)
            total_tokens = sum(
                len(str(r.inputs or {})) // 4 + len(str(r.outputs or {})) // 4 
                for r in successful_runs
            )
            
            return {
                "total_runs": len(runs),
                "successful_runs": len(successful_runs),
                "error_rate": error_rate,
                "success_rate": 1 - error_rate,
                "avg_latency": avg_latency,
                "total_tokens": total_tokens,
                "time_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
            }
            
        except Exception as e:
            print(f"Error getting project stats: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize with your project name
    monitor = ProductionMonitor("my-production-app")
    
    # Example of logging a run
    try:
        # Simulate a successful API call
        start_time = datetime.now()
        model = ChatOpenAI(model="gpt-4")
        question = "What is machine learning?"
        response = model.invoke(question)
        
        # Log the successful run
        run_id = monitor.log_run(
            inputs={"question": question},
            outputs={"response": response.content},
            run_type="llm",
            latency=(datetime.now() - start_time).total_seconds(),
            model="gpt-4",
            temperature=0.7
        )
        
        print(f"Generated response: {response.content}")
        print(f"Run logged with ID: {run_id}")
        
        # Simulate user feedback
        monitor.log_feedback(
            run_id=run_id,
            score=0.9,  # 0-1 scale
            feedback="Accurate and helpful response"
        )
        
        # Get project statistics
        stats = monitor.get_project_stats(days=7)
        print("\nProject Statistics (last 7 days):")
        print(f"Total runs: {stats['total_runs']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Average latency: {stats['avg_latency']:.2f}s")
        print(f"Estimated tokens used: {stats['total_tokens']:,}")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
        raise`;

  return (
    <DocSection
      id="langsmith"
      title="LangSmith - Monitoring & Evaluation"
      description="Trace, monitor, and evaluate your LLM applications with comprehensive observability and testing tools."
      badges={["Monitoring", "Evaluation", "Production"]}
      externalLinks={[
        { title: "LangSmith Docs", url: "https://docs.smith.langchain.com/" },
        { title: "Dashboard", url: "https://smith.langchain.com/" },
        { title: "Python SDK", url: "https://python.langchain.com/docs/langsmith/" },
        { title: "API Reference", url: "https://api.python.langchain.com/en/latest/langsmith/langsmith.html" }
      ]}
    >
      <Tabs defaultValue="quickstart" className="w-full">
        <TabsList className="grid w-full grid-cols-4 mb-6">
          <TabsTrigger value="quickstart">Quick Start</TabsTrigger>
          <TabsTrigger value="features">Features</TabsTrigger>
          <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
        </TabsList>
        
        <TabsContent value="quickstart" className="space-y-8">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-4">
                <div className="p-2 rounded-lg bg-amber-50 dark:bg-amber-900/20">
                  <Zap className="w-6 h-6 text-amber-500" />
                </div>
                <div>
                  <CardTitle>Quick Start with LangSmith</CardTitle>
                  <CardDescription>
                    Get started with LangSmith in minutes. Follow these steps to set up monitoring and evaluation for your LLM applications.
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-8">
              {/* Prerequisites Section */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold flex items-center gap-2 text-blue-600 dark:text-blue-400">
                  <CheckCircle className="w-5 h-5" />
                  Prerequisites
                </h3>
                <ul className="list-disc pl-5 space-y-2 text-muted-foreground">
                  <li>Python 3.8+ or Node.js 16+ installed</li>
                  <li>LangSmith API key (get it from <a href="https://smith.langchain.com/settings" className="text-primary hover:underline">settings</a>)</li>
                  <li>Basic understanding of LangChain or LLM applications</li>
                </ul>
              </div>

              {/* Installation */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <Terminal className="w-5 h-5 text-blue-500" />
                  1. Installation
                </h3>
                <div className="space-y-2">
                  <p className="text-muted-foreground">Install LangSmith using your preferred package manager:</p>
                  <Tabs defaultValue="python" className="w-full">
                    <TabsList className="grid w-full grid-cols-2 max-w-xs mb-4">
                      <TabsTrigger value="python">Python</TabsTrigger>
                      <TabsTrigger value="typescript">TypeScript</TabsTrigger>
                    </TabsList>
                    <TabsContent value="python">
                      <CodeBlock 
                        language="bash"
                        code="pip install -U langsmith"
                        showLineNumbers={false}
                      />
                    </TabsContent>
                    <TabsContent value="typescript">
                      <CodeBlock 
                        language="bash"
                        code="npm install @langchain/langgraph @langchain/langsmith"
                        showLineNumbers={false}
                      />
                    </TabsContent>
                  </Tabs>
                  <p className="text-sm text-muted-foreground">
                    For Jupyter notebook users, you may need to restart the kernel after installation.
                  </p>
                </div>
              </div>
              
              <div className="space-y-4">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <Settings className="w-5 h-5 text-purple-500" />
                  2. Configure Environment
                </h3>
                <div className="space-y-3">
                  <div>
                    <p className="text-muted-foreground mb-2">Set up your environment variables:</p>
                    <Tabs defaultValue="env" className="w-full">
                      <TabsList className="grid w-full grid-cols-2 max-w-xs mb-4">
                        <TabsTrigger value="env">.env File</TabsTrigger>
                        <TabsTrigger value="bash">Terminal</TabsTrigger>
                      </TabsList>
                      <TabsContent value="env">
                        <CodeBlock 
                          language="bash"
                          title=".env"
                          code={"# Required for LangSmith tracing\n" +
                          "LANGCHAIN_TRACING_V2=true\n" +
                          "LANGCHAIN_API_KEY=your_api_key_here\n" +
                          "LANGCHAIN_PROJECT=your_project_name\n" +
                          "# Optional: Enable additional debugging\n" +
                          "LANGCHAIN_VERBOSE=true"}
                          showLineNumbers={true}
                        />
                      </TabsContent>
                      <TabsContent value="bash">
                        <CodeBlock 
                          language="bash"
                          title="Terminal"
                          code={"# Get your API key from https://smith.langchain.com/settings\n" +
                          "export LANGCHAIN_TRACING_V2=true\n" +
                          "export LANGCHAIN_API_KEY=your_api_key_here\n" +
                          "export LANGCHAIN_PROJECT=your_project_name"}
                          showLineNumbers={true}
                        />
                      </TabsContent>
                    </Tabs>
                  </div>
                  
                  <div className="p-4 bg-muted/30 rounded-lg">
                    <h4 className="font-medium flex items-center gap-2 mb-2">
                      <Info className="w-4 h-4 text-blue-500" />
                      Best Practices
                    </h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Use different project names for different environments (e.g., <code className="bg-muted px-1 rounded">myapp-dev</code>, <code className="bg-muted px-1 rounded">myapp-prod</code>)</li>
                      <li>• Never commit API keys to version control - use <code className="bg-muted px-1 rounded">.env</code> files and add them to <code className="bg-muted px-1 rounded">.gitignore</code></li>
                      <li>• Consider using environment management tools like <code className="bg-muted px-1 rounded">direnv</code> or <code className="bg-muted px-1 rounded">dotenv</code></li>
                      <li>• For production, use environment variables set in your deployment platform</li>
                    </ul>
                  </div>
                  
                  <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
                    <h4 className="font-medium flex items-center gap-2 text-amber-700 dark:text-amber-400">
                      <AlertTriangle className="w-4 h-4" />
                      Security Note
                    </h4>
                    <p className="text-sm text-amber-700/80 dark:text-amber-400/80 mt-1">
                      Keep your API keys secure. Never expose them in client-side code or public repositories. 
                      Use environment variables and server-side code for API interactions.
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="space-y-2">
                <h3 className="text-lg font-medium flex items-center gap-2">
                  <Code2 className="w-5 h-5 text-green-500" />
                  3. Start Tracing
                </h3>
                <p className="text-muted-foreground">Add tracing to your LangChain application:</p>
                <CodeBlock 
                  language="python"
                  code={"from langchain_openai import ChatOpenAI\n" +
                  "from langchain.callbacks.tracers import LangChainTracer\n\n" +
                  "# Initialize with tracing\n" +
                  "llm = ChatOpenAI(\n" +
                  "    model=\"gpt-4\",\n" +
                  "    temperature=0.7,\n" +
                  "    callbacks=[LangChainTracer()]  # This enables tracing\n" +
                  ")\n\n" +
                  "# Your existing code continues here\n" +
                  "response = llm.invoke(\"Explain quantum computing in simple terms\")\n" +
                  "print(response.content)"}
                  showLineNumbers={true}
                />
              </div>
              
              <div className="space-y-2">
                <h3 className="text-lg font-medium flex items-center gap-2">
                  <BarChart className="w-5 h-5 text-cyan-500" />
                  4. View Traces
                </h3>
                <p className="text-muted-foreground">
                Visit the{' '}
                <a href="https://smith.langchain.com" className="text-primary hover:underline">
                  LangSmith Dashboard
                </a>{' '}
                to view your traces and monitor your application's performance.
              </p>
              </div>
              
              <div className="space-y-2">
                <h3 className="text-lg font-medium flex items-center gap-2">
                  <AlertOctagon className="w-5 h-5 text-rose-500" />
                  Troubleshooting
                </h3>
                <ul className="list-disc pl-5 space-y-1 text-muted-foreground">
                  <li>Make sure your API key is correctly set in the environment variables</li>
                  <li>Verify that <code className="bg-muted px-1 rounded">LANGCHAIN_TRACING_V2</code> is set to "true"</li>
                  <li>Check that your project name doesn't contain spaces or special characters</li>
                  <li>Ensure you have internet connectivity to reach the LangSmith service</li>
                </ul>
              </div>
            </CardContent>
            <CardFooter>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <BookOpen className="w-4 h-4" />
                <span>For more details, check out the <a href="https://docs.smith.langchain.com/" className="text-primary hover:underline">LangSmith documentation</a>.</span>
              </div>
            </CardFooter>
          </Card>
        </TabsContent>
        
        <TabsContent value="features" className="space-y-8">
          {/* Key Features */}
          <div className="space-y-4">
            <h2 className="text-2xl font-semibold">Key Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <FeatureCard
                icon={<Eye className="w-6 h-6" />}
                title="Request Tracing"
                description="Comprehensive tracing of LLM requests, chains, and agent interactions."
                features={[
                  "End-to-end visibility",
                  "Performance metrics",
                  "Error tracking",
                  "Input/output logging"
                ]}
              />
            <FeatureCard
              icon={<BarChart3 className="w-6 h-6" />}
              title="Evaluation Framework"
              description="Systematic evaluation of LLM applications with custom metrics and datasets."
              features={[
                "Custom evaluators",
                "Benchmark datasets",
                "A/B testing",
                "Regression detection"
              ]}
            />
            <FeatureCard
              icon={<Settings className="w-6 h-6" />}
              title="Production Monitoring"
              description="Real-time monitoring of production applications with alerts and dashboards."
              features={[
                "Real-time dashboards",
                "Performance alerts",
                "Cost tracking",
                "Usage analytics"
              ]}
            />
            <FeatureCard
              icon={<Target className="w-6 h-6" />}
              title="Dataset Management"
              description="Create, manage, and version datasets for testing and evaluation."
              features={[
                "Dataset versioning",
                "Example management",
                "Import/export",
                "Collaborative editing"
              ]}
            />
            <FeatureCard
              icon={<Zap className="w-6 h-6" />}
              title="Optimization Insights"
              description="Identify bottlenecks and optimization opportunities in your applications."
              features={[
                "Performance profiling",
                "Cost optimization",
                "Quality improvements",
                "Usage patterns"
              ]}
            />
            <FeatureCard
              icon={<AlertTriangle className="w-6 h-6" />}
              title="Debugging Tools"
              description="Advanced debugging capabilities for complex LLM applications."
              features={[
                "Step-by-step debugging",
                "Variable inspection",
                "Error analysis",
                "Replay functionality"
              ]}
            />
            </div>
          </div>

          {/* Code Examples */}
          <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Implementation Guide</h2>
          <Tabs defaultValue="setup" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="setup">Setup & Tracing</TabsTrigger>
              <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
              <TabsTrigger value="monitoring">Production Monitoring</TabsTrigger>
            </TabsList>
            
            <TabsContent value="setup" className="space-y-4">
              <CodeBlock
                title="LangSmith Setup and Basic Tracing"
                language="python"
                code={setupCode}
              />
            </TabsContent>
            
            <TabsContent value="evaluation" className="space-y-4">
              <CodeBlock
                title="Application Evaluation"
                language="python"
                code={evaluationCode}
              />
            </TabsContent>
            
            <TabsContent value="monitoring" className="space-y-4">
              <CodeBlock
                title="Production Monitoring"
                language="python"
                code={monitoringCode}
              />
            </TabsContent>
          </Tabs>
        </div>

        {/* Evaluation Strategies */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Evaluation Strategies</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Built-in Evaluators</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium">QA Evaluator</h4>
                    <p className="text-sm text-muted-foreground">Evaluates question-answering accuracy</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Criteria Evaluator</h4>
                    <p className="text-sm text-muted-foreground">Custom criteria-based evaluation</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Similarity Evaluator</h4>
                    <p className="text-sm text-muted-foreground">Semantic similarity scoring</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Embedding Distance</h4>
                    <p className="text-sm text-muted-foreground">Vector-based similarity metrics</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-card">
              <CardHeader>
                <CardTitle>Custom Metrics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div>
                    <h4 className="font-medium">Domain-Specific</h4>
                    <p className="text-sm text-muted-foreground">Custom business logic evaluation</p>
                  </div>
                  <div>
                    <h4 className="font-medium">LLM-as-Judge</h4>
                    <p className="text-sm text-muted-foreground">Use LLMs to evaluate responses</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Human Feedback</h4>
                    <p className="text-sm text-muted-foreground">Integrate human evaluation</p>
                  </div>
                  <div>
                    <h4 className="font-medium">Multi-Metric</h4>
                    <p className="text-sm text-muted-foreground">Combine multiple evaluation approaches</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Best Practices */}
        <Card className="shadow-card border-l-4 border-l-primary">
          <CardHeader>
            <CardTitle>LangSmith Best Practices</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h4 className="font-medium">Development Workflow</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Create separate projects for dev/staging/prod</li>
                  <li>• Use descriptive run names and metadata</li>
                  <li>• Tag experiments for easy filtering</li>
                  <li>• Regularly review and analyze traces</li>
                </ul>
              </div>
              <div className="space-y-3">
                <h4 className="font-medium">Production Monitoring</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Set up alerts for error rates and latency</li>
                  <li>• Monitor token usage and costs</li>
                  <li>• Collect user feedback systematically</li>
                  <li>• Use sampling for high-volume applications</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
    </DocSection>
  );
};

export default LangSmithSection;