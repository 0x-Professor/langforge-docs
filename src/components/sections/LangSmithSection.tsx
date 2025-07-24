import { DocSection, FeatureCard, QuickStart } from '@/components/DocSection';
import { CodeBlock } from '@/components/CodeBlock';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Settings, BarChart3, Eye, AlertTriangle, Zap, Target } from 'lucide-react';

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

  const evaluationCode = `from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI

client = Client()

# Define your application
def my_chatbot(inputs):
    model = ChatOpenAI(model="gpt-4")
    return {"output": model.invoke(inputs["question"]).content}

# Create evaluation dataset
dataset = client.create_dataset(
    "chatbot-eval",
    description="Evaluation dataset for chatbot"
)

# Add examples to dataset
examples = [
    {"question": "What is Python?", "expected": "Python is a programming language"},
    {"question": "How to use loops?", "expected": "Use for/while loops"},
]

for example in examples:
    client.create_example(
        inputs={"question": example["question"]},
        outputs={"expected": example["expected"]},
        dataset_id=dataset.id
    )

# Define evaluators
def correctness_evaluator(run, example):
    """Evaluate if the response is correct."""
    prediction = run.outputs["output"]
    expected = example.outputs["expected"]
    
    # Simple similarity check (in practice, use better metrics)
    score = 1.0 if expected.lower() in prediction.lower() else 0.0
    
    return {"key": "correctness", "score": score}

def relevance_evaluator(run, example):
    """Evaluate response relevance using LLM."""
    model = ChatOpenAI(model="gpt-4")
    
    prompt = f"""
    Question: {example.inputs['question']}
    Response: {run.outputs['output']}
    
    Rate the relevance of the response to the question on a scale of 1-5.
    Return only the number.
    """
    
    score = float(model.invoke(prompt).content.strip())
    return {"key": "relevance", "score": score / 5.0}

# Run evaluation
results = evaluate(
    my_chatbot,
    data=dataset,
    evaluators=[correctness_evaluator, relevance_evaluator],
    experiment_prefix="chatbot-eval-v1"
)

print(f"Evaluation results: {results}")`;

  const monitoringCode = `from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
import time

client = Client()

class ProductionMonitor:
    def __init__(self, project_name: str):
        self.client = Client()
        self.project_name = project_name
        
    def log_run(self, inputs, outputs, run_type="llm", **metadata):
        """Log a production run."""
        run = self.client.create_run(
            name=f"production-{run_type}",
            project_name=self.project_name,
            run_type=run_type,
            inputs=inputs,
            outputs=outputs,
            extra=metadata
        )
        return run.id
    
    def log_feedback(self, run_id: str, score: float, feedback: str = ""):
        """Log user feedback for a run."""
        self.client.create_feedback(
            run_id=run_id,
            key="user_satisfaction",
            score=score,
            comment=feedback
        )
    
    def get_project_stats(self, days: int = 7):
        """Get project statistics."""
        runs = list(self.client.list_runs(
            project_name=self.project_name,
            start_time=time.time() - (days * 24 * 60 * 60)
        ))
        
        return {
            "total_runs": len(runs),
            "error_rate": len([r for r in runs if r.error]) / len(runs) if runs else 0,
            "avg_latency": sum(r.latency for r in runs if r.latency) / len(runs) if runs else 0
        }

# Usage in production
monitor = ProductionMonitor("my-production-app")

# Log application runs
def production_chatbot(question: str) -> str:
    start_time = time.time()
    
    try:
        model = ChatOpenAI(model="gpt-4")
        response = model.invoke(question)
        
        # Log successful run
        run_id = monitor.log_run(
            inputs={"question": question},
            outputs={"response": response.content},
            latency=time.time() - start_time,
            model="gpt-4"
        )
        
        return response.content, run_id
        
    except Exception as e:
        # Log failed run
        run_id = monitor.log_run(
            inputs={"question": question},
            outputs={"error": str(e)},
            latency=time.time() - start_time,
            error=True
        )
        raise e

# Example usage with feedback
response, run_id = production_chatbot("What is machine learning?")
print(response)

# Later, log user feedback
monitor.log_feedback(run_id, score=0.8, feedback="Good response")

# Get project statistics
stats = monitor.get_project_stats()
print(f"Project stats: {stats}")`;

  return (
    <DocSection
      id="langsmith"
      title="LangSmith - Monitoring & Evaluation"
      description="Trace, monitor, and evaluate your LLM applications with comprehensive observability and testing tools."
      badges={["Monitoring", "Evaluation", "Production"]}
      externalLinks={[
        { title: "LangSmith Docs", url: "https://docs.smith.langchain.com/" },
        { title: "Dashboard", url: "https://smith.langchain.com/" },
        { title: "Python SDK", url: "https://python.langchain.com/docs/langsmith/" }
      ]}
    >
      <div className="space-y-8">
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
      </div>
    </DocSection>
  );
};