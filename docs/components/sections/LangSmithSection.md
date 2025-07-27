# LangSmithSection

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
    
      
        
          
Quick Start

          
Features

          
Evaluation

          
Monitoring

          
Next Steps

        
        
        
          
            
              
                
                  

                
                  
Quick Start with LangSmith

                  
Get started with LangSmith in minutes. Follow these steps to set up monitoring and evaluation for your LLM applications.

                

            
            
              {/* Prerequisites Section */}
              
                
                  
Prerequisites

                
                  
Python 3.8+ or Node.js 16+ installed

                  LangSmith API key (get it from 
settings
)
                  
Basic understanding of LangChain or LLM applications

                

              {/* Installation */}
              
                
                  
1. Installation

                
                  
Install LangSmith using your preferred package manager:

                  
                    
                      
Python

                      
TypeScript

                    
                    
                      

                    
                      

                  
                  
For Jupyter notebook users, you may need to restart the kernel after installation.

                

              
              
                
                  
2. Configure Environment

                
                  
                    
Set up your environment variables:

                    
                      
                        
.env File

                        
Terminal

                      
                      
                        

                      
                        

                    

                  
                  
                    
                      
Best Practices

                    
                      • Use different project names for different environments (e.g., 
myapp-dev
, 
myapp-prod
)
                      • Never commit API keys to version control - use 
.env
 files and add them to 
.gitignore

                      • Consider using environment management tools like 
direnv
 or 
dotenv

                      
• For production, use environment variables set in your deployment platform

                    

                  
                  
                    
                      
Security Note

                    
Keep your API keys secure. Never expose them in client-side code or public repositories. 
                      Use environment variables and server-side code for API interactions.

                  

              
              
              
                
                  
3. Start Tracing

                
                  
                    
Add tracing to your application:

                    
                      
                        
Python

                        
TypeScript

                      
                      
                      
                        
                          
Basic Example

                          

                        
                        
                          
With LangChain Expression Language (LCEL)

                          

                      
                      
                      
                        
                          
Basic Example

                          

                        
                        
                          
With LangChain Expression Language (LCEL)

                          

                      

                  
                  
                  
                    
                      
Best Practices for Tracing

                    
                      Wrap your main application logic in a 
with tracing_enabled()
 context manager for automatic tracing of all operations
                      
Use unique and descriptive names for your chains and components to make traces easier to identify

                      
Add metadata to your traces for better filtering and organization in the LangSmith UI

                      
Consider using environment variables to control tracing in different environments (development/staging/production)

                      
For long-running applications, periodically flush traces to ensure they're sent to LangSmith

                    

                

              
              
                
                  
4. View and Analyze Traces

                
                  
                    
                      Once your application is running with tracing enabled, visit the{' '}
                      
LangSmith Dashboard
{' '}
                      to view your traces.
                    
                    
                    
                      
                        
                          
Key Dashboard Features

                        
                          
Traces:
 View detailed execution traces of your LLM calls
                          
Latency:
 Monitor performance metrics and identify bottlenecks
                          
Token Usage:
 Track token consumption and costs
                          
Error Tracking:
 Quickly identify and debug failed runs
                          
Filtering:
 Filter traces by time, tags, metadata, and more

                      
                      
                      
                        
                          
Tips for Effective Analysis

                        
                          Use 
metadata
 to add custom tags and filters to your traces
                          
Compare different model versions using the comparison view

                          
Set up alerts for latency spikes or error rates

                          
Export traces for offline analysis or reporting

                          
Use the search functionality to find specific traces by content or metadata

                        

                    

                  
                  
                    
                      
Pro Tip: Debugging with Traces

                    
When debugging, look for traces with high latency or errors. Expand each step to see:

                    
                      
Input and output of each LLM call

                      
Token usage and cost information

                      
Execution time for each component

                      
Any error messages or exceptions

                    

                

              
              
                
                  
5. Troubleshooting & Common Issues

                
                  
                    
                      
                        
Common Issues

                      
                        
                          
No traces appearing?

                          
                            Verify 
LANGCHAIN_TRACING_V2=true
 is set
                            
Check your API key is correctly configured

                            
Ensure your code is using the correct callback handler

                          

                        
                          
Authentication errors?

                          
                            
Verify your API key is valid and has proper permissions

                            
Check for any typos in the API key

                            
Ensure you're using the correct environment

                          

                        
                          
High latency?

                          
                            
Check your network connection

                            
Verify the LangSmith service status

                            
Review your batch processing settings

                          

                      

                    
                    
                      
                        
Debugging Tips

                      
                        Enable verbose logging with 
LANGCHAIN_VERBOSE=true

                        
Check browser's developer console for network errors

                        
Verify your project name doesn't contain special characters

                        
Test with a minimal example to isolate the issue

                      

                  
                  
                  
                    
                      
Diagnostic Commands

                    
                      
                            
Test your LangSmith connection:

                            

                          
                            
Check environment variables:

                            

                    

                  
                  
                    
                      
Need More Help?

                    
If you're still experiencing issues, please:

                    
                      Check the 
official documentation

                      Search the 
GitHub discussions

                      Open an issue on 
GitHub
 with details about your problem

                  

              

            
              
                
                For more details, check out the 
LangSmith documentation
.

            

        
        
        
          {/* Key Features */}
          
            
Key Features

            
              }
                title="Request Tracing"
                description="Comprehensive tracing of LLM requests, chains, and agent interactions."
                features={[
                  "End-to-end visibility",
                  "Performance metrics",
                  "Error tracking",
                  "Input/output logging"
                ]}
              />
            }
              title="Evaluation Framework"
              description="Systematic evaluation of LLM applications with custom metrics and datasets."
              features={[
                "Custom evaluators",
                "Benchmark datasets",
                "A/B testing",
                "Regression detection"
              ]}
            />
            }
              title="Production Monitoring"
              description="Real-time monitoring of production applications with alerts and dashboards."
              features={[
                "Real-time dashboards",
                "Performance alerts",
                "Cost tracking",
                "Usage analytics"
              ]}
            />
            }
              title="Dataset Management"
              description="Create, manage, and version datasets for testing and evaluation."
              features={[
                "Dataset versioning",
                "Example management",
                "Import/export",
                "Collaborative editing"
              ]}
            />
            }
              title="Optimization Insights"
              description="Identify bottlenecks and optimization opportunities in your applications."
              features={[
                "Performance profiling",
                "Cost optimization",
                "Quality improvements",
                "Usage patterns"
              ]}
            />
            
}
              title="Debugging Tools"
              description="Advanced debugging capabilities for complex LLM applications."
              features={[
                "Step-by-step debugging",
                "Variable inspection",
                "Error analysis",
                "Replay functionality"
              ]}
            />

          

          {/* Code Examples */}
          
          
Implementation Guide

          
            
              
Setup & Tracing

              
Evaluation

              
Production Monitoring

            
            
            
              

            
            
              

            
            
              

          

        {/* Evaluation Strategies */}
        
          
Evaluation Strategies

          
            
              
                
Built-in Evaluators

              
              
                
                  
                    
QA Evaluator

                    
Evaluates question-answering accuracy

                  
                  
                    
Criteria Evaluator

                    
Custom criteria-based evaluation

                  
                  
                    
Similarity Evaluator

                    
Semantic similarity scoring

                  
                  
                    
Embedding Distance

                    
Vector-based similarity metrics

                  

              

            
              
                
Custom Metrics

              
              
                
                  
                    
Domain-Specific

                    
Custom business logic evaluation

                  
                  
                    
LLM-as-Judge

                    
Use LLMs to evaluate responses

                  
                  
                    
Human Feedback

                    
Integrate human evaluation

                  
                  
                    
Multi-Metric

                    
Combine multiple evaluation approaches

                  

              

          

        {/* Best Practices */}
        
          
            
LangSmith Best Practices

          
          
            
              
                
Development Workflow

                
                  
• Create separate projects for dev/staging/prod

                  
• Use descriptive run names and metadata

                  
• Tag experiments for easy filtering

                  
• Regularly review and analyze traces

                

              
                
Production Monitoring

                
                  
• Set up alerts for error rates and latency

                  
• Monitor token usage and costs

                  
• Collect user feedback systematically

                  
• Use sampling for high-volume applications

                

            

        

      
      
        
          
            
              
Next Steps with LangSmith

          
          
            
              
What We've Covered

              
                
Setting up and configuring LangSmith for your projects

                
Implementing tracing for LLM applications

                
Viewing and analyzing traces in the dashboard

                
Advanced usage patterns and best practices

                
Integrating with other LangChain components

              

            
            
              
Continue Learning

              
                
                  
                    
Explore the Documentation

                    
Dive deeper into LangSmith's features and capabilities

                  
                  
                    
                      
                        
LangSmith Docs

                    

                
                
                
                  
                    
Join the Community

                    
Get help and share your experiences with other developers

                  
                  
                    
                      
                        
                          

                        GitHub Discussions
                      

                  

              

            
            
              
Next Steps

              
                
                  
                    
Try It Yourself

                  
                    
Set up a new project and enable tracing

                    
Implement custom evaluators for your specific use case

                    
Set up monitoring alerts for your production applications

                  

                
                
                  
                    
Need Help?

                  
If you have questions or run into issues, don't hesitate to reach out:

                  
                    Open an issue on 
GitHub

                    Ask a question in the 
discussions

                    Check the 
documentation
 for more examples

                

            

        

    

  );
};

LangSmithSection;