# LangSmith: The Complete Guide

<div class="tip">
  <strong>🚀 New in v0.1.0</strong>: Enhanced tracing, custom evaluators, and production monitoring features.
</div>

## Introduction

LangSmith is a powerful platform for developing, monitoring, and improving LLM applications in production. It provides the tools you need to build reliable, high-performing LLM applications with confidence.

### Key Benefits

- **Debugging**: Trace and visualize complex LLM calls and chains
- **Evaluation**: Measure and improve model performance with custom metrics
- **Monitoring**: Track production performance and get alerts for issues
- **Collaboration**: Share and compare results across your team
- **Optimization**: Identify and fix performance bottlenecks

## Quick Start

### 1. Installation

```bash
pip install langsmith
```

### 2. Set Up Your Environment

```python
import os
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Set your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["LANGCHAIN_API_KEY"] = "your-langchain-api-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize your model
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
```

### 3. Create a Simple Chain

```python
# Define a prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the following question: {question}"
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Test the chain
response = chain.run(question="What is LangSmith?")
print(response)
```

### 4. Set Up Tracing

```python
# Enable tracing (already set in environment variables above)
# All subsequent chain runs will be traced automatically

# Run your chain with tracing
with tracing_enabled():
    result = chain.run(question="How does LangSmith help with LLM development?")
    print(f"Trace URL: {tracing.get_trace_url()}")
```

### 5. Create a Test Dataset

```python
from langsmith import Client

client = Client()

dataset_name = "example-qa-dataset"
try:
    dataset = client.read_dataset(dataset_name=dataset_name)
except:
    # Create a new dataset if it doesn't exist
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Example QA dataset for testing"
    )
    
    # Add examples
    examples = [
        ({"question": "What is LangSmith?"}, {"answer": "LangSmith is a platform for developing and monitoring LLM applications."}),
        ({"question": "How does tracing work?"}, {"answer": "Tracing captures the execution of LLM calls and chains for debugging and analysis."}),
    ]
    
    client.create_examples(
        inputs=[e[0] for e in examples],
        outputs=[e[1] for e in examples],
        dataset_id=dataset.id
    )
```

### 6. Run Evaluation

```python
# Define evaluation criteria
eval_config = RunEvalConfig(
    evaluators=[
        "qa",  # Built-in QA evaluator
        {
            "criteria": {
                "helpfulness": "How helpful is the response?",
                "relevance": "How relevant is the response to the question?",
                "conciseness": "How concise is the response?",
            }
        },
        # Custom evaluator function
        {
            "custom_evaluator": {
                "name": "custom_eval",
                "evaluation_function": lambda input, output, reference: {
                    "custom_score": 0.95,
                    "reasoning": "The response is well-structured and informative."
                }
            }
        }
    ],
    # Optional: Add metadata for analysis
    eval_llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
)

# Run evaluation
results = run_on_dataset(
    dataset_name=dataset_name,
    llm_or_chain_factory=lambda: chain,  # Your chain
    evaluation=eval_config,
    verbose=True,
    project_name="my-first-eval"
)

print(f"Evaluation complete. Results: {results}")
```

## Core Concepts

### 1. Tracing

Tracing allows you to visualize and debug the execution of your LLM applications.

#### Basic Tracing

```python
from langsmith import traceable
from langchain.callbacks.manager import tracing_v2_enabled

@traceable
def process_query(question: str) -> str:
    """Process a user question and return a response."""
    # Your LLM chain or processing logic here
    return chain.run(question=question)

# Enable tracing for this block
with tracing_v2_enabled(project_name="my-llm-app"):
    result = process_query("What is LangSmith?")
    print(f"Trace URL: {tracing.get_trace_url()}")
```

#### Nested Tracing

```python
@traceable
def retrieve_context(question: str) -> str:
    """Retrieve relevant context for a question."""
    # Simulate retrieval
    return "LangSmith is a platform for developing and monitoring LLM applications."

@traceable
def generate_response(question: str, context: str) -> str:
    """Generate a response using the provided context."""
    prompt = f"""Answer the question based on the following context:
    
    {context}
    
    Question: {question}"""
    return llm.predict(prompt)

@traceable
def answer_question(question: str) -> str:
    """End-to-end question answering."""
    context = retrieve_context(question)
    return generate_response(question, context)

# All nested calls will be traced
with tracing_v2_enabled(project_name="nested-tracing"):
    response = answer_question("What is LangSmith?")
    print(response)
```

### 2. Evaluation

LangSmith provides powerful tools for evaluating your LLM applications.

#### Custom Evaluators

```python
from typing import Dict, Any
from langchain.evaluation import load_evaluator

def custom_evaluator(run, example) -> Dict[str, Any]:
    """Custom evaluator that checks response length and content."""
    prediction = run.outputs["output"]
    
    # Initialize evaluators
    fact_evaluator = load_evaluator("criteria", criteria="factuality")
    
    # Run evaluations
    fact_result = fact_evaluator.evaluate_strings(
        prediction=prediction,
        input=example.inputs["question"]
    )
    
    # Calculate custom metrics
    word_count = len(prediction.split())
    
    return {
        "fact_score": fact_result["score"],
        "word_count": word_count,
        "is_too_short": word_count < 5,
        "feedback": fact_result["reasoning"]
    }

# Use the custom evaluator
eval_config = RunEvalConfig(
    custom_evaluators=[custom_evaluator],
    eval_llm=ChatOpenAI(temperature=0, model="gpt-4")
)
```

#### Human Feedback

```python
from langsmith import Client
from langchain.callbacks import get_openai_callback

client = Client()

# Record human feedback
def record_feedback(run_id: str, score: int, comment: str = ""):
    client.create_feedback(
        run_id,
        key="human_rating",
        score=score,  # 1-5 scale
        comment=comment,
    )

# Example usage
with get_openai_callback() as cb:
    result = chain.run(question="What is LangSmith?")
    print(f"Generated response: {result}")
    
    # Get the trace URL for human review
    trace_url = tracing.get_trace_url()
    print(f"Review at: {trace_url}")
    
    # Simulate human feedback (in a real app, this would come from a UI)
    record_feedback(
        run_id=tracing.get_current_run_id(),
        score=4,
        comment="Good response, but could be more detailed."
    )
```

### 3. Monitoring

Monitor your LLM applications in production with real-time metrics and alerts.

#### Setting Up Monitoring

```python
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

# Define metrics to track
metrics = [
    "latency",
    "token_usage",
    "feedback.human_rating",
    "evaluation.fact_score"
]

# Get metrics for the last 24 hours
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=1)

metrics_data = client.read_metrics(
    project_name="my-llm-app",
    metrics=metrics,
    start_time=start_time,
    end_time=end_time,
    group_by=["model", "prompt_version"]
)

# Analyze metrics
print(f"Average latency: {metrics_data['latency'].mean()}s")
print(f"Total tokens used: {metrics_data['token_usage'].sum()}")
```

#### Setting Up Alerts

```python
# Create an alert for high latency
alert_config = {
    "name": "High Latency Alert",
    "description": "Alert when average latency exceeds threshold",
    "metric": "latency",
    "condition": ">",
    "threshold": 5.0,  # seconds
    "window": "1h",    # 1-hour rolling window
    "notification_channels": ["email:your-email@example.com"],
    "severity": "high"
}

client.create_alert(
    project_name="my-llm-app",
    **alert_config
)
```

## Real-World Use Cases

### 1. Customer Support Chatbot

```python
from typing import List, Dict, Any
from langchain.schema import SystemMessage, HumanMessage

class SupportBot:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
        self.context = []
        
    @traceable
    def generate_response(self, user_input: str) -> str:
        """Generate a response to a user's support request."""
        # Add to conversation history
        self.context.append(HumanMessage(content=user_input))
        
        # Create prompt with context
        messages = [
            SystemMessage(content="You are a helpful customer support agent."),
            *self.context[-6:]  # Last 3 exchanges (user + assistant)
        ]
        
        # Generate response
        response = self.llm(messages)
        
        # Add to context
        self.context.append(response)
        
        return response.content

# Initialize bot
bot = SupportBot()

# Example conversation
with tracing_v2_enabled(project_name="support-bot"):
    print(bot.generate_response("I can't log into my account."))
    print(bot.generate_response("I've tried resetting my password but it's not working."))
```

### 2. Content Moderation Pipeline

```python
from enum import Enum
from pydantic import BaseModel

class ModerationResult(BaseModel):
    is_safe: bool
    reason: str
    confidence: float
    flagged_categories: List[str]
    explanation: str

class ContentModerator:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        
    @traceable
    def moderate_content(self, text: str) -> ModerationResult:
        """Check if content violates moderation policies."""
        prompt = f"""Analyze the following content for policy violations:
        
        {text}
        
        Check for:
        - Hate speech or discrimination
        - Harassment or bullying
        - Violence or harmful content
        - Sexual content
        - Personal information
        - Spam or scams
        
        Return a JSON object with:
        - is_safe (boolean)
        - reason (string)
        - confidence (float 0-1)
        - flagged_categories (list of strings)
        - explanation (string)"""
        
        response = self.llm.predict(prompt)
        return ModerationResult.parse_raw(response)

# Example usage
moderator = ContentModerator()

with tracing_v2_enabled(project_name="content-moderation"):
    result = moderator.moderate_content("This is a test message with no issues.")
    print(f"Is safe: {result.is_safe}")
    print(f"Reason: {result.reason}")
```

## Best Practices

### 1. Effective Tracing

- **Use meaningful names**: Give your traces and spans descriptive names
- **Add metadata**: Include relevant context in your traces
- **Handle errors**: Use try/except blocks and log errors appropriately
- **Use spans**: Group related operations together

```python
from langsmith import trace_span

def process_document(document: str) -> dict:
    """Process a document through multiple steps."""
    with trace_span("document_processing") as span:
        # Add metadata to the span
        span.metadata.update({
            "document_length": len(document),
            "processing_start_time": datetime.utcnow().isoformat()
        })
        
        try:
            # Step 1: Extract text
            with trace_span("text_extraction"):
                text = extract_text(document)
                
            # Step 2: Analyze sentiment
            with trace_span("sentiment_analysis"):
                sentiment = analyze_sentiment(text)
                
            # Step 3: Generate summary
            with trace_span("summarization"):
                summary = generate_summary(text)
                
            return {
                "text": text,
                "sentiment": sentiment,
                "summary": summary,
                "status": "success"
            }
            
        except Exception as e:
            # Log the error
            span.metadata["error"] = str(e)
            raise
```

### 2. Effective Evaluation

- **Define clear criteria**: Be specific about what makes a good response
- **Use multiple evaluators**: Combine automated and human evaluation
- **Test edge cases**: Include challenging examples in your test sets
- **Iterate**: Use evaluation results to improve your prompts and models

### 3. Production Monitoring

- **Set up alerts**: Get notified of issues in real-time
- **Track key metrics**: Monitor latency, token usage, and quality scores
- **A/B test**: Compare different model versions or prompts
- **Retain data**: Keep enough history to identify trends and patterns

## Troubleshooting

### Common Issues

1. **Missing Traces**
   - Verify `LANGCHAIN_TRACING_V2` is set to "true"
   - Check your API key has the correct permissions
   - Ensure your code is running in a traced context

2. **Evaluation Errors**
   - Check that your dataset format matches expected input
   - Verify evaluator requirements are met
   - Ensure your API keys have the necessary permissions

3. **Performance Issues**
   - Check for rate limiting
   - Optimize your prompts to reduce token usage
   - Consider batching requests when possible

## API Reference

### Core Functions

- `tracing_v2_enabled()`: Context manager for enabling tracing
- `traceable`: Decorator for tracing functions
- `RunEvalConfig`: Configuration for evaluation runs
- `Client`: Main client for interacting with the LangSmith API

### Client Methods

- `create_dataset()`: Create a new dataset
- `create_examples()`: Add examples to a dataset
- `run_on_dataset()`: Run evaluation on a dataset
- `read_metrics()`: Read metrics for a project
- `create_alert()`: Create a monitoring alert

## Next Steps

1. **Explore the UI**: Visit [LangSmith Dashboard](https://smith.langchain.com) to view your traces and metrics
2. **Join the Community**: Get help and share your experiences in the [LangChain Community](https://community.langchain.com/)
3. **Read the Docs**: Check out the [official documentation](https://docs.langchain.com/langsmith/) for more details
4. **Try Examples**: Experiment with the example notebooks in the [LangSmith Examples](https://github.com/langchain-ai/langsmith-examples) repository
