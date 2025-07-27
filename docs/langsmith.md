# LangSmith Documentation

## Introduction
LangSmith is a platform for monitoring, evaluating, and improving LLM applications in production. It helps you:
- Debug and analyze LLM applications
- Evaluate model performance
- Monitor production deployments
- Improve prompts and chains

## Key Features

### 1. Tracing
- Visualize LLM calls and chains
- Track token usage and costs
- Debug complex workflows

### 2. Evaluation
- Automated evaluation metrics
- Human-in-the-loop feedback
- Custom evaluators

### 3. Monitoring
- Real-time performance metrics
- Alerting for anomalies
- Production insights

## Quick Start

```python
import os
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.chat_models import ChatOpenAI

# Set your API key
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# Initialize your model
llm = ChatOpenAI(temperature=0)

# Define your evaluation config
eval_config = RunEvalConfig(
    evaluators=[
        "qa",
        "criteria": {
            "helpfulness": "How helpful is the response?",
            "harmlessness": "How harmless is the response?",
        }
    ]
)

# Run evaluation
results = run_on_dataset(
    dataset_name="your-dataset",
    llm_or_chain_factory=llm,
    evaluation=eval_config,
)
```

## Common Use Cases
- Debugging complex LLM chains
- Evaluating model performance
- Monitoring production deployments
- Collecting user feedback
