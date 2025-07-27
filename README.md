# LangChain Ecosystem Documentation

This repository contains concise, markdown-based documentation for the LangChain ecosystem, including LangChain, LangSmith, LangGraph, and LangServe.

## 📚 Documentation

### Core Components

1. **[LangChain](langchain.md)** - Framework for developing applications with language models
   - Models, Prompts, Memory, Chains
   - Common NLP tasks and workflows

2. **[LangSmith](langsmith.md)** - Platform for monitoring and improving LLM applications
   - Debugging and analysis
   - Evaluation and monitoring

3. **[LangGraph](langgraph.md)** - Library for stateful, multi-actor applications
   - Complex workflows
   - Multi-agent systems

4. **[LangServe](langserve.md)** - Deploy LangChain chains as REST APIs
   - Automatic API generation
   - Built-in documentation
## 🚀 Quick Start

### LangChain Example

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)
template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)

print(llm(prompt.format(product="colorful socks")))
```

## 📖 Documentation Structure

- `/docs` - Detailed documentation for each component
  - `langchain.md` - Core LangChain documentation
  - `langsmith.md` - Monitoring and evaluation
  - `langgraph.md` - Stateful workflows
  - `langserve.md` - API deployment

## ✨ Key Features

- **Unified Interface**: Consistent API across LLM providers
- **Modular Design**: Mix and match components as needed
- **Production Ready**: Built-in support for scaling and monitoring
- **Extensible**: Easy to add custom components and integrations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ by Muhammad Mazhar Saeed aka Professor
</div>
