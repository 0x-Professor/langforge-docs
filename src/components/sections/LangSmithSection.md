# LangSmith Section Component

[![Documentation](https://img.shields.io/badge/Documentation-100%25-brightgreen)](https://docs.smith.langchain.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.9.5-blue.svg)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-18.2.0-61DAFB.svg)](https://reactjs.org/)
[![LangSmith](https://img.shields.io/badge/LangSmith-0.1.0-FF6B35.svg)](https://smith.langchain.com/)

## Overview

The `LangSmithSection` is a comprehensive React component that provides documentation and interactive examples for LangSmith, a platform for monitoring, evaluating, and improving LLM applications in production. This component is part of the LangForge documentation system.

## Features

- **Interactive Code Examples**: Live Python code snippets demonstrating LangSmith functionality
- **Comprehensive Documentation**: Detailed guides for setup, monitoring, and evaluation
- **Responsive Design**: Works on desktop and mobile devices
- **Tabbed Interface**: Organized content in easy-to-navigate tabs
- **Syntax Highlighting**: Code examples with syntax highlighting

## Installation

This component is part of the LangForge documentation system. To use it in your project:

```bash
# Clone the repository
git clone https://github.com/your-org/langforge-docs.git
cd langforge-docs

# Install dependencies
npm install
```

## Usage

Import and use the component in your React application:

```tsx
import { LangSmithSection } from '@/components/sections/LangSmithSection';

function App() {
  return (
    <div className="container mx-auto p-4">
      <LangSmithSection />
    </div>
  );
}
```

## Component Structure

```
LangSmithSection/
├── Quick Start Tab
│   ├── Installation
│   ├── Environment Setup
│   ├── Basic Tracing
│   └── Troubleshooting
├── Features Tab
│   ├── Request Tracing
│   ├── Evaluation Framework
│   ├── Production Monitoring
│   ├── Dataset Management
│   ├── Optimization Insights
│   └── Debugging Tools
├── Evaluation Tab
│   ├── Built-in Evaluators
│   ├── Custom Metrics
│   └── Evaluation Strategies
└── Monitoring Tab
    ├── Real-time Dashboards
    ├── Alert Configuration
    └── Performance Metrics
```

## Code Examples

### Basic Setup

```python
import os
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain.callbacks.tracers import LangChainTracer

# Configure environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "your-project-name"

# Initialize client and model
client = Client()
model = ChatOpenAI(
    model="gpt-4",
    callbacks=[LangChainTracer()]
)
```

## Props

This component accepts the following props:

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| className | string | '' | Additional CSS classes for styling |
| ...rest | HTMLAttributes | {} | Additional HTML attributes |

## Styling

The component uses Tailwind CSS for styling. You can override styles by adding custom classes through the `className` prop or by extending the theme in your Tailwind configuration.

## Dependencies

- React 18.2.0+
- TypeScript 4.9.5+
- Lucide React (for icons)
- @radix-ui/react-tabs
- @radix-ui/react-slot
- class-variance-authority
- clsx
- tailwind-merge

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Radix UI](https://www.radix-ui.com/)
- [Lucide Icons](https://lucide.dev/)

## Support

For support, please open an issue in the repository or contact the maintainers.
