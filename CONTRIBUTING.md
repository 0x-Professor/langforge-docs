# Contributing to LangForge Documentation

Thank you for your interest in contributing to LangForge Documentation! This guide will help you get started and ensure your contributions align with our standards.

## 🎯 Ways to Contribute

### 📝 **Documentation Improvements**
- Fix typos, grammar, or formatting issues
- Add missing examples or clarify existing ones
- Improve navigation and organization
- Add new tutorials or guides
- Translate content to other languages

### 💻 **Code Contributions**
- Fix bugs in the documentation platform
- Add new features (search, analytics, etc.)
- Improve performance and accessibility
- Enhance the build system

### 🐛 **Issue Reporting**
- Report bugs or broken links
- Suggest new features or improvements
- Request new examples or tutorials

## 🚀 Getting Started

### Prerequisites
- Node.js 16+ and npm 8+
- Git
- Text editor (VS Code recommended)
- Basic knowledge of Markdown

### Setup Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/langforge-docs.git
   cd langforge-docs
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Open in browser**
   ```
   http://localhost:3000
   ```

## 📁 Project Structure

```
langforge-docs/
├── docs/                    # Documentation content
│   ├── getting-started/     # Beginner guides
│   ├── examples/           # Code examples
│   ├── guides/             # How-to guides
│   └── components/         # Component docs
├── server.js               # Express server
├── tests/                  # Test files
├── .github/               # GitHub workflows
└── scripts/               # Build scripts
```

## ✍️ Writing Guidelines

### Documentation Style
- **Clear and concise**: Use simple, direct language
- **Action-oriented**: Start with verbs ("Create", "Build", "Configure")
- **Code-heavy**: Include working examples for everything
- **Visual**: Use diagrams, tables, and formatting effectively

### Markdown Standards
- Use `#` for main headings, `##` for sections, `###` for subsections
- Include code blocks with proper language highlighting
- Add alt text for images: `![Alt text](image.png)`
- Use tables for structured data
- Include emoji for visual appeal (sparingly)

### Code Examples
- **Working code**: All examples must be tested and functional
- **Complete examples**: Include all necessary imports and setup
- **Comments**: Explain complex parts
- **Multiple languages**: Provide Python and TypeScript when possible

Example format:
```python
# Install: pip install langchain openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Initialize the LLM
llm = OpenAI(temperature=0.7, api_key="your-key")

# Create a prompt template
template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms"
)

# Generate response
response = llm(template.format(topic="machine learning"))
print(response)
```

## 🔄 Contribution Workflow

### 1. **Create an Issue** (for significant changes)
- Describe the problem or improvement
- Discuss approach with maintainers
- Get approval before starting work

### 2. **Create a Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

### 3. **Make Your Changes**
- Follow our style guidelines
- Add tests for new functionality
- Update documentation as needed

### 4. **Test Your Changes**
```bash
npm run lint          # Check code style
npm test              # Run tests
npm run build         # Test build process
```

### 5. **Commit Your Changes**
Use conventional commit format:
```bash
git commit -m "docs: add RAG system example"
git commit -m "fix: resolve broken link in quickstart"
git commit -m "feat: add search functionality"
```

### 6. **Submit a Pull Request**
- Fill out the PR template completely
- Link to related issues
- Add screenshots for UI changes
- Request review from maintainers

## 📋 Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows style guidelines (ESLint passes)
- [ ] All tests pass (`npm test`)
- [ ] Documentation is updated
- [ ] Examples are tested and working
- [ ] No breaking changes (or documented)
- [ ] PR title follows conventional commits
- [ ] PR description explains the changes

## 🧪 Testing

### Running Tests
```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run specific test file
npm test -- tests/api.test.js
```

### Writing Tests
- Test all new functionality
- Include edge cases
- Use descriptive test names
- Follow existing test patterns

## 📖 Documentation Types

### **Getting Started**
- Installation guides
- Quick start tutorials
- Basic concepts

### **Examples**
- Working code samples
- Step-by-step tutorials
- Real-world use cases

### **Guides**
- How-to articles
- Best practices
- Troubleshooting

### **Reference**
- API documentation
- Configuration options
- Advanced topics

## 🌍 Translation Guidelines

We welcome translations! Here's how to contribute:

1. **Check existing translations** in the `/docs/translations/` folder
2. **Create language folder**: `/docs/translations/es/` (for Spanish)
3. **Translate core pages first**: README, getting-started, basic examples
4. **Maintain structure**: Keep the same folder structure as English docs
5. **Update navigation**: Add language links to navigation

## 🎨 Style Guide

### Tone and Voice
- **Friendly and approachable**: Write like you're helping a colleague
- **Professional**: Maintain technical accuracy
- **Encouraging**: Help users feel confident
- **Inclusive**: Use inclusive language and examples

### Formatting
- **Headers**: Use sentence case (not title case)
- **Code**: Always use syntax highlighting
- **Links**: Use descriptive link text
- **Lists**: Use parallel structure
- **Emphasis**: Use **bold** for important terms, *italics* for emphasis

## 🔧 Development Guidelines

### Code Style
- Follow ESLint configuration
- Use Prettier for formatting
- Write self-documenting code
- Include JSDoc comments for functions

### Performance
- Optimize images and assets
- Minimize bundle size
- Use lazy loading where appropriate
- Test on mobile devices

### Accessibility
- Use semantic HTML
- Include alt text for images
- Ensure keyboard navigation works
- Test with screen readers

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation changes
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested

## 🎉 Recognition

Contributors are recognized in several ways:

- **Contributors list**: Added to README.md
- **Release notes**: Mentioned in release announcements
- **Hall of fame**: Featured on documentation site
- **Swag**: LangForge stickers and swag for significant contributions

## 💬 Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/0x-Professor/langforge-docs/discussions)
- **Bug reports**: Create an [issue](https://github.com/0x-Professor/langforge-docs/issues)
- **Feature requests**: Use our [feature request template](https://github.com/0x-Professor/langforge-docs/issues/new?template=feature_request.md)
- **Real-time chat**: Join our Discord community

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make LangForge Documentation better for everyone! 🚀

