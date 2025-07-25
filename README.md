# LangForge Documentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5.3-blue.svg)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-18.3.1-61DAFB.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4.1-646CFF.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4.1-06B6D4.svg)](https://tailwindcss.com/)
[![shadcn/ui](https://img.shields.io/badge/shadcn_ui-0.0.1-000000.svg)](https://ui.shadcn.com/)

## 🚀 Overview

LangForge is a comprehensive documentation platform for LangChain and its ecosystem, providing detailed guides, API references, and interactive examples for building with large language models (LLMs).

## ✨ Features

- **Interactive Documentation**: Live code examples and interactive components
- **Comprehensive Guides**: In-depth tutorials and how-to guides
- **API Reference**: Detailed documentation for all LangChain components
- **Modern Tech Stack**: Built with React 18, TypeScript, and Vite
- **Beautiful UI**: Styled with Tailwind CSS and shadcn/ui components
- **Responsive Design**: Works on desktop and mobile devices

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/langforge-docs.git
   cd langforge-docs
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Build for production**
   ```bash
   npm run build
   npm run preview
   ```

## 🏗️ Project Structure

```
langforge-docs/
├── public/                 # Static files
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── sections/       # Documentation sections
│   │   └── ui/             # shadcn/ui components
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Utility functions
│   ├── pages/              # Page components
│   ├── App.tsx             # Main application component
│   └── main.tsx            # Application entry point
├── .eslintrc.cjs           # ESLint configuration
├── index.html              # HTML template
├── package.json            # Project dependencies
├── postcss.config.js       # PostCSS configuration
├── tailwind.config.js      # Tailwind CSS configuration
└── tsconfig.json           # TypeScript configuration
```

## 📚 Documentation Sections

1. **Introduction**
   - Getting Started
   - Core Concepts
   - Installation Guide

2. **LangChain**
   - Core Components
   - Modules
   - Integrations

3. **LangGraph**
   - Basics
   - Agents
   - Streaming

4. **LangSmith**
   - Tracing
   - Evaluation
   - Monitoring

5. **LangServe**
   - Setup
   - APIs
   - Deployment

6. **MCP (Model Control Protocol)**
   - Introduction
   - Server
   - Client
   - SDKs

7. **Agent Architecture**
   - Multi-Agent Systems
   - Agent Communication
   - Advanced Patterns

## 🛠️ Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Adding New Content

1. Create a new component in `src/components/sections/`
2. Add the component to the router in `src/pages/Index.tsx`
3. Update the navigation in `src/components/Navigation.tsx`

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - For building the amazing LLM framework
- [shadcn/ui](https://ui.shadcn.com/) - For the beautiful UI components
- [Vite](https://vitejs.dev/) - For the fast development experience
- [Tailwind CSS](https://tailwindcss.com/) - For the utility-first CSS framework

---

<div align="center">
  Made with ❤️ by the Muhammad Mazhar Saeed aka Professor
</div>
