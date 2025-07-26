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
   git clone https://github.com/0x-Professor/langforge-docs.git
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

This project follows a standard React application structure, enhanced with specific directories for documentation content and UI components. Below is a detailed breakdown of the key directories and files:

```
langforge-docs/
├── public/                 # Static assets like images, fonts, and the favicon.
├── src/                    # Contains all the source code for the React application.
│   ├── components/         # Reusable React components used throughout the documentation.
│   │   ├── sections/       # Individual documentation sections (e.g., Introduction, LangChain).
│   │   └── ui/             # Shadcn/ui components, customized for the project's design system.
│   ├── hooks/              # Custom React hooks for shared logic and state management.
│   ├── lib/                # Utility functions and helper modules.
│   ├── pages/              # Top-level page components that define routes and layouts.
│   ├── App.tsx             # The main application component, responsible for routing and global layout.
│   └── main.tsx            # The entry point of the React application, where the app is rendered.
├── .eslintrc.cjs           # ESLint configuration for code linting and style enforcement.
├── index.html              # The main HTML file, serving as the entry point for the web application.
├── package.json            # Defines project metadata, dependencies, and npm scripts.
├── postcss.config.js       # PostCSS configuration for processing CSS with plugins like Autoprefixer.
├── tailwind.config.ts      # Tailwind CSS configuration for defining design tokens and utility classes.
└── tsconfig.json           # TypeScript configuration for compiling TypeScript code.
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



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - For building the amazing LLM framework
- [shadcn/ui](https://ui.shadcn.com/) - For the beautiful UI components
- [Vite](https://vitejs.dev/) - For the fast development experience
- [Tailwind CSS](https://tailwindcss.com/) - For the utility-first CSS framework

---

<div align="center">
  Made with ❤️ by Muhammad Mazhar Saeed aka Professor
</div>


## 🚀 Deployment

To deploy your own version, ensure you have built the project:

```bash
npm run build
```

Then, deploy the `dist` directory to your preferred static site hosting service.




## 🤝 Contributing

We welcome contributions to the LangForge Documentation! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## 📄 Code of Conduct

This project adheres to the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.



## 🐛 Troubleshooting

This section provides solutions to common issues you might encounter while setting up or developing with LangForge Documentation.

### 1. `npm install` fails with ERESOLVE errors

This usually indicates a conflict in peer dependencies, especially with `react` versions. The project is configured to use React 18, but some dependencies might require React 19 or vice-versa.

**Solution:**
Run `npm install` with the `--force` flag to bypass peer dependency conflicts. This will force npm to install the packages even if there are conflicts. While generally not recommended for production, it's often necessary for development environments with complex dependency trees.

```bash
npm install --force
```

### 2. `npm run build` shows warnings about large chunks

When building the project, you might see warnings like "Some chunks are larger than 500 kB after minification." This means that some of your JavaScript bundles are quite large, which can impact loading performance.

**Solution:**
These are warnings, not errors, and the build will still complete. For production deployments, consider implementing code splitting using dynamic `import()` statements for larger components or libraries. You can also adjust the `build.chunkSizeWarningLimit` in `vite.config.ts` to increase the threshold, though this only hides the warning and doesn't solve the underlying performance issue.

### 3. Development server (`npm run dev`) not starting

If the development server fails to start, check the following:

- **Port in use**: Another process might be using port 5173 (the default Vite port). You can try killing the process or configuring Vite to use a different port.
- **Dependency issues**: Ensure all dependencies are correctly installed. Run `npm install --force` again.
- **Syntax errors**: Check your recent code changes for any syntax errors that might prevent the application from compiling.

### 4. Images or assets not loading in production build

If your deployed application is missing images or other static assets, it might be due to incorrect paths.

**Solution:**
Ensure that all asset paths in your code are relative to the `public` directory or are correctly handled by Vite. For static deployments, Vite usually handles asset paths correctly, but issues can arise with custom configurations or when moving files manually. Verify the `base` option in `vite.config.ts` if you are deploying to a subpath.

### 5. Navigation links not working or pages not rendering

If clicking on navigation links doesn't load the correct content or pages appear blank, it could be a routing issue.

**Solution:**
Verify that your routing configuration in `src/App.tsx` and `src/pages/Index.tsx` is correct. Ensure that the paths defined in your router match the components you intend to render. Also, check for any JavaScript errors in the browser console that might indicate issues with component rendering or data fetching.

If you encounter an issue not listed here, please refer to the [Contributing Guidelines](CONTRIBUTING.md) for how to report a bug or seek assistance.

