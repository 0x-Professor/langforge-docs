# LangServer: The Complete Guide

<div class="tip">
  <strong>🚀 New in v0.2.0</strong>: Enhanced LSP support, multi-root workspaces, semantic tokens, and advanced code actions.
</div>

## Introduction

LangServer is a high-performance, extensible language server that brings rich language features to your development environment. Built on the [Language Server Protocol (LSP)](https://microsoft.github.io/language-server-protocol/), it provides intelligent code assistance across multiple programming languages with a focus on Python and LLM development workflows.

### Why LangServer?

- **Unified Development Experience**: Consistent features across all supported editors
- **Lightning Fast**: Optimized for large codebases with thousands of files
- **AI-Powered**: Integrates with LLMs for advanced code understanding
- **Extensible**: Built with customization in mind

### Key Features

#### Core Language Features
- 🎯 **Intelligent Code Completion**
- 📝 **Signature Help** with parameter information
- 📚 **Hover Documentation** with rich content support
- 🔍 **Go to Definition** & **Find References**
- 🏷 **Semantic Highlighting**
- 📝 **Code Actions** & **Quick Fixes**
- 🛠 **Refactoring** support

#### Advanced Capabilities
- 🧠 **AI-Assisted Code Understanding**
- 🧩 **Plugin System** for custom extensions
- ⚡ **Incremental Sync** for large files
- 🔄 **Multi-root Workspace** support
- 🌐 **Remote Development** ready

## Getting Started

### System Requirements

- **Python**: 3.8+ (3.10+ recommended)
- **Node.js**: 16+ (for VS Code extension and web features)
- **RAM**: Minimum 4GB (8GB+ recommended for large projects)
- **Disk Space**: 500MB free space

### Installation Options

#### Basic Installation
```bash
# Core package
pip install langserver

# With all optional dependencies
pip install 'langserver[all]'  # Includes testing, docs, and dev tools

# For AI features (requires API key)
pip install 'langserver[ai]'
```

#### Development Installation
```bash
git clone https://github.com/langserver/langserver.git
cd langserver
pip install -e '.[dev]'  # Editable install with dev dependencies
pre-commit install  # Install git hooks
```

### Configuration

LangServer can be configured through multiple methods (in order of priority):

1. **Editor Settings** (highest priority)
2. **Command-line Arguments**
3. **Configuration Files**
   - `pyproject.toml` (recommended)
   - `.langserver.toml`
   - `setup.cfg`
4. **Environment Variables**
5. **Default Values** (lowest priority)

#### Basic Configuration

Create a `pyproject.toml` in your project root with the following structure:

```toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[tool.langserver]
# Server Configuration
host = "127.0.0.1"  # Bind address
port = 2087  # Default port
log_level = "info"  # debug, info, warning, error
log_file = ".langserver/langserver.log"  # Relative to workspace root

# Feature Toggles
enable_completion = true
enable_hover = true
enable_diagnostics = true
enable_formatting = true
enable_rename = true
enable_references = true
enable_workspace_symbols = true

# Python-specific Settings
[tool.langserver.python]
python_path = ".venv/bin/python"  # Path to Python interpreter
import_strategy = "useBundled"  # or "fromEnvironment"
venv_path = ".venv"

# Path Configuration
[tool.langserver.paths]
source_roots = ["src"]  # Source directories to index
exclude_patterns = ["**/__pycache__", "**/.git", "**/node_modules"]

# Completion Settings
[tool.langserver.completion]
enable_snippets = true
resolve_eagerly = false
imports_from_environment = true

# Diagnostic Settings
[tool.langserver.diagnostics]
enable = true
run = "onSave"  # or "onType", "off"
max_number = 100

# Formatting Settings
[tool.langserver.formatting]
provider = "black"  # or "autopep8", "yapf", "none"
line_length = 88

# AI Features (optional)
[tool.langserver.ai]
enabled = true
model = "gpt-4"  # or "gpt-3.5-turbo", "claude-2"
api_key = "${OPENAI_API_KEY}"  # Read from environment

# Logging Configuration
[tool.langserver.logging]
level = "info"
path = ".langserver/logs"
max_size = "10MB"
backup_count = 3

# Performance Settings
[tool.langserver.performance]
max_workers = 4  # Number of worker processes
max_memory = "2GB"  # Memory limit
jedi_settings = { extra_paths = ["src"], environment = ".venv" }

# Extension Settings
[tool.langserver.extensions]
# Enable/disable specific extensions
pylsp = true
pyright = false
jedi = true
```

### Configuration Options Explained

#### Core Server Settings
- `host`/`port`: Network interface and port to bind to
- `log_level`: Verbosity of logs (debug, info, warning, error)
- `log_file`: Path to log file (relative to workspace root)

#### Python-specific Settings
- `python_path`: Path to Python interpreter
- `import_strategy`: How to handle imports ("useBundled" or "fromEnvironment")
- `venv_path`: Path to virtual environment

#### Path Configuration
- `source_roots`: Directories containing source code to index
- `exclude_patterns`: Glob patterns to exclude from indexing

#### Completion Settings
- `enable_snippets`: Whether to enable snippet completions
- `resolve_eagerly`: Resolve completion items immediately
- `imports_from_environment`: Include symbols from installed packages

#### Diagnostic Settings
- `enable`: Enable/disable diagnostics
- `run`: When to run diagnostics ("onSave", "onType", "off")
- `max_number`: Maximum number of diagnostics to report

#### Formatting Settings
- `provider`: Code formatter to use ("black", "autopep8", "yapf", "none")
- `line_length`: Maximum line length for formatting

### Environment Variables

Configuration can also be set via environment variables:

```bash
# Core Settings
export LANG_SERVER_HOST="127.0.0.1"
export LANG_SERVER_PORT=2087
export LANG_SERVER_LOG_LEVEL="info"

# Python Environment
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# AI Features
export OPENAI_API_KEY="sk-..."

export LANG_SERVER_CONFIG="pyproject.toml"
```

### Editor-specific Configuration

#### VS Code
```json
{
    "langserver.python.pythonPath": ".venv/bin/python",
    "langserver.python.analysis.extraPaths": ["src"],
    "langserver.formatting.provider": "black"
}
```

#### Neovim (with nvim-lspconfig)
```lua
require('lspconfig').langserver.setup {
    cmd = { 'langserver', 'start' },
    settings = {
        langserver = {
            python = {
                pythonPath = '.venv/bin/python',
                analysis = {
                    extraPaths = { 'src' },
                    typeCheckingMode = 'basic'
                }
            }
        }
    }
}
```

## Server Management

### Starting the Server

#### Command Line Interface

```bash
# Basic usage
langserver start

# With custom configuration
langserver start --config pyproject.toml

# Development mode with debug logging
LANG_SERVER_DEBUG=1 langserver start --log-level=debug

# Custom host and port
langserver start --host 0.0.0.0 --port 3000

# Profile server performance
langserver start --profile --profile-output profile.json

# Check server status
langserver status

# Stop a running server
langserver stop

# View server logs
tail -f .langserver/langserver.log
```

#### Running as a Daemon

```bash
# Install systemd service (Linux)
sudo langserver install-service --user

# Start the service
systemctl --user start langserver

# Enable on boot
systemctl --user enable langserver

# View logs
journalctl --user -u langserver -f
```

#### Programmatic Usage

```python
from pathlib import Path
from langserver import LanguageServer, ServerConfig

# Create a configuration
config = ServerConfig(
    host="127.0.0.1",
    port=2087,
    log_level="info",
    config_path=Path("pyproject.toml"),
    enable_completion=True,
    enable_hover=True,
    enable_diagnostics=True,
)

# Create and start the server
server = LanguageServer("my-langserver", "1.0.0", config=config)

# Register custom features
@server.feature("textDocument/didOpen")
async def on_did_open(ls, params):
    """Handle document open event."""
    doc = ls.workspace.get_document(params.text_document.uri)
    ls.logger.info(f"Document opened: {doc.uri}")

# Start the server
if __name__ == "__main__":
    server.start()
```

## Editor Integration

### VS Code

1. Install the LangServer extension from the VS Code marketplace
2. Add this to your VS Code settings (`settings.json`):

```json
{
    "langserver.enable": true,
    "langserver.path": "langserver",
    "langserver.trace.server": "verbose",
    "langserver.config": "${workspaceFolder}/pyproject.toml",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true,
        "source.fixAll": true
    },
    "[python]": {
        "editor.defaultFormatter": "langserver.vscode-langserver",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

### Neovim (0.5+)

Using built-in LSP client:

```lua
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

-- Check if the config is already defined
if not configs.langserver then
  configs.langserver = {
    default_config = {
      cmd = {'langserver', 'start'};
      filetypes = {'python', 'javascript', 'typescript', 'go', 'rust'};
      root_dir = function(fname)
        return lspconfig.util.root_pattern('pyproject.toml', '.git')(fname) or vim.loop.os_homedir()
      end;
      settings = {
        langserver = {
          python = {
            analysis = {
              typeCheckingMode = "basic",
              autoSearchPaths = true,
              useLibraryCodeForTypes = true
            }
          }
        }
      };
    };
  }
end

-- Setup language server
lspconfig.langserver.setup{
  on_attach = function(client, bufnr)
    -- Mappings
    local bufopts = { noremap=true, silent=true, buffer=bufnr }
    vim.keymap.set('n', 'gD', vim.lsp.buf.declaration, bufopts)
    vim.keymap.set('n', 'gd', vim.lsp.buf.definition, bufopts)
    vim.keymap.set('n', 'K', vim.lsp.buf.hover, bufopts)
    vim.keymap.set('n', 'gi', vim.lsp.buf.implementation, bufopts)
    vim.keymap.set('n', '<C-k>', vim.lsp.buf.signature_help, bufopts)
    vim.keymap.set('n', '<space>wa', vim.lsp.buf.add_workspace_folder, bufopts)
    vim.keymap.set('n', '<space>wr', vim.lsp.buf.remove_workspace_folder, bufopts)
    vim.keymap.set('n', '<space>wl', function()
      print(vim.inspect(vim.lsp.buf.list_workspace_folders()))
    end, bufopts)
    vim.keymap.set('n', '<space>D', vim.lsp.buf.type_definition, bufopts)
    vim.keymap.set('n', '<space>rn', vim.lsp.buf.rename, bufopts)
    vim.keymap.set('n', '<space>ca', vim.lsp.buf.code_action, bufopts)
    vim.keymap.set('n', 'gr', vim.lsp.buf.references, bufopts)
    vim.keymap.set('n', '<space>f', function() vim.lsp.buf.format { async = true } end, bufopts)
  end
}
```

### Emacs (lsp-mode)

```elisp
(use-package lsp-mode
  :ensure t
  :hook ((python-mode . lsp)
         (js-mode . lsp)
         (typescript-mode . lsp)
         (go-mode . lsp)
         (rust-mode . lsp)
         (lsp-mode . lsp-enable-which-key-integration))
  :commands lsp
  :custom
  (lsp-lens-enable t)
  (lsp-headerline-breadcrumb-enable t)
  (lsp-modeline-diagnostics-enable t)
  (lsp-prefer-flymake nil)  ; Use lsp-ui and flycheck
  (lsp-enable-file-watchers t)
  (lsp-enable-symbol-highlighting t)
  (lsp-enable-indentation t)
  (lsp-enable-on-type-formatting t)
  (lsp-enable-snippet t)
  (lsp-enable-folding t)
  (lsp-enable-imenu t)
  (lsp-enable-dap-auto-configure t)
  (lsp-keep-workspace-alive nil)
  (lsp-log-io nil)  ; Only for debugging
  (lsp-auto-guess-root t)  ; Automatically find project root
  (lsp-file-watch-threshold 2000)  ; Increase if needed for large projects

  ;; LangServer specific settings
  (lsp-langserver-python-command "langserver")
  (lsp-langserver-python-args '("start"))
  (lsp-langserver-configuration-sections '("python" "jedi" "pylsp"))
  (lsp-langserver-configuration "python.analysis.typeCheckingMode" "basic")
  (lsp-langserver-configuration "python.analysis.autoImportCompletions" t)
  (lsp-langserver-configuration "python.analysis.autoSearchPaths" t)
  (lsp-langserver-configuration "python.analysis.useLibraryCodeForTypes" t))

;; Optional: Better UI
(use-package lsp-ui
  :ensure t
  :after lsp-mode
  :commands lsp-ui-mode
  :custom
  (lsp-ui-doc-enable t)
  (lsp-ui-doc-header t)
  (lsp-ui-doc-include-signature t)
  (lsp-ui-doc-position 'at-point)
  (lsp-ui-doc-max-width 100)
  (lsp-ui-doc-max-height 30)
  (lsp-ui-sideline-enable t)
  (lsp-ui-sideline-show-hover t)
  (lsp-ui-sideline-show-code-actions t)
  (lsp-ui-sideline-show-diagnostics t)
  (lsp-ui-sideline-ignore-duplicate t))

;; Optional: Company mode for completions
(use-package company-lsp
  :ensure t
  :after (lsp-mode company)
  :commands company-lsp
  :config
  (push 'company-lsp company-backends))

;; Optional: Debugging support
(use-package dap-mode
  :ensure t
  :after lsp-mode
  :config
  (dap-mode t)
  (dap-ui-mode t))

## Core Concepts

### 1. Language Server Protocol (LSP)

LangServer implements the Language Server Protocol, which provides a standardized way for tools to communicate with language servers. Key features include:

- **Text Document Synchronization**: Keeps the server's document model in sync with the editor
- **Language Features**: Completion, hover, signature help, etc.
- **Workspace Management**: Handles workspace folders and configurations

### 2. Sessions and Connections

- Each editor instance creates a new session
- Sessions maintain their own document state
- Connections handle the JSON-RPC communication

### 3. Document Management

- Documents are tracked by URI
- Supports incremental updates
- Maintains a symbol table for quick lookups

## Configuration

### Server Configuration

Configuration can be provided through multiple sources (in order of priority):

1. Editor configuration (settings.json, etc.)
2. Command-line arguments
3. Configuration file (pyproject.toml, .langserver.toml, etc.)
4. Environment variables
5. Default values

### Common Configuration Options

```toml
[tool.langserver]
# Server settings
host = "127.0.0.1"
port = 2087
log_level = "info"
log_file = "langserver.log"

# Feature toggles
enable_completion = true
enable_hover = true
enable_diagnostics = true
enable_formatting = true

# Performance options
workspace_folders = ["."]
max_workers = 4
jedi_settings = { extra_paths = ["src"] }

# Language-specific settings
[python]
python_path = "./venv/bin/python"
auto_import_completion = true
include_docstrings = true

[typescript]
check_js = true
prefer_types = true
```

## Editor Integration

### VS Code

1. Install the LangServer extension from the VS Code marketplace
2. Add this to your VS Code settings:

```json
{
    "langserver.enable": true,
    "langserver.path": "langserver",
    "langserver.trace.server": "verbose",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### Vim/Neovim

Using [coc.nvim](https://github.com/neoclide/coc.nvim):

```vim
" Install the extension
:CocInstall coc-langserver

" Add to your init.vim
let g:coc_global_extensions = ['coc-langserver']

" Key mappings
nmap <silent> gd <Plug>(coc-definition)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> gr <Plug>(coc-references)
nmap <silent> [g <Plug>(coc-diagnostic-prev)
nmap <silent> ]g <Plug>(coc-diagnostic-next)
```

### Emacs

Using [lsp-mode](https://emacs-lsp.github.io/lsp-mode/):

```elisp
(use-package lsp-mode
  :ensure t
  :hook ((python-mode . lsp)
         (typescript-mode . lsp)
         (go-mode . lsp))
  :commands lsp)

(use-package lsp-ui :ensure t)
(use-package company-lsp :ensure t)

;; Configure for Python
(with-eval-after-load 'lsp-mode
  (require 'lsp-python-ms)
  (setq lsp-python-ms-python-executable "python3")
  (add-to-list 'lsp-enabled-clients 'mspyls))
```

## Advanced Usage

### Custom Commands

Define custom commands in your configuration:

```toml
[tool.langserver.commands]
format_document = { command = "black", args = ["--quiet", "-"] }
lint = { command = "pylint", args = ["--output-format=text"] }
```

### Custom Hover Content

```python
from langserver import LanguageServer
from lsprotocol.types import Hover, Position, TextDocumentIdentifier

server = LanguageServer("my-server", "v0.1")

@server.feature("textDocument/hover")
def hover(params: TextDocumentIdentifier, position: Position) -> Hover:
    """Return custom hover content."""
    doc = server.workspace.get_document(params.text_document.uri)
    word = doc.word_at_position(position)
    
    if word == "TODO":
        return Hover(
            contents="This is a TODO item. Consider implementing this feature.",
            range=doc.word_range_at_position(position)
        )
    
    return None
```

### Custom Diagnostics

```python
from langserver import LanguageServer
from lsprotocol.types import (
    Diagnostic, DiagnosticSeverity, Position, Range,
    PublishDiagnosticsParams, TextDocumentIdentifier
)

server = LanguageServer("my-server", "v0.1")

@server.feature("textDocument/didSave")
def validate_document(ls, params: TextDocumentIdentifier):
    """Validate document on save."""
    doc = server.workspace.get_document(params.text_document.uri)
    diagnostics = []
    
    # Simple example: Check for TODOs
    for line_num, line in enumerate(doc.lines, 1):
        if "TODO" in line:
            diagnostic = Diagnostic(
                range=Range(
                    start=Position(line=line_num-1, character=line.index("TODO")),
                    end=Position(line=line_num-1, character=line.index("TODO") + 4)
                ),
                message="TODO found",
                severity=DiagnosticSeverity.Information,
                source="langserver"
            )
            diagnostics.append(diagnostic)
    
    # Publish diagnostics
    ls.publish_diagnostics(
        PublishDiagnosticsParams(
            uri=params.text_document.uri,
            diagnostics=diagnostics
        )
    )
```

## Best Practices

### Performance Optimization

1. **Use Workspace Folders**: Properly configure workspace folders to limit file scanning
2. **Enable Caching**: Use the built-in caching mechanism for better performance
3. **Limit Concurrent Operations**: Adjust `max_workers` based on your system
4. **Use .gitignore**: Add patterns to ignore unnecessary files

### Error Handling

- Implement proper error handling in custom commands
- Use the logging system for debugging
- Handle missing dependencies gracefully

### Security

- Run the server with minimal permissions
- Validate all user input
- Use secure connections when possible

## Troubleshooting

### Common Issues

1. **Server Not Starting**
   - Check if the port is already in use
   - Verify Python version compatibility
   - Check log files for errors

2. **No Completions**
   - Ensure the language server is properly initialized
   - Check that the file type is supported
   - Verify the Python environment has all required packages

3. **Performance Issues**
   - Check for large files or directories being indexed
   - Review log files for slow operations
   - Consider increasing memory limits if needed

### Debugging

Enable debug logging:

```bash
LANG_SERVER_DEBUG=1 langserver start --log-level=debug
```

Check log files in the configured log directory or standard error output.

## API Reference

### Core Classes

- `LanguageServer`: Main server class
- `Workspace`: Manages workspace state
- `Document`: Represents a text document
- `Session`: Manages a client connection

### Extension Points

- `@server.feature()`: Register a feature handler
- `@server.command()`: Register a custom command
- `@server.thread()`: Run a handler in a separate thread

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

LangServer is licensed under the [MIT License](LICENSE).

## Getting Help

- [Documentation](https://langserver.readthedocs.io)
- [GitHub Issues](https://github.com/langserver/langserver/issues)
- [Community Chat](https://gitter.im/langserver/community)

---

*This documentation is a work in progress. Please contribute to improve it!*
