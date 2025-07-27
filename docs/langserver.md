# LangServer: The Complete Guide

<div class="tip">
  <strong>🚀 New in v0.1.0</strong>: Enhanced language server protocol support, improved performance, and new code intelligence features.
</div>

## Introduction

LangServer is a language server implementation that provides rich language features for LLM development environments. It implements the [Language Server Protocol (LSP)](https://microsoft.github.io/language-server-protocol/) to provide features like code completion, diagnostics, and documentation on hover across various editors and IDEs.

### Key Features

- **Code Intelligence**: Smart code completion, signature help, and hover documentation
- **Real-time Feedback**: Instant diagnostics and error checking
- **Cross-Editor Support**: Works with VS Code, Vim, Emacs, and more
- **Extensible**: Easy to customize and extend for specific needs
- **Performance Optimized**: Built for speed with large codebases

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for some editor integrations)
- Your preferred code editor with LSP support

### Installation

```bash
# Install the LangServer package
pip install langserver

# Or install with optional dependencies
pip install 'langserver[all]'
```

### Basic Configuration

Create a `pyproject.toml` file in your project root:

```toml
[tool.langserver]
# Enable/disable specific features
enable_completion = true
enable_hover = true
enable_diagnostics = true

# Configure Python path if needed
python_path = "./venv/bin/python"

# Additional import paths
import_paths = ["src"]

# Configure logging
log_level = "info"
log_file = "langserver.log"
```

### Starting the Server

#### Command Line

```bash
# Start the language server
langserver start

# Start with custom configuration
langserver start --config pyproject.toml

# Start in development mode with debug logging
LANG_SERVER_DEBUG=1 langserver start --log-level=debug
```

#### Programmatic Usage

```python
from langserver import start_server

# Start the server with custom configuration
start_server(
    host="127.0.0.1",
    port=2087,
    log_level="info",
    config_path="pyproject.toml"
)
```

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
