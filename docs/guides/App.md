# Application Architecture for LangChain Projects

This guide covers how to structure and organize LangChain applications for scalability, maintainability, and production deployment.

## Overview

Building robust LangChain applications requires careful consideration of architecture patterns, dependency management, and routing strategies. This guide covers:

- Application structure and organization
- Dependency injection and state management
- Routing and navigation patterns
- Error handling and monitoring

## Application Structure Patterns

### Modular Application Architecture

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import logging

class LangChainApplication:
    """Main application class for LangChain projects"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.services = {}
        self.routes = {}
        
        # Initialize core components
        self._initialize_services()
        self._setup_routes()
        self._configure_error_handling()
    
    def _initialize_services(self):
        """Initialize core application services"""
        
        # LLM Service
        self.services['llm'] = OpenAI(
            temperature=self.config.get('temperature', 0.7),
            api_key=self.config.get('openai_api_key')
        )
        
        # Memory Service
        self.services['memory'] = ConversationBufferMemory()
        
        # Conversation Chain
        self.services['conversation'] = ConversationChain(
            llm=self.services['llm'],
            memory=self.services['memory']
        )
        
        self.logger.info("Core services initialized successfully")
    
    def _setup_routes(self):
        """Setup application routes and handlers"""
        
        self.routes = {
            '/chat': self.handle_chat,
            '/documents': self.handle_documents,
            '/search': self.handle_search,
            '/health': self.handle_health_check
        }
        
        self.logger.info(f"Configured {len(self.routes)} application routes")
    
    def _configure_error_handling(self):
        """Configure global error handling"""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up error handlers
        self.error_handlers = {
            'not_found': self._handle_not_found,
            'server_error': self._handle_server_error,
            'rate_limit': self._handle_rate_limit
        }
    
    def handle_request(self, path: str, **kwargs) -> Dict[str, Any]:
        """Handle incoming requests"""
        
        try:
            if path in self.routes:
                return self.routes[path](**kwargs)
            else:
                return self.error_handlers['not_found'](path)
                
        except Exception as e:
            self.logger.error(f"Request failed: {str(e)}")
            return self.error_handlers['server_error'](e)
    
    def handle_chat(self, message: str, **kwargs) -> Dict[str, Any]:
        """Handle chat interactions"""
        
        try:
            response = self.services['conversation'].predict(input=message)
            
            return {
                "success": True,
                "response": response,
                "metadata": {
                    "model": "openai",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Chat handling failed: {str(e)}")
            return {
                "success": False,
                "error": "Unable to process chat message",
                "details": str(e)
            }
    
    def handle_documents(self, action: str, **kwargs) -> Dict[str, Any]:
        """Handle document operations"""
        
        if action == "upload":
            return self._handle_document_upload(**kwargs)
        elif action == "search":
            return self._handle_document_search(**kwargs)
        elif action == "summarize":
            return self._handle_document_summarize(**kwargs)
        else:
            return {"error": f"Unknown document action: {action}"}
    
    def handle_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Handle search requests"""
        
        try:
            # Implement search logic here
            results = self._perform_search(query)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return {
                "success": False,
                "error": "Search functionality unavailable",
                "details": str(e)
            }
    
    def handle_health_check(self, **kwargs) -> Dict[str, Any]:
        """Health check endpoint"""
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {}
        }
        
        # Check service health
        for service_name, service in self.services.items():
            try:
                # Simple health check - could be more sophisticated
                if hasattr(service, 'health_check'):
                    health_status["services"][service_name] = service.health_check()
                else:
                    health_status["services"][service_name] = "operational"
            except Exception as e:
                health_status["services"][service_name] = f"error: {str(e)}"
                health_status["status"] = "degraded"
        
        return health_status
    
    def _handle_not_found(self, path: str) -> Dict[str, Any]:
        """Handle 404 Not Found errors"""
        
        return {
            "error": "Not Found",
            "message": f"The requested path '{path}' was not found",
            "available_paths": list(self.routes.keys()),
            "status_code": 404
        }
    
    def _handle_server_error(self, error: Exception) -> Dict[str, Any]:
        """Handle 500 Server Error"""
        
        return {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing your request",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_rate_limit(self, **kwargs) -> Dict[str, Any]:
        """Handle rate limiting errors"""
        
        return {
            "error": "Rate Limit Exceeded",
            "message": "Too many requests. Please try again later.",
            "status_code": 429,
            "retry_after": 60
        }
```

### Dependency Management

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ServiceProvider(Protocol):
    """Protocol for service providers"""
    
    def get_service(self, service_name: str) -> Any:
        """Get a service by name"""
        ...
    
    def register_service(self, service_name: str, service: Any) -> None:
        """Register a service"""
        ...

class DependencyContainer:
    """Simple dependency injection container"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
    
    def register_service(self, name: str, service: Any) -> None:
        """Register a service instance"""
        self._services[name] = service
    
    def register_factory(self, name: str, factory: callable) -> None:
        """Register a service factory"""
        self._factories[name] = factory
    
    def get_service(self, name: str) -> Any:
        """Get a service by name"""
        
        if name in self._services:
            return self._services[name]
        
        if name in self._factories:
            service = self._factories[name]()
            self._services[name] = service  # Cache the instance
            return service
        
        raise ValueError(f"Service '{name}' not found")
    
    def has_service(self, name: str) -> bool:
        """Check if a service is registered"""
        return name in self._services or name in self._factories

# Example usage
def create_application_with_di():
    """Create application with dependency injection"""
    
    container = DependencyContainer()
    
    # Register services
    container.register_factory('llm', lambda: OpenAI(temperature=0.7))
    container.register_factory('memory', lambda: ConversationBufferMemory())
    
    # Register conversation chain with dependencies
    def create_conversation_chain():
        return ConversationChain(
            llm=container.get_service('llm'),
            memory=container.get_service('memory')
        )
    
    container.register_factory('conversation', create_conversation_chain)
    
    # Create application with container
    app_config = {
        'temperature': 0.7,
        'openai_api_key': 'your-key-here'
    }
    
    app = LangChainApplication(app_config)
    app.container = container
    
    return app
```

### State Management

```python
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class ApplicationState:
    """Application state management"""
    
    current_user: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: list = None
    last_activity: Optional[datetime] = None
    error_count: int = 0
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        
        if self.last_activity is None:
            self.last_activity = datetime.now()

class StateManager:
    """Manage application state"""
    
    def __init__(self):
        self._states: Dict[str, ApplicationState] = {}
    
    def get_state(self, session_id: str) -> ApplicationState:
        """Get state for a session"""
        
        if session_id not in self._states:
            self._states[session_id] = ApplicationState(session_id=session_id)
        
        return self._states[session_id]
    
    def update_state(self, session_id: str, **updates) -> ApplicationState:
        """Update state for a session"""
        
        state = self.get_state(session_id)
        
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        
        state.last_activity = datetime.now()
        
        return state
    
    def cleanup_expired_states(self, max_age_hours: int = 24):
        """Clean up expired states"""
        
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        expired_sessions = [
            session_id for session_id, state in self._states.items()
            if state.last_activity < cutoff
        ]
        
        for session_id in expired_sessions:
            del self._states[session_id]
        
        return len(expired_sessions)
```

## Production Configuration

### Environment Configuration

```python
import os
from typing import Dict, Any

class Config:
    """Application configuration management"""
    
    def __init__(self, env: str = 'development'):
        self.env = env
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration based on environment"""
        
        base_config = {
            'debug': False,
            'testing': False,
            'log_level': 'INFO',
            'max_retries': 3,
            'timeout': 30,
        }
        
        if self.env == 'development':
            base_config.update({
                'debug': True,
                'log_level': 'DEBUG',
            })
        
        elif self.env == 'production':
            base_config.update({
                'log_level': 'WARNING',
                'max_retries': 5,
                'timeout': 60,
            })
        
        elif self.env == 'testing':
            base_config.update({
                'testing': True,
                'log_level': 'DEBUG',
                'timeout': 10,
            })
        
        # Override with environment variables
        for key in base_config:
            env_key = f'LANGCHAIN_{key.upper()}'
            if env_key in os.environ:
                base_config[key] = os.environ[env_key]
        
        return base_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.env == 'production'
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled"""
        return self._config.get('debug', False)
```

## Testing Application Structure

```python
import pytest
from unittest.mock import Mock, patch

def test_langchain_application_initialization():
    """Test application initialization"""
    
    config = {
        'temperature': 0.5,
        'openai_api_key': 'test-key'
    }
    
    with patch('langchain.llms.OpenAI'):
        app = LangChainApplication(config)
        
        assert app.config == config
        assert 'llm' in app.services
        assert 'memory' in app.services
        assert 'conversation' in app.services

def test_request_routing():
    """Test request routing"""
    
    config = {'openai_api_key': 'test-key'}
    
    with patch('langchain.llms.OpenAI'):
        app = LangChainApplication(config)
        
        # Test valid route
        with patch.object(app, 'handle_health_check', return_value={'status': 'ok'}):
            result = app.handle_request('/health')
            assert result['status'] == 'ok'
        
        # Test invalid route
        result = app.handle_request('/invalid')
        assert result['error'] == 'Not Found'
        assert result['status_code'] == 404

def test_error_handling():
    """Test error handling"""
    
    config = {'openai_api_key': 'test-key'}
    
    with patch('langchain.llms.OpenAI'):
        app = LangChainApplication(config)
        
        # Test server error handling
        with patch.object(app, 'handle_chat', side_effect=Exception('Test error')):
            result = app.handle_request('/chat', message='test')
            assert result['error'] == 'Internal Server Error'
            assert result['status_code'] == 500
```

This architecture provides a solid foundation for building scalable, maintainable LangChain applications with proper error handling, dependency management, and testing support.