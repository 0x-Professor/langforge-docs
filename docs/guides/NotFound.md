# Error Handling and Not Found Pages

This guide covers how to implement proper error handling and create user-friendly not found pages in LangChain applications, particularly when building conversational AI systems and document retrieval applications.

## Overview

Proper error handling is crucial for building robust LangChain applications. Users should receive helpful feedback when:

- Requested information cannot be found
- LLM calls fail or timeout
- Document retrieval returns no results
- Chain execution encounters errors

## Implementing Error Handling in LangChain

### Basic Error Handling

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_llm_call(prompt_text: str, retries: int = 3):
    """Safely call LLM with error handling and retries"""
    
    llm = OpenAI(temperature=0.7)
    prompt = PromptTemplate(template="{text}", input_variables=["text"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    for attempt in range(retries):
        try:
            result = chain.run(text=prompt_text)
            return {"success": True, "result": result}
        
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == retries - 1:
                return {
                    "success": False, 
                    "error": "Unable to process request after multiple attempts",
                    "details": str(e)
                }
    
    return {"success": False, "error": "Unknown error occurred"}
```

### Document Retrieval Error Handling

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

def safe_document_search(query: str, vectorstore: FAISS, min_score: float = 0.7):
    """Search documents with relevance filtering"""
    
    try:
        # Perform similarity search with scores
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=5)
        
        # Filter by relevance score
        relevant_docs = [
            doc for doc, score in docs_and_scores 
            if score >= min_score
        ]
        
        if not relevant_docs:
            return {
                "success": False,
                "error": "No relevant documents found",
                "suggestion": "Try rephrasing your question or using different keywords"
            }
        
        return {"success": True, "documents": relevant_docs}
        
    except Exception as e:
        logger.error(f"Document search failed: {str(e)}")
        return {
            "success": False,
            "error": "Search functionality is temporarily unavailable",
            "details": str(e)
        }
```

### Conversational AI Error Responses

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class RobustConversationChain:
    def __init__(self):
        self.llm = OpenAI(temperature=0.7)
        self.memory = ConversationBufferMemory()
        self.chain = ConversationChain(llm=self.llm, memory=self.memory)
        
        # Error response templates
        self.error_responses = {
            "not_found": "I couldn't find information about that topic. Could you please provide more details or try a different question?",
            "api_error": "I'm experiencing technical difficulties right now. Please try again in a moment.",
            "unclear_request": "I'm not sure I understand your request. Could you please rephrase or provide more context?",
            "timeout": "That request is taking longer than expected. Let me try to help you with something else."
        }
    
    def get_response(self, user_input: str) -> dict:
        """Get response with comprehensive error handling"""
        
        try:
            # Validate input
            if not user_input.strip():
                return {
                    "response": "Please provide a question or message for me to respond to.",
                    "error_type": "empty_input"
                }
            
            # Get AI response
            response = self.chain.predict(input=user_input)
            
            # Check for potentially problematic responses
            if len(response.strip()) < 10:
                return {
                    "response": self.error_responses["unclear_request"],
                    "error_type": "short_response"
                }
            
            return {"response": response, "success": True}
            
        except Exception as e:
            logger.error(f"Conversation failed: {str(e)}")
            return {
                "response": self.error_responses["api_error"],
                "error_type": "api_error",
                "details": str(e)
            }
```

## User-Friendly Error Messages

### Creating Helpful Error Responses

```python
class ErrorMessageHandler:
    """Generate user-friendly error messages for different scenarios"""
    
    @staticmethod
    def format_not_found_message(query: str, suggestions: list = None):
        """Format a helpful message when content is not found"""
        
        message = f"I couldn't find specific information about '{query}'."
        
        if suggestions:
            message += "\n\nHere are some related topics that might help:"
            for suggestion in suggestions:
                message += f"\n• {suggestion}"
        
        message += "\n\nTry rephrasing your question or being more specific."
        
        return message
    
    @staticmethod
    def format_system_error_message(error_type: str = "general"):
        """Format system error messages"""
        
        messages = {
            "rate_limit": "I'm receiving too many requests right now. Please wait a moment and try again.",
            "timeout": "Your request is taking longer than expected. Please try again or simplify your question.",
            "api_down": "I'm temporarily unable to access my knowledge base. Please try again later.",
            "general": "Something went wrong while processing your request. Please try again."
        }
        
        return messages.get(error_type, messages["general"])
```

## Best Practices

### Error Logging and Monitoring

```python
import json
from datetime import datetime

class ErrorLogger:
    """Log errors for monitoring and debugging"""
    
    def __init__(self, log_file: str = "langchain_errors.log"):
        self.log_file = log_file
    
    def log_error(self, error_type: str, details: dict):
        """Log error with context"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "details": details
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
```

### Graceful Degradation

```python
def search_with_fallback(query: str, primary_retriever, fallback_retriever):
    """Implement fallback search mechanisms"""
    
    try:
        # Try primary search method
        results = primary_retriever.get_relevant_documents(query)
        if results:
            return {"source": "primary", "results": results}
    except Exception as e:
        logger.warning(f"Primary search failed: {e}")
    
    try:
        # Fallback to secondary search
        results = fallback_retriever.get_relevant_documents(query)
        if results:
            return {"source": "fallback", "results": results}
    except Exception as e:
        logger.error(f"Fallback search also failed: {e}")
    
    return {
        "source": "none",
        "results": [],
        "message": "Unable to retrieve relevant information at this time"
    }
```

## Common Error Scenarios

1. **No Relevant Documents Found**: Provide search suggestions and related topics
2. **API Rate Limits**: Implement retry logic with exponential backoff
3. **Network Timeouts**: Graceful degradation with cached responses
4. **Invalid User Input**: Clear validation messages and examples
5. **System Overload**: Queue requests or provide estimated wait times

## Testing Error Handling

```python
def test_error_scenarios():
    """Test various error conditions"""
    
    # Test empty query
    response = safe_document_search("")
    assert not response["success"]
    
    # Test malformed input
    response = safe_llm_call("@#$%^&*()")
    assert "error" in response
    
    # Test with no results
    response = safe_document_search("extremely specific nonsense query")
    assert "No relevant documents found" in response.get("error", "")
```