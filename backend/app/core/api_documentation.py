"""
API documentation generator and endpoint testing utilities.
"""

import json
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.core.config import settings


class APIDocumentationGenerator:
    """Generate comprehensive API documentation."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.docs_dir = Path("docs/api")
        self.docs_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI specification."""
        return get_openapi(
            title=self.app.title,
            version=self.app.version,
            description=self.app.description,
            routes=self.app.routes,
        )
    
    def save_openapi_spec(self, format: str = "json") -> Path:
        """Save OpenAPI specification to file."""
        spec = self.generate_openapi_spec()
        
        if format.lower() == "json":
            file_path = self.docs_dir / "openapi.json"
            with open(file_path, "w") as f:
                json.dump(spec, f, indent=2)
        elif format.lower() == "yaml":
            file_path = self.docs_dir / "openapi.yaml"
            with open(file_path, "w") as f:
                yaml.dump(spec, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return file_path
    
    def generate_endpoint_documentation(self) -> Dict[str, Any]:
        """Generate detailed endpoint documentation."""
        endpoints = {}
        
        for route in self.app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                for method in route.methods:
                    if method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                        endpoint_key = f"{method} {route.path}"
                        
                        endpoints[endpoint_key] = {
                            "method": method,
                            "path": route.path,
                            "name": getattr(route, 'name', ''),
                            "summary": self._extract_summary(route),
                            "description": self._extract_description(route),
                            "parameters": self._extract_parameters(route),
                            "request_body": self._extract_request_body(route),
                            "responses": self._extract_responses(route),
                            "examples": self._generate_examples(route, method)
                        }
        
        return endpoints
    
    def _extract_summary(self, route) -> str:
        """Extract endpoint summary."""
        if hasattr(route, 'endpoint') and route.endpoint:
            doc = route.endpoint.__doc__
            if doc:
                return doc.split('\n')[0].strip()
        return ""
    
    def _extract_description(self, route) -> str:
        """Extract endpoint description."""
        if hasattr(route, 'endpoint') and route.endpoint:
            doc = route.endpoint.__doc__
            if doc:
                lines = doc.split('\n')[1:]
                return '\n'.join(line.strip() for line in lines if line.strip())
        return ""
    
    def _extract_parameters(self, route) -> List[Dict[str, Any]]:
        """Extract endpoint parameters."""
        parameters = []
        
        # Extract path parameters
        if hasattr(route, 'path_regex'):
            import re
            path_params = re.findall(r'\{([^}]+)\}', route.path)
            for param in path_params:
                parameters.append({
                    "name": param,
                    "in": "path",
                    "required": True,
                    "type": "string",
                    "description": f"Path parameter: {param}"
                })
        
        return parameters
    
    def _extract_request_body(self, route) -> Optional[Dict[str, Any]]:
        """Extract request body schema."""
        # This would need more sophisticated inspection of the route's endpoint
        # For now, return a placeholder
        return None
    
    def _extract_responses(self, route) -> Dict[str, Any]:
        """Extract response schemas."""
        # This would need more sophisticated inspection of the route's endpoint
        # For now, return common responses
        return {
            "200": {"description": "Successful response"},
            "400": {"description": "Bad request"},
            "404": {"description": "Not found"},
            "500": {"description": "Internal server error"}
        }
    
    def _generate_examples(self, route, method: str) -> Dict[str, Any]:
        """Generate example requests and responses."""
        examples = {}
        
        # Generate examples based on endpoint patterns
        if "/files/upload" in route.path and method == "POST":
            examples["request"] = {
                "description": "Upload audio or video file",
                "content_type": "multipart/form-data",
                "body": "Binary file data"
            }
            examples["response"] = {
                "status": 200,
                "body": {
                    "id": "file_123abc",
                    "filename": "voice_sample.mp3",
                    "file_size": 2048576,
                    "duration": 30.5,
                    "sample_rate": 44100,
                    "status": "uploaded"
                }
            }
        
        elif "/synthesis/synthesize" in route.path and method == "POST":
            examples["request"] = {
                "description": "Create speech synthesis task",
                "content_type": "application/json",
                "body": {
                    "text": "Hello, this is a test of voice synthesis.",
                    "voice_model_id": "voice_model_123",
                    "language": "english",
                    "voice_settings": {
                        "pitch_shift": 0.0,
                        "speed_factor": 1.0,
                        "emotion_intensity": 1.0
                    }
                }
            }
            examples["response"] = {
                "status": 200,
                "body": {
                    "task_id": "synthesis_abc123",
                    "status": "pending",
                    "message": "Synthesis task created successfully",
                    "estimated_completion": "2024-01-01T12:00:30Z"
                }
            }
        
        return examples
    
    def generate_markdown_documentation(self) -> str:
        """Generate markdown documentation."""
        endpoints = self.generate_endpoint_documentation()
        
        markdown = f"""# Voice Style Replication API Documentation

## Overview

{self.app.description}

**Version:** {self.app.version}
**Base URL:** `{settings.API_BASE_URL}/api/v1`

## Authentication

Currently, the API uses session-based authentication. Include the session ID in the `X-Session-ID` header.

## Error Handling

All endpoints return structured error responses with the following format:

```json
{{
  "error_id": "unique_error_identifier",
  "detail": "Human-readable error message",
  "category": "error_category",
  "is_retryable": true,
  "recovery_suggestions": [
    "Suggestion 1",
    "Suggestion 2"
  ]
}}
```

## Rate Limiting

API requests are rate-limited to prevent abuse. Rate limit information is included in response headers:

- `X-RateLimit-Limit`: Maximum requests per time window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## Endpoints

"""
        
        # Group endpoints by category
        categories = {
            "Files": [],
            "Text": [],
            "Voice": [],
            "Synthesis": [],
            "Performance": [],
            "Session": []
        }
        
        for endpoint_key, endpoint_info in endpoints.items():
            path = endpoint_info["path"]
            if "/files" in path:
                categories["Files"].append((endpoint_key, endpoint_info))
            elif "/text" in path:
                categories["Text"].append((endpoint_key, endpoint_info))
            elif "/voice" in path:
                categories["Voice"].append((endpoint_key, endpoint_info))
            elif "/synthesis" in path:
                categories["Synthesis"].append((endpoint_key, endpoint_info))
            elif "/performance" in path:
                categories["Performance"].append((endpoint_key, endpoint_info))
            elif "/session" in path:
                categories["Session"].append((endpoint_key, endpoint_info))
        
        # Generate documentation for each category
        for category, endpoints_list in categories.items():
            if endpoints_list:
                markdown += f"\n### {category}\n\n"
                
                for endpoint_key, endpoint_info in endpoints_list:
                    markdown += f"#### {endpoint_key}\n\n"
                    
                    if endpoint_info["summary"]:
                        markdown += f"**Summary:** {endpoint_info['summary']}\n\n"
                    
                    if endpoint_info["description"]:
                        markdown += f"{endpoint_info['description']}\n\n"
                    
                    # Parameters
                    if endpoint_info["parameters"]:
                        markdown += "**Parameters:**\n\n"
                        for param in endpoint_info["parameters"]:
                            required = " (required)" if param.get("required") else ""
                            markdown += f"- `{param['name']}` ({param['type']}){required}: {param.get('description', '')}\n"
                        markdown += "\n"
                    
                    # Examples
                    if endpoint_info["examples"]:
                        examples = endpoint_info["examples"]
                        
                        if "request" in examples:
                            markdown += "**Example Request:**\n\n"
                            markdown += f"```http\n{endpoint_info['method']} {endpoint_info['path']}\n"
                            markdown += f"Content-Type: {examples['request'].get('content_type', 'application/json')}\n\n"
                            
                            if isinstance(examples['request'].get('body'), dict):
                                markdown += json.dumps(examples['request']['body'], indent=2)
                            else:
                                markdown += str(examples['request'].get('body', ''))
                            
                            markdown += "\n```\n\n"
                        
                        if "response" in examples:
                            markdown += "**Example Response:**\n\n"
                            markdown += f"```http\nHTTP/1.1 {examples['response']['status']} OK\n"
                            markdown += "Content-Type: application/json\n\n"
                            markdown += json.dumps(examples['response']['body'], indent=2)
                            markdown += "\n```\n\n"
                    
                    markdown += "---\n\n"
        
        # Add additional sections
        markdown += self._generate_workflow_documentation()
        markdown += self._generate_error_codes_documentation()
        
        return markdown
    
    def _generate_workflow_documentation(self) -> str:
        """Generate workflow documentation."""
        return """
## Workflows

### Complete Voice Cloning Workflow

1. **Upload Reference Audio**
   ```http
   POST /api/v1/files/upload
   ```

2. **Validate and Process File**
   ```http
   GET /api/v1/files/{file_id}/status
   ```

3. **Analyze Voice Characteristics**
   ```http
   POST /api/v1/voice/analyze
   ```

4. **Validate Text Input**
   ```http
   POST /api/v1/text/validate
   ```

5. **Create Synthesis Task**
   ```http
   POST /api/v1/synthesis/synthesize
   ```

6. **Monitor Synthesis Progress**
   ```http
   GET /api/v1/synthesis/status/{task_id}
   ```

7. **Download Result**
   ```http
   GET /api/v1/synthesis/download/{task_id}
   ```

### Cross-Language Synthesis

For cross-language synthesis, use the specialized endpoint:

```http
POST /api/v1/synthesis/synthesize/cross-language
```

This maintains voice characteristics while adapting to the target language phonetics.

"""
    
    def _generate_error_codes_documentation(self) -> str:
        """Generate error codes documentation."""
        return """
## Error Codes

| Code | Category | Description | Retryable |
|------|----------|-------------|-----------|
| 400 | validation | Invalid request format or parameters | No |
| 401 | authentication | Authentication required or failed | No |
| 403 | authentication | Insufficient permissions | No |
| 404 | system | Resource not found | No |
| 408 | synthesis | Request timeout | Yes |
| 413 | file_processing | File too large | No |
| 415 | file_processing | Unsupported file format | No |
| 422 | validation | Validation error | No |
| 429 | rate_limit | Rate limit exceeded | Yes |
| 500 | system | Internal server error | Yes |
| 503 | system | Service temporarily unavailable | Yes |

## Response Headers

All API responses include the following headers:

- `X-Request-ID`: Unique identifier for the request
- `X-Response-Time`: Response time in milliseconds
- `X-API-Version`: API version used

## SDKs and Libraries

### JavaScript/TypeScript

Use the provided API client:

```typescript
import { apiClient } from './lib/api';

// Upload file and synthesize speech
const file = new File([audioData], 'voice.mp3');
const uploadResult = await apiClient.uploadFile(file);
const synthesisResult = await apiClient.synthesizeSpeech({
  text: "Hello world",
  voice_model_id: uploadResult.voice_model_id,
  language: "english"
});
```

### Python

```python
import requests

# Upload file
with open('voice.mp3', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/files/upload',
        files={'file': f}
    )
    upload_result = response.json()

# Synthesize speech
synthesis_response = requests.post(
    'http://localhost:8000/api/v1/synthesis/synthesize',
    json={
        'text': 'Hello world',
        'voice_model_id': upload_result['voice_model_id'],
        'language': 'english'
    }
)
```

"""
    
    def save_markdown_documentation(self) -> Path:
        """Save markdown documentation to file."""
        markdown = self.generate_markdown_documentation()
        file_path = self.docs_dir / "README.md"
        
        with open(file_path, "w") as f:
            f.write(markdown)
        
        return file_path
    
    def generate_postman_collection(self) -> Dict[str, Any]:
        """Generate Postman collection for API testing."""
        endpoints = self.generate_endpoint_documentation()
        
        collection = {
            "info": {
                "name": self.app.title,
                "description": self.app.description,
                "version": self.app.version,
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "variable": [
                {
                    "key": "baseUrl",
                    "value": f"{settings.API_BASE_URL}/api/v1",
                    "type": "string"
                }
            ],
            "item": []
        }
        
        # Group requests by category
        categories = {}
        for endpoint_key, endpoint_info in endpoints.items():
            path = endpoint_info["path"]
            category = "Other"
            
            if "/files" in path:
                category = "Files"
            elif "/text" in path:
                category = "Text"
            elif "/voice" in path:
                category = "Voice"
            elif "/synthesis" in path:
                category = "Synthesis"
            elif "/performance" in path:
                category = "Performance"
            elif "/session" in path:
                category = "Session"
            
            if category not in categories:
                categories[category] = []
            
            # Create Postman request
            request = {
                "name": f"{endpoint_info['method']} {endpoint_info['path']}",
                "request": {
                    "method": endpoint_info["method"],
                    "header": [
                        {
                            "key": "Content-Type",
                            "value": "application/json",
                            "type": "text"
                        }
                    ],
                    "url": {
                        "raw": "{{baseUrl}}" + endpoint_info["path"],
                        "host": ["{{baseUrl}}"],
                        "path": endpoint_info["path"].strip("/").split("/")
                    }
                }
            }
            
            # Add request body if available
            if endpoint_info["examples"] and "request" in endpoint_info["examples"]:
                example = endpoint_info["examples"]["request"]
                if isinstance(example.get("body"), dict):
                    request["request"]["body"] = {
                        "mode": "raw",
                        "raw": json.dumps(example["body"], indent=2)
                    }
            
            categories[category].append(request)
        
        # Add categories to collection
        for category, requests in categories.items():
            collection["item"].append({
                "name": category,
                "item": requests
            })
        
        return collection
    
    def save_postman_collection(self) -> Path:
        """Save Postman collection to file."""
        collection = self.generate_postman_collection()
        file_path = self.docs_dir / "postman_collection.json"
        
        with open(file_path, "w") as f:
            json.dump(collection, f, indent=2)
        
        return file_path
    
    def generate_all_documentation(self) -> Dict[str, Path]:
        """Generate all documentation formats."""
        return {
            "openapi_json": self.save_openapi_spec("json"),
            "openapi_yaml": self.save_openapi_spec("yaml"),
            "markdown": self.save_markdown_documentation(),
            "postman": self.save_postman_collection()
        }


def generate_api_documentation(app: FastAPI) -> Dict[str, Path]:
    """Generate comprehensive API documentation."""
    generator = APIDocumentationGenerator(app)
    return generator.generate_all_documentation()