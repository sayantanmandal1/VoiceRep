"""
Comprehensive endpoint testing utilities for Voice Style Replication API.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger("endpoint_testing")


class TestResult(BaseModel):
    """Test result model."""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None


class EndpointTest(BaseModel):
    """Endpoint test configuration."""
    name: str
    method: str
    path: str
    headers: Optional[Dict[str, str]] = None
    query_params: Optional[Dict[str, str]] = None
    json_data: Optional[Dict[str, Any]] = None
    files: Optional[Dict[str, Any]] = None
    expected_status: int = 200
    timeout: float = 30.0
    description: Optional[str] = None


class EndpointTester:
    """Comprehensive endpoint testing utility."""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or f"http://localhost:8000/api/v1"
        self.session_id: Optional[str] = None
        self.test_results: List[TestResult] = []
    
    async def setup_session(self) -> bool:
        """Set up test session."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/session/create")
                if response.status_code == 200:
                    data = response.json()
                    self.session_id = data.get("session_id")
                    logger.info(f"Test session created: {self.session_id}")
                    return True
                else:
                    logger.error(f"Failed to create session: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            return False
    
    async def run_test(self, test: EndpointTest) -> TestResult:
        """Run a single endpoint test."""
        start_time = time.time()
        
        try:
            headers = test.headers or {}
            if self.session_id:
                headers["X-Session-ID"] = self.session_id
            
            async with httpx.AsyncClient(timeout=test.timeout) as client:
                url = f"{self.base_url}{test.path}"
                
                # Prepare request parameters
                request_params = {
                    "url": url,
                    "headers": headers,
                    "params": test.query_params
                }
                
                if test.json_data:
                    request_params["json"] = test.json_data
                
                if test.files:
                    request_params["files"] = test.files
                
                # Make request
                response = await client.request(test.method, **request_params)
                
                response_time = (time.time() - start_time) * 1000
                
                # Parse response
                try:
                    response_data = response.json()
                except:
                    response_data = {"raw_response": response.text}
                
                # Determine success
                success = response.status_code == test.expected_status
                error_message = None if success else f"Expected {test.expected_status}, got {response.status_code}"
                
                result = TestResult(
                    endpoint=f"{test.method} {test.path}",
                    method=test.method,
                    status_code=response.status_code,
                    response_time_ms=response_time,
                    success=success,
                    error_message=error_message,
                    response_data=response_data
                )
                
                logger.info(f"Test completed: {test.name} - {'PASS' if success else 'FAIL'}")
                return result
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            result = TestResult(
                endpoint=f"{test.method} {test.path}",
                method=test.method,
                status_code=0,
                response_time_ms=response_time,
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Test failed: {test.name} - {str(e)}")
            return result
    
    async def run_test_suite(self, tests: List[EndpointTest]) -> List[TestResult]:
        """Run a suite of endpoint tests."""
        logger.info(f"Starting test suite with {len(tests)} tests")
        
        # Set up session
        await self.setup_session()
        
        results = []
        for test in tests:
            result = await self.run_test(test)
            results.append(result)
            self.test_results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(0.1)
        
        logger.info(f"Test suite completed: {sum(1 for r in results if r.success)}/{len(results)} passed")
        return results
    
    def generate_test_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        
        avg_response_time = sum(r.response_time_ms for r in results) / total_tests if total_tests > 0 else 0
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "average_response_time_ms": round(avg_response_time, 2)
            },
            "results": [result.model_dump() for result in results],
            "failed_tests": [
                {
                    "endpoint": r.endpoint,
                    "error": r.error_message,
                    "status_code": r.status_code
                }
                for r in results if not r.success
            ],
            "performance_metrics": {
                "fastest_response_ms": min(r.response_time_ms for r in results) if results else 0,
                "slowest_response_ms": max(r.response_time_ms for r in results) if results else 0,
                "response_time_distribution": self._calculate_response_time_distribution(results)
            }
        }
        
        return report
    
    def _calculate_response_time_distribution(self, results: List[TestResult]) -> Dict[str, int]:
        """Calculate response time distribution."""
        if not results:
            return {}
        
        distribution = {
            "under_100ms": 0,
            "100_500ms": 0,
            "500_1000ms": 0,
            "1000_5000ms": 0,
            "over_5000ms": 0
        }
        
        for result in results:
            time_ms = result.response_time_ms
            if time_ms < 100:
                distribution["under_100ms"] += 1
            elif time_ms < 500:
                distribution["100_500ms"] += 1
            elif time_ms < 1000:
                distribution["500_1000ms"] += 1
            elif time_ms < 5000:
                distribution["1000_5000ms"] += 1
            else:
                distribution["over_5000ms"] += 1
        
        return distribution
    
    def save_test_report(self, results: List[TestResult], file_path: str = None) -> Path:
        """Save test report to file."""
        if not file_path:
            file_path = f"test_report_{int(time.time())}.json"
        
        report = self.generate_test_report(results)
        
        path = Path(file_path)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to: {path}")
        return path


def create_comprehensive_test_suite() -> List[EndpointTest]:
    """Create comprehensive test suite for all endpoints."""
    tests = [
        # Health checks
        EndpointTest(
            name="Health Check",
            method="GET",
            path="/",
            description="Basic health check endpoint"
        ),
        
        EndpointTest(
            name="Detailed Health Check",
            method="GET",
            path="/health",
            description="Comprehensive health check"
        ),
        
        # Session management
        EndpointTest(
            name="Create Session",
            method="POST",
            path="/session/create",
            description="Create new user session"
        ),
        
        EndpointTest(
            name="Get Session Info",
            method="GET",
            path="/session/info",
            description="Get current session information"
        ),
        
        # File operations
        EndpointTest(
            name="File Upload - Invalid Format",
            method="POST",
            path="/files/upload",
            files={"file": ("test.txt", b"invalid file content", "text/plain")},
            expected_status=415,
            description="Test file upload with invalid format"
        ),
        
        # Text validation
        EndpointTest(
            name="Text Validation - Valid Text",
            method="POST",
            path="/text/validate",
            json_data={
                "text": "Hello, this is a test message for voice synthesis.",
                "language": "english"
            },
            description="Validate normal text input"
        ),
        
        EndpointTest(
            name="Text Validation - Empty Text",
            method="POST",
            path="/text/validate",
            json_data={"text": ""},
            expected_status=422,
            description="Test validation with empty text"
        ),
        
        EndpointTest(
            name="Text Validation - Long Text",
            method="POST",
            path="/text/validate",
            json_data={"text": "A" * 1500},  # Exceeds 1000 character limit
            expected_status=422,
            description="Test validation with text exceeding length limit"
        ),
        
        EndpointTest(
            name="Language Detection",
            method="POST",
            path="/text/detect-language",
            json_data={"text": "Bonjour, comment allez-vous?"},
            description="Test language detection"
        ),
        
        # Voice analysis (would need actual file)
        EndpointTest(
            name="Voice Analysis - Missing File",
            method="POST",
            path="/voice/analyze",
            json_data={"reference_audio_id": "nonexistent_file"},
            expected_status=404,
            description="Test voice analysis with missing file"
        ),
        
        # Synthesis
        EndpointTest(
            name="Synthesis - Missing Voice Model",
            method="POST",
            path="/synthesis/synthesize",
            json_data={
                "text": "Test synthesis",
                "voice_model_id": "nonexistent_model",
                "language": "english"
            },
            expected_status=404,
            description="Test synthesis with missing voice model"
        ),
        
        EndpointTest(
            name="Synthesis Status - Invalid Task",
            method="GET",
            path="/synthesis/status/invalid_task_id",
            expected_status=404,
            description="Test synthesis status with invalid task ID"
        ),
        
        # Performance monitoring
        EndpointTest(
            name="Performance Metrics",
            method="GET",
            path="/performance/metrics",
            description="Get system performance metrics"
        ),
        
        EndpointTest(
            name="System Status",
            method="GET",
            path="/performance/status",
            description="Get system status information"
        ),
        
        # Cross-language synthesis
        EndpointTest(
            name="Cross-Language Synthesis - Missing Model",
            method="POST",
            path="/synthesis/synthesize/cross-language",
            json_data={
                "text": "Hello world",
                "source_voice_model_id": "nonexistent_model",
                "target_language": "spanish"
            },
            expected_status=404,
            description="Test cross-language synthesis with missing model"
        ),
        
        # Synthesis statistics
        EndpointTest(
            name="Synthesis Statistics",
            method="GET",
            path="/synthesis/stats",
            description="Get synthesis statistics"
        )
    ]
    
    return tests


def create_load_test_suite() -> List[EndpointTest]:
    """Create load testing suite for performance testing."""
    tests = []
    
    # Create multiple concurrent requests to test endpoints
    for i in range(10):
        tests.extend([
            EndpointTest(
                name=f"Load Test Health Check {i+1}",
                method="GET",
                path="/health",
                description=f"Load test health check request {i+1}"
            ),
            
            EndpointTest(
                name=f"Load Test Text Validation {i+1}",
                method="POST",
                path="/text/validate",
                json_data={
                    "text": f"Load test message number {i+1} for performance testing.",
                    "language": "english"
                },
                description=f"Load test text validation request {i+1}"
            ),
            
            EndpointTest(
                name=f"Load Test Performance Metrics {i+1}",
                method="GET",
                path="/performance/metrics",
                description=f"Load test performance metrics request {i+1}"
            )
        ])
    
    return tests


async def run_comprehensive_tests() -> Dict[str, Any]:
    """Run comprehensive endpoint tests."""
    tester = EndpointTester()
    
    # Run main test suite
    logger.info("Running comprehensive test suite")
    main_tests = create_comprehensive_test_suite()
    main_results = await tester.run_test_suite(main_tests)
    
    # Generate and save report
    report = tester.generate_test_report(main_results)
    report_path = tester.save_test_report(main_results, "comprehensive_test_report.json")
    
    return {
        "report": report,
        "report_path": str(report_path),
        "results": main_results
    }


async def run_load_tests() -> Dict[str, Any]:
    """Run load tests for performance evaluation."""
    tester = EndpointTester()
    
    logger.info("Running load test suite")
    load_tests = create_load_test_suite()
    
    # Run tests concurrently for load testing
    start_time = time.time()
    tasks = [tester.run_test(test) for test in load_tests]
    load_results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Generate report with load testing metrics
    report = tester.generate_test_report(load_results)
    report["load_test_metrics"] = {
        "total_requests": len(load_tests),
        "total_time_seconds": round(total_time, 2),
        "requests_per_second": round(len(load_tests) / total_time, 2),
        "concurrent_requests": len(load_tests)
    }
    
    report_path = tester.save_test_report(load_results, "load_test_report.json")
    
    return {
        "report": report,
        "report_path": str(report_path),
        "results": load_results
    }


if __name__ == "__main__":
    # Run tests when script is executed directly
    async def main():
        print("Running comprehensive endpoint tests...")
        comprehensive_results = await run_comprehensive_tests()
        print(f"Comprehensive tests completed: {comprehensive_results['report']['summary']}")
        
        print("\nRunning load tests...")
        load_results = await run_load_tests()
        print(f"Load tests completed: {load_results['report']['summary']}")
        print(f"Load test metrics: {load_results['report']['load_test_metrics']}")
    
    asyncio.run(main())