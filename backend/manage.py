#!/usr/bin/env python3
"""
Management CLI for Voice Style Replication system.
Provides commands for testing, documentation generation, and system management.
"""

import asyncio
import click
import json
import sys
from pathlib import Path
from typing import Optional

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import app
from app.core.api_documentation import generate_api_documentation
from app.core.endpoint_testing import run_comprehensive_tests, run_load_tests
from app.core.logging_config import setup_logging, get_logger
from app.core.error_handling import error_recovery_manager

# Set up logging
setup_logging()
logger = get_logger("manage")


@click.group()
def cli():
    """Voice Style Replication Management CLI."""
    pass


@cli.group()
def docs():
    """Documentation management commands."""
    pass


@docs.command()
@click.option('--output-dir', '-o', default='docs/api', help='Output directory for documentation')
def generate(output_dir: str):
    """Generate comprehensive API documentation."""
    click.echo("Generating API documentation...")
    
    try:
        # Set output directory
        from app.core.api_documentation import APIDocumentationGenerator
        generator = APIDocumentationGenerator(app)
        generator.docs_dir = Path(output_dir)
        
        # Generate all documentation
        files = generator.generate_all_documentation()
        
        click.echo(f"Documentation generated successfully:")
        for doc_type, file_path in files.items():
            click.echo(f"  {doc_type}: {file_path}")
        
        logger.info(f"API documentation generated in {output_dir}")
        
    except Exception as e:
        click.echo(f"Error generating documentation: {str(e)}", err=True)
        logger.error(f"Documentation generation failed: {str(e)}")
        sys.exit(1)


@docs.command()
@click.option('--format', '-f', type=click.Choice(['json', 'yaml']), default='json', help='Output format')
@click.option('--output', '-o', help='Output file path')
def openapi(format: str, output: Optional[str]):
    """Generate OpenAPI specification."""
    click.echo(f"Generating OpenAPI spec in {format} format...")
    
    try:
        from app.core.api_documentation import APIDocumentationGenerator
        generator = APIDocumentationGenerator(app)
        
        if output:
            generator.docs_dir = Path(output).parent
            file_path = generator.save_openapi_spec(format)
            if output != str(file_path):
                # Move to specified location
                Path(file_path).rename(output)
                file_path = Path(output)
        else:
            file_path = generator.save_openapi_spec(format)
        
        click.echo(f"OpenAPI spec generated: {file_path}")
        logger.info(f"OpenAPI spec generated: {file_path}")
        
    except Exception as e:
        click.echo(f"Error generating OpenAPI spec: {str(e)}", err=True)
        logger.error(f"OpenAPI generation failed: {str(e)}")
        sys.exit(1)


@cli.group()
def test():
    """Testing commands."""
    pass


@test.command()
@click.option('--output', '-o', help='Output file for test report')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def endpoints(output: Optional[str], verbose: bool):
    """Run comprehensive endpoint tests."""
    click.echo("Running comprehensive endpoint tests...")
    
    try:
        # Run tests
        results = asyncio.run(run_comprehensive_tests())
        
        report = results['report']
        summary = report['summary']
        
        # Display summary
        click.echo(f"\nTest Results:")
        click.echo(f"  Total tests: {summary['total_tests']}")
        click.echo(f"  Passed: {summary['passed']}")
        click.echo(f"  Failed: {summary['failed']}")
        click.echo(f"  Success rate: {summary['success_rate']:.1f}%")
        click.echo(f"  Average response time: {summary['average_response_time_ms']:.2f}ms")
        
        # Show failed tests
        if report['failed_tests']:
            click.echo(f"\nFailed Tests:")
            for failed_test in report['failed_tests']:
                click.echo(f"  - {failed_test['endpoint']}: {failed_test['error']}")
        
        # Show performance metrics
        perf = report['performance_metrics']
        click.echo(f"\nPerformance Metrics:")
        click.echo(f"  Fastest response: {perf['fastest_response_ms']:.2f}ms")
        click.echo(f"  Slowest response: {perf['slowest_response_ms']:.2f}ms")
        
        # Verbose output
        if verbose:
            click.echo(f"\nDetailed Results:")
            for result in report['results']:
                status = "PASS" if result['success'] else "FAIL"
                click.echo(f"  {result['endpoint']}: {status} ({result['response_time_ms']:.2f}ms)")
        
        # Save report if requested
        if output:
            with open(output, 'w') as f:
                json.dump(report, f, indent=2)
            click.echo(f"\nDetailed report saved to: {output}")
        
        logger.info(f"Endpoint tests completed: {summary['passed']}/{summary['total_tests']} passed")
        
        # Exit with error code if tests failed
        if summary['failed'] > 0:
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"Error running tests: {str(e)}", err=True)
        logger.error(f"Endpoint testing failed: {str(e)}")
        sys.exit(1)


@test.command()
@click.option('--output', '-o', help='Output file for load test report')
@click.option('--requests', '-r', default=30, help='Number of concurrent requests')
def load(output: Optional[str], requests: int):
    """Run load tests."""
    click.echo(f"Running load tests with {requests} concurrent requests...")
    
    try:
        # Run load tests
        results = asyncio.run(run_load_tests())
        
        report = results['report']
        summary = report['summary']
        load_metrics = report['load_test_metrics']
        
        # Display results
        click.echo(f"\nLoad Test Results:")
        click.echo(f"  Total requests: {load_metrics['total_requests']}")
        click.echo(f"  Total time: {load_metrics['total_time_seconds']:.2f}s")
        click.echo(f"  Requests per second: {load_metrics['requests_per_second']:.2f}")
        click.echo(f"  Success rate: {summary['success_rate']:.1f}%")
        click.echo(f"  Average response time: {summary['average_response_time_ms']:.2f}ms")
        
        # Performance distribution
        dist = report['performance_metrics']['response_time_distribution']
        click.echo(f"\nResponse Time Distribution:")
        click.echo(f"  Under 100ms: {dist['under_100ms']}")
        click.echo(f"  100-500ms: {dist['100_500ms']}")
        click.echo(f"  500-1000ms: {dist['500_1000ms']}")
        click.echo(f"  1000-5000ms: {dist['1000_5000ms']}")
        click.echo(f"  Over 5000ms: {dist['over_5000ms']}")
        
        # Save report if requested
        if output:
            with open(output, 'w') as f:
                json.dump(report, f, indent=2)
            click.echo(f"\nDetailed report saved to: {output}")
        
        logger.info(f"Load tests completed: {load_metrics['requests_per_second']:.2f} req/s")
        
    except Exception as e:
        click.echo(f"Error running load tests: {str(e)}", err=True)
        logger.error(f"Load testing failed: {str(e)}")
        sys.exit(1)


@cli.group()
def system():
    """System management commands."""
    pass


@system.command()
def status():
    """Show system status and health."""
    click.echo("Checking system status...")
    
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Check basic health
        health_response = client.get("/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            click.echo(f"System Status: {health_data['status']}")
            click.echo(f"Database: {health_data['database']}")
            click.echo(f"Performance Monitoring: {health_data['performance_monitoring']}")
        else:
            click.echo("System appears to be unhealthy", err=True)
            sys.exit(1)
        
        # Check performance metrics
        metrics_response = client.get("/api/v1/performance/metrics")
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            click.echo(f"\nPerformance Metrics:")
            click.echo(f"  Queue Length: {metrics['queue_length']}")
            click.echo(f"  System Load: {metrics['system_load']:.2f}")
            click.echo(f"  Memory Usage: {metrics['memory_usage']:.1f}%")
            click.echo(f"  Active Tasks: {metrics['active_tasks']}")
        
        logger.info("System status check completed")
        
    except Exception as e:
        click.echo(f"Error checking system status: {str(e)}", err=True)
        logger.error(f"System status check failed: {str(e)}")
        sys.exit(1)


@system.command()
def errors():
    """Show error statistics and recent errors."""
    click.echo("Retrieving error statistics...")
    
    try:
        stats = error_recovery_manager.get_error_statistics()
        
        click.echo(f"Error Statistics:")
        click.echo(f"  Total errors: {stats['total_errors']}")
        click.echo(f"  Recent errors (last hour): {stats['recent_errors']}")
        
        if stats.get('category_breakdown'):
            click.echo(f"\nErrors by Category:")
            for category, count in stats['category_breakdown'].items():
                click.echo(f"  {category}: {count}")
        
        if stats.get('severity_breakdown'):
            click.echo(f"\nErrors by Severity:")
            for severity, count in stats['severity_breakdown'].items():
                click.echo(f"  {severity}: {count}")
        
        if stats.get('circuit_breaker_states'):
            click.echo(f"\nCircuit Breaker States:")
            for service, state in stats['circuit_breaker_states'].items():
                click.echo(f"  {service}: {state}")
        
        logger.info("Error statistics retrieved")
        
    except Exception as e:
        click.echo(f"Error retrieving error statistics: {str(e)}", err=True)
        logger.error(f"Error statistics retrieval failed: {str(e)}")
        sys.exit(1)


@system.command()
@click.option('--lines', '-n', default=50, help='Number of log lines to show')
@click.option('--level', '-l', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), 
              default='INFO', help='Minimum log level')
def logs(lines: int, level: str):
    """Show recent log entries."""
    click.echo(f"Showing last {lines} log entries (level: {level})...")
    
    try:
        from app.core.config import settings
        log_file = Path(settings.LOG_DIR) / "voice_replication.log"
        
        if not log_file.exists():
            click.echo("Log file not found", err=True)
            sys.exit(1)
        
        # Read log file and filter by level
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
        
        # Filter by level and get last N lines
        level_priority = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
        min_priority = level_priority[level]
        
        filtered_lines = []
        for line in log_lines:
            for log_level, priority in level_priority.items():
                if log_level in line and priority >= min_priority:
                    filtered_lines.append(line.strip())
                    break
        
        # Show last N lines
        recent_lines = filtered_lines[-lines:] if len(filtered_lines) > lines else filtered_lines
        
        for line in recent_lines:
            click.echo(line)
        
        click.echo(f"\nShowing {len(recent_lines)} of {len(filtered_lines)} matching log entries")
        
    except Exception as e:
        click.echo(f"Error reading logs: {str(e)}", err=True)
        logger.error(f"Log reading failed: {str(e)}")
        sys.exit(1)


@cli.group()
def db():
    """Database management commands."""
    pass


@db.command()
def init():
    """Initialize database tables."""
    click.echo("Initializing database...")
    
    try:
        from app.core.database import engine, Base
        Base.metadata.create_all(bind=engine)
        click.echo("Database initialized successfully")
        logger.info("Database initialized")
        
    except Exception as e:
        click.echo(f"Error initializing database: {str(e)}", err=True)
        logger.error(f"Database initialization failed: {str(e)}")
        sys.exit(1)


@db.command()
def reset():
    """Reset database (drop and recreate all tables)."""
    if not click.confirm("This will delete all data. Are you sure?"):
        click.echo("Operation cancelled")
        return
    
    click.echo("Resetting database...")
    
    try:
        from app.core.database import engine, Base
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        click.echo("Database reset successfully")
        logger.info("Database reset")
        
    except Exception as e:
        click.echo(f"Error resetting database: {str(e)}", err=True)
        logger.error(f"Database reset failed: {str(e)}")
        sys.exit(1)


@cli.command()
def dev():
    """Start development server with auto-reload."""
    click.echo("Starting development server...")
    
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        click.echo("\nDevelopment server stopped")
    except Exception as e:
        click.echo(f"Error starting development server: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--workers', default=1, help='Number of worker processes')
def serve(host: str, port: int, workers: int):
    """Start production server."""
    click.echo(f"Starting production server on {host}:{port} with {workers} workers...")
    
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        click.echo("\nProduction server stopped")
    except Exception as e:
        click.echo(f"Error starting production server: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def version():
    """Show version information."""
    click.echo("Voice Style Replication System")
    click.echo("Version: 1.0.0")
    click.echo("API Version: v1")
    
    # Show additional system info
    try:
        import platform
        import sys
        
        click.echo(f"\nSystem Information:")
        click.echo(f"  Python: {sys.version}")
        click.echo(f"  Platform: {platform.platform()}")
        click.echo(f"  Architecture: {platform.architecture()[0]}")
        
    except Exception:
        pass


if __name__ == '__main__':
    cli()