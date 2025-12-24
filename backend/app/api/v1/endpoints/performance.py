"""
API endpoints for performance monitoring and metrics.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
from datetime import datetime

from app.services.performance_monitoring_service import performance_monitor, queue_manager

router = APIRouter()


@router.get("/metrics/summary")
async def get_performance_summary():
    """
    Get comprehensive performance summary.
    
    Returns performance statistics, resource usage, and threshold compliance.
    """
    try:
        summary = performance_monitor.get_performance_summary()
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")


@router.get("/metrics/operations")
async def get_operation_statistics(
    operation_type: Optional[str] = Query(None, description="Filter by operation type")
):
    """
    Get performance statistics for operations.
    
    Args:
        operation_type: Optional filter for specific operation type
    
    Returns operation statistics including count, timing, and success rates.
    """
    try:
        stats = performance_monitor.get_operation_statistics(operation_type)
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get operation statistics: {str(e)}")


@router.get("/metrics/resources")
async def get_resource_usage(
    minutes: int = Query(60, ge=1, le=1440, description="Time window in minutes")
):
    """
    Get recent resource usage data.
    
    Args:
        minutes: Time window for resource data (1-1440 minutes)
    
    Returns CPU, memory, and disk usage over the specified time window.
    """
    try:
        usage_data = performance_monitor.get_resource_usage(minutes=minutes)
        return {
            "status": "success",
            "data": usage_data,
            "time_window_minutes": minutes,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get resource usage: {str(e)}")


@router.get("/metrics/estimate")
async def estimate_completion_time(
    operation_type: str = Query(..., description="Type of operation"),
    input_size: Optional[float] = Query(None, description="Optional input size for better estimation")
):
    """
    Estimate completion time for an operation.
    
    Args:
        operation_type: Type of operation (e.g., 'voice_analysis', 'speech_synthesis')
        input_size: Optional input size parameter
    
    Returns estimated completion time in seconds.
    """
    try:
        estimated_time = performance_monitor.estimate_completion_time(operation_type, input_size)
        
        if estimated_time is None:
            return {
                "status": "success",
                "data": {
                    "estimated_time_seconds": None,
                    "message": "No historical data available for this operation type"
                },
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "status": "success",
            "data": {
                "operation_type": operation_type,
                "estimated_time_seconds": estimated_time,
                "estimated_time_formatted": f"{estimated_time:.1f}s"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to estimate completion time: {str(e)}")


@router.post("/metrics/export")
async def export_performance_metrics():
    """
    Export performance metrics to JSON file.
    
    Returns path to exported metrics file.
    """
    try:
        filepath = performance_monitor.export_metrics()
        return {
            "status": "success",
            "data": {
                "filepath": filepath,
                "message": "Performance metrics exported successfully"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")


@router.get("/queue/status")
async def get_queue_status():
    """
    Get current queue status and wait times.
    
    Returns queue sizes, estimated wait times, and processing statistics.
    """
    try:
        status = queue_manager.get_queue_status()
        return {
            "status": "success",
            "data": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")


@router.post("/monitoring/start")
async def start_monitoring():
    """
    Start background performance monitoring.
    
    Begins continuous monitoring of system resources.
    """
    try:
        performance_monitor.start_monitoring()
        return {
            "status": "success",
            "message": "Performance monitoring started",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.post("/monitoring/stop")
async def stop_monitoring():
    """
    Stop background performance monitoring.
    
    Stops continuous monitoring of system resources.
    """
    try:
        performance_monitor.stop_monitoring()
        return {
            "status": "success",
            "message": "Performance monitoring stopped",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


@router.get("/dashboard")
async def get_performance_dashboard():
    """
    Get comprehensive performance dashboard data.
    
    Returns all performance metrics, queue status, and system health in one response.
    """
    try:
        # Get all performance data
        summary = performance_monitor.get_performance_summary()
        queue_status = queue_manager.get_queue_status()
        resource_usage = performance_monitor.get_resource_usage(minutes=30)
        
        # Calculate additional metrics
        recent_operations = performance_monitor.get_operation_statistics()
        
        # Determine system health status
        threshold_compliance = summary.get('threshold_compliance', {})
        health_score = sum(1 for ok in threshold_compliance.values() if ok) / len(threshold_compliance) if threshold_compliance else 1.0
        
        if health_score >= 0.8:
            health_status = "excellent"
        elif health_score >= 0.6:
            health_status = "good"
        elif health_score >= 0.4:
            health_status = "fair"
        else:
            health_status = "poor"
        
        dashboard_data = {
            "system_health": {
                "status": health_status,
                "score": health_score,
                "compliance": threshold_compliance
            },
            "performance_summary": summary,
            "queue_status": queue_status,
            "recent_resource_usage": resource_usage[-10:] if resource_usage else [],  # Last 10 data points
            "operation_statistics": recent_operations,
            "alerts": _generate_performance_alerts(summary, queue_status),
            "recommendations": _generate_performance_recommendations(summary, queue_status)
        }
        
        return {
            "status": "success",
            "data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance dashboard: {str(e)}")


def _generate_performance_alerts(summary: Dict[str, Any], queue_status: Dict[str, Any]) -> list:
    """Generate performance alerts based on current metrics."""
    alerts = []
    
    # Check threshold compliance
    threshold_compliance = summary.get('threshold_compliance', {})
    current_resources = summary.get('current_resources')
    
    if not threshold_compliance.get('cpu_usage_ok', True):
        cpu_percent = current_resources.get('cpu_percent', 0) if current_resources else 0
        alerts.append({
            "level": "warning",
            "type": "high_cpu_usage",
            "message": f"High CPU usage detected: {cpu_percent:.1f}%",
            "recommendation": "Consider scaling up resources or optimizing processing"
        })
    
    if not threshold_compliance.get('memory_usage_ok', True):
        memory_percent = current_resources.get('memory_percent', 0) if current_resources else 0
        alerts.append({
            "level": "warning",
            "type": "high_memory_usage",
            "message": f"High memory usage detected: {memory_percent:.1f}%",
            "recommendation": "Monitor for memory leaks or consider increasing available memory"
        })
    
    if not threshold_compliance.get('queue_size_ok', True):
        queue_size = queue_status.get('total_queue_size', 0)
        alerts.append({
            "level": "warning",
            "type": "large_queue",
            "message": f"Large queue size detected: {queue_size} tasks",
            "recommendation": "Consider adding more processing capacity or implementing request throttling"
        })
    
    if not threshold_compliance.get('success_rate_ok', True):
        success_rate = summary.get('overall_success_rate', 1.0) * 100
        alerts.append({
            "level": "error",
            "type": "low_success_rate",
            "message": f"Low success rate detected: {success_rate:.1f}%",
            "recommendation": "Investigate recent failures and check system logs"
        })
    
    return alerts


def _generate_performance_recommendations(summary: Dict[str, Any], queue_status: Dict[str, Any]) -> list:
    """Generate performance optimization recommendations."""
    recommendations = []
    
    current_resources = summary.get('current_resources')
    if not current_resources:
        return recommendations
    
    cpu_percent = current_resources.get('cpu_percent', 0)
    memory_percent = current_resources.get('memory_percent', 0)
    queue_size = queue_status.get('total_queue_size', 0)
    
    # CPU recommendations
    if cpu_percent > 70:
        recommendations.append({
            "category": "cpu",
            "priority": "high" if cpu_percent > 85 else "medium",
            "title": "Optimize CPU Usage",
            "description": "CPU usage is high. Consider optimizing algorithms or scaling horizontally.",
            "actions": [
                "Profile CPU-intensive operations",
                "Implement caching for repeated computations",
                "Consider using async processing for I/O operations"
            ]
        })
    
    # Memory recommendations
    if memory_percent > 70:
        recommendations.append({
            "category": "memory",
            "priority": "high" if memory_percent > 85 else "medium",
            "title": "Optimize Memory Usage",
            "description": "Memory usage is high. Monitor for leaks and optimize data structures.",
            "actions": [
                "Profile memory usage patterns",
                "Implement proper cleanup for temporary files",
                "Consider streaming processing for large files"
            ]
        })
    
    # Queue recommendations
    if queue_size > 10:
        recommendations.append({
            "category": "queue",
            "priority": "medium",
            "title": "Optimize Queue Processing",
            "description": "Queue size is growing. Consider optimizing processing speed or capacity.",
            "actions": [
                "Implement batch processing for similar requests",
                "Add more worker processes",
                "Implement request prioritization"
            ]
        })
    
    # General recommendations
    operation_stats = summary.get('operation_statistics', {})
    if operation_stats:
        for op_type, stats in operation_stats.items():
            avg_time = stats.get('avg_time', 0)
            if op_type == 'voice_analysis' and avg_time > 30:
                recommendations.append({
                    "category": "performance",
                    "priority": "medium",
                    "title": f"Optimize {op_type.replace('_', ' ').title()}",
                    "description": f"Average {op_type} time is {avg_time:.1f}s, which exceeds optimal range.",
                    "actions": [
                        "Profile the analysis pipeline",
                        "Consider model optimization",
                        "Implement result caching"
                    ]
                })
    
    return recommendations


@router.get("/health")
async def performance_health_check():
    """
    Check if system performance is within acceptable thresholds.
    
    Returns health status and any threshold violations.
    """
    try:
        summary = performance_monitor.get_performance_summary()
        threshold_compliance = summary.get('threshold_compliance', {})
        
        # Determine overall health
        all_ok = all(threshold_compliance.values())
        
        health_status = "healthy" if all_ok else "degraded"
        
        violations = [
            key for key, value in threshold_compliance.items()
            if not value
        ]
        
        return {
            "status": "success",
            "data": {
                "health_status": health_status,
                "threshold_compliance": threshold_compliance,
                "violations": violations,
                "current_resources": summary.get('current_resources'),
                "active_operations": summary.get('active_operations', 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check performance health: {str(e)}")
