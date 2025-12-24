"use client";

import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface PerformanceMetrics {
  total_operations: number;
  overall_success_rate: number;
  active_operations: number;
  current_resources: {
    cpu_percent: number;
    memory_percent: number;
    memory_available_mb: number;
    active_tasks: number;
    queue_length: number;
  } | null;
  threshold_compliance: {
    success_rate_ok: boolean;
    cpu_usage_ok: boolean;
    memory_usage_ok: boolean;
    queue_size_ok: boolean;
  };
  monitoring_active: boolean;
}

interface QueueStatus {
  queue_sizes: {
    high: number;
    normal: number;
    low: number;
  };
  total_queue_size: number;
  estimated_wait_times: {
    high: number;
    normal: number;
    low: number;
  };
}

interface PerformanceMonitorProps {
  className?: string;
  refreshInterval?: number;
}

export default function PerformanceMonitor({ 
  className = "", 
  refreshInterval = 5000 
}: PerformanceMonitorProps) {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>('');

  const fetchMetrics = async () => {
    try {
      const [metricsResponse, queueResponse] = await Promise.all([
        axios.get('http://localhost:8000/api/v1/performance/metrics/summary'),
        axios.get('http://localhost:8000/api/v1/performance/queue/status')
      ]);

      setMetrics(metricsResponse.data.data);
      setQueueStatus(queueResponse.data.data);
      setError('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch performance metrics');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  const getStatusColor = (isOk: boolean) => {
    return isOk ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
  };

  const getStatusIcon = (isOk: boolean) => {
    return isOk ? '✓' : '⚠';
  };

  if (isLoading) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 ${className}`}>
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/4 mb-4"></div>
          <div className="space-y-3">
            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded"></div>
            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-4/6"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 ${className}`}>
        <div className="text-red-600 dark:text-red-400">
          <h3 className="font-semibold mb-2">Performance Monitor Error</h3>
          <p className="text-sm">{error}</p>
          <button 
            onClick={fetchMetrics}
            className="mt-2 px-3 py-1 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 rounded text-sm hover:bg-red-200 dark:hover:bg-red-800"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          System Performance
        </h3>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${metrics?.monitoring_active ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {metrics?.monitoring_active ? 'Monitoring' : 'Offline'}
          </span>
        </div>
      </div>

      {/* System Health */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">System Health</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Success Rate</span>
              <span className={`text-sm font-medium ${getStatusColor(metrics?.threshold_compliance.success_rate_ok || false)}`}>
                {getStatusIcon(metrics?.threshold_compliance.success_rate_ok || false)} {((metrics?.overall_success_rate || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">CPU Usage</span>
              <span className={`text-sm font-medium ${getStatusColor(metrics?.threshold_compliance.cpu_usage_ok || false)}`}>
                {getStatusIcon(metrics?.threshold_compliance.cpu_usage_ok || false)} {metrics?.current_resources?.cpu_percent?.toFixed(1) || 0}%
              </span>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Memory Usage</span>
              <span className={`text-sm font-medium ${getStatusColor(metrics?.threshold_compliance.memory_usage_ok || false)}`}>
                {getStatusIcon(metrics?.threshold_compliance.memory_usage_ok || false)} {metrics?.current_resources?.memory_percent?.toFixed(1) || 0}%
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Queue Size</span>
              <span className={`text-sm font-medium ${getStatusColor(metrics?.threshold_compliance.queue_size_ok || false)}`}>
                {getStatusIcon(metrics?.threshold_compliance.queue_size_ok || false)} {queueStatus?.total_queue_size || 0}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Current Activity */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Current Activity</h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {metrics?.active_operations || 0}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">Active Operations</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {metrics?.total_operations || 0}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">Total Processed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
              {queueStatus?.total_queue_size || 0}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400">Queued Tasks</div>
          </div>
        </div>
      </div>

      {/* Queue Details */}
      {queueStatus && queueStatus.total_queue_size > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Queue Status</h4>
          <div className="space-y-2">
            {Object.entries(queueStatus.queue_sizes).map(([priority, count]) => (
              <div key={priority} className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                  {priority} Priority
                </span>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {count} tasks
                  </span>
                  {count > 0 && (
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      (~{Math.round(queueStatus.estimated_wait_times[priority as keyof typeof queueStatus.estimated_wait_times])}s)
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Refresh indicator */}
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          <span>Auto-refresh every {refreshInterval / 1000}s</span>
          <button 
            onClick={fetchMetrics}
            className="hover:text-gray-700 dark:hover:text-gray-300"
          >
            Refresh now
          </button>
        </div>
      </div>
    </div>
  );
}