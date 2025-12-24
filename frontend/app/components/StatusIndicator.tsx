'use client';

import React from 'react';

interface StatusIndicatorProps {
  status: 'idle' | 'processing' | 'completed' | 'error' | 'warning' | 'queued';
  message?: string;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
  animated?: boolean;
}

interface SystemStatusProps {
  services: Array<{
    name: string;
    status: 'online' | 'offline' | 'degraded' | 'maintenance';
    responseTime?: number;
    lastCheck?: Date;
  }>;
  className?: string;
}

const statusConfig = {
  idle: {
    color: 'text-gray-600 dark:text-gray-400',
    bgColor: 'bg-gray-100 dark:bg-gray-800',
    borderColor: 'border-gray-300 dark:border-gray-600',
    icon: (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
      </svg>
    ),
    label: 'Ready'
  },
  processing: {
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-50 dark:bg-blue-900/20',
    borderColor: 'border-blue-200 dark:border-blue-800',
    icon: (
      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
    ),
    label: 'Processing'
  },
  completed: {
    color: 'text-green-600 dark:text-green-400',
    bgColor: 'bg-green-50 dark:bg-green-900/20',
    borderColor: 'border-green-200 dark:border-green-800',
    icon: (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
      </svg>
    ),
    label: 'Completed'
  },
  error: {
    color: 'text-red-600 dark:text-red-400',
    bgColor: 'bg-red-50 dark:bg-red-900/20',
    borderColor: 'border-red-200 dark:border-red-800',
    icon: (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
      </svg>
    ),
    label: 'Error'
  },
  warning: {
    color: 'text-yellow-600 dark:text-yellow-400',
    bgColor: 'bg-yellow-50 dark:bg-yellow-900/20',
    borderColor: 'border-yellow-200 dark:border-yellow-800',
    icon: (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
      </svg>
    ),
    label: 'Warning'
  },
  queued: {
    color: 'text-purple-600 dark:text-purple-400',
    bgColor: 'bg-purple-50 dark:bg-purple-900/20',
    borderColor: 'border-purple-200 dark:border-purple-800',
    icon: (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
      </svg>
    ),
    label: 'Queued'
  }
};

const sizeClasses = {
  sm: {
    container: 'px-2 py-1 text-xs',
    icon: 'w-3 h-3'
  },
  md: {
    container: 'px-3 py-2 text-sm',
    icon: 'w-4 h-4'
  },
  lg: {
    container: 'px-4 py-3 text-base',
    icon: 'w-5 h-5'
  }
};

export default function StatusIndicator({ 
  status, 
  message, 
  className = "",
  size = 'md',
  showIcon = true,
  animated = true
}: StatusIndicatorProps) {
  const config = statusConfig[status];
  const sizeConfig = sizeClasses[size];

  return (
    <div className={`
      inline-flex items-center space-x-2 rounded-full border
      ${config.color} ${config.bgColor} ${config.borderColor}
      ${sizeConfig.container}
      ${animated ? 'transition-all duration-200' : ''}
      ${className}
    `}>
      {showIcon && (
        <div className={sizeConfig.icon}>
          {config.icon}
        </div>
      )}
      <span className="font-medium">
        {message || config.label}
      </span>
    </div>
  );
}

// Simple status dot for compact displays
export function StatusDot({ 
  status, 
  className = "",
  size = 'md'
}: { 
  status: 'online' | 'offline' | 'warning' | 'error';
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}) {
  const dotSizes = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4'
  };

  const dotColors = {
    online: 'bg-green-500',
    offline: 'bg-gray-400',
    warning: 'bg-yellow-500',
    error: 'bg-red-500'
  };

  return (
    <div className={`
      ${dotSizes[size]} ${dotColors[status]} rounded-full
      ${status === 'online' ? 'animate-pulse' : ''}
      ${className}
    `} />
  );
}

// System status overview
export function SystemStatus({ services, className = "" }: SystemStatusProps) {
  const getOverallStatus = () => {
    const offlineCount = services.filter(s => s.status === 'offline').length;
    const degradedCount = services.filter(s => s.status === 'degraded').length;
    
    if (offlineCount > 0) return 'error';
    if (degradedCount > 0) return 'warning';
    return 'completed';
  };

  const formatResponseTime = (time?: number) => {
    if (!time) return 'N/A';
    return `${time}ms`;
  };

  const formatLastCheck = (date?: Date) => {
    if (!date) return 'Never';
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          System Status
        </h3>
        <StatusIndicator status={getOverallStatus()} size="sm" />
      </div>

      <div className="space-y-3">
        {services.map((service, index) => (
          <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <div className="flex items-center space-x-3">
              <StatusDot 
                status={
                  service.status === 'online' ? 'online' :
                  service.status === 'offline' ? 'offline' :
                  service.status === 'degraded' ? 'warning' : 'error'
                } 
              />
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white">
                  {service.name}
                </h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                  {service.status}
                </p>
              </div>
            </div>
            
            <div className="text-right text-sm text-gray-600 dark:text-gray-400">
              <div>Response: {formatResponseTime(service.responseTime)}</div>
              <div>Checked: {formatLastCheck(service.lastCheck)}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Real-time status updates component
export function RealTimeStatus({ 
  isConnected, 
  lastUpdate, 
  className = "" 
}: { 
  isConnected: boolean;
  lastUpdate?: Date;
  className?: string;
}) {
  const formatLastUpdate = (date?: Date) => {
    if (!date) return 'Never';
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const seconds = Math.floor(diff / 1000);
    
    if (seconds < 10) return 'Just now';
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ago`;
  };

  return (
    <div className={`flex items-center space-x-2 text-sm ${className}`}>
      <StatusDot status={isConnected ? 'online' : 'offline'} size="sm" />
      <span className="text-gray-600 dark:text-gray-400">
        {isConnected ? 'Connected' : 'Disconnected'}
      </span>
      {lastUpdate && (
        <>
          <span className="text-gray-400 dark:text-gray-500">•</span>
          <span className="text-gray-500 dark:text-gray-400">
            Updated {formatLastUpdate(lastUpdate)}
          </span>
        </>
      )}
    </div>
  );
}