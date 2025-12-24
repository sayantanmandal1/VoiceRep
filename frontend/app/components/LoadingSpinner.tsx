'use client';

import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'blue' | 'green' | 'red' | 'yellow' | 'gray';
  className?: string;
  text?: string;
  fullScreen?: boolean;
}

const sizeClasses = {
  sm: 'w-4 h-4',
  md: 'w-6 h-6',
  lg: 'w-8 h-8',
  xl: 'w-12 h-12'
};

const colorClasses = {
  blue: 'border-blue-600',
  green: 'border-green-600',
  red: 'border-red-600',
  yellow: 'border-yellow-600',
  gray: 'border-gray-600'
};

export default function LoadingSpinner({ 
  size = 'md', 
  color = 'blue', 
  className = "",
  text,
  fullScreen = false
}: LoadingSpinnerProps) {
  const spinnerElement = (
    <div className={`animate-spin rounded-full border-2 border-t-transparent ${sizeClasses[size]} ${colorClasses[color]} ${className}`} />
  );

  if (fullScreen) {
    return (
      <div className="fixed inset-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-50">
        <div className="flex flex-col items-center space-y-4">
          <div className={`animate-spin rounded-full border-4 border-t-transparent ${colorClasses[color]} w-12 h-12`} />
          {text && (
            <p className="text-gray-700 dark:text-gray-300 font-medium">{text}</p>
          )}
        </div>
      </div>
    );
  }

  if (text) {
    return (
      <div className="flex items-center space-x-3">
        {spinnerElement}
        <span className="text-gray-700 dark:text-gray-300">{text}</span>
      </div>
    );
  }

  return spinnerElement;
}

// Specialized loading components
export function LoadingOverlay({ 
  isVisible, 
  text = "Loading...", 
  className = "" 
}: { 
  isVisible: boolean; 
  text?: string; 
  className?: string; 
}) {
  if (!isVisible) return null;

  return (
    <div className={`absolute inset-0 bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm flex items-center justify-center z-10 ${className}`}>
      <LoadingSpinner size="lg" text={text} />
    </div>
  );
}

export function LoadingButton({ 
  isLoading, 
  children, 
  loadingText = "Loading...",
  disabled,
  className = "",
  ...props 
}: { 
  isLoading: boolean; 
  children: React.ReactNode;
  loadingText?: string;
  disabled?: boolean;
  className?: string;
  [key: string]: any;
}) {
  return (
    <button
      disabled={isLoading || disabled}
      className={`flex items-center justify-center space-x-2 ${className}`}
      {...props}
    >
      {isLoading && <LoadingSpinner size="sm" />}
      <span>{isLoading ? loadingText : children}</span>
    </button>
  );
}

export function LoadingCard({ 
  title, 
  description, 
  className = "" 
}: { 
  title?: string; 
  description?: string; 
  className?: string; 
}) {
  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 ${className}`}>
      <div className="animate-pulse">
        {title && (
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/4 mb-4"></div>
        )}
        <div className="space-y-3">
          <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded"></div>
          <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-5/6"></div>
          {description && (
            <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-4/6"></div>
          )}
        </div>
      </div>
    </div>
  );
}