'use client';

import React from 'react';

interface ProgressBarProps {
  progress: number;
  className?: string;
  color?: 'blue' | 'green' | 'red' | 'yellow' | 'purple';
  size?: 'sm' | 'md' | 'lg';
  showPercentage?: boolean;
  animated?: boolean;
}

interface CircularProgressProps {
  progress: number;
  size?: number;
  strokeWidth?: number;
  color?: 'blue' | 'green' | 'red' | 'yellow' | 'purple';
  className?: string;
  showPercentage?: boolean;
}

interface StepProgressProps {
  steps: Array<{
    label: string;
    status: 'completed' | 'current' | 'pending';
    description?: string;
  }>;
  className?: string;
}

const colorClasses = {
  blue: {
    bg: 'bg-blue-600',
    text: 'text-blue-600',
    stroke: 'stroke-blue-600'
  },
  green: {
    bg: 'bg-green-600',
    text: 'text-green-600',
    stroke: 'stroke-green-600'
  },
  red: {
    bg: 'bg-red-600',
    text: 'text-red-600',
    stroke: 'stroke-red-600'
  },
  yellow: {
    bg: 'bg-yellow-600',
    text: 'text-yellow-600',
    stroke: 'stroke-yellow-600'
  },
  purple: {
    bg: 'bg-purple-600',
    text: 'text-purple-600',
    stroke: 'stroke-purple-600'
  }
};

const sizeClasses = {
  sm: 'h-1',
  md: 'h-2',
  lg: 'h-3'
};

export function ProgressBar({ 
  progress, 
  className = "", 
  color = 'blue',
  size = 'md',
  showPercentage = false,
  animated = true
}: ProgressBarProps) {
  const clampedProgress = Math.max(0, Math.min(100, progress));
  
  return (
    <div className={`w-full ${className}`}>
      <div className={`w-full bg-gray-200 dark:bg-gray-700 rounded-full ${sizeClasses[size]}`}>
        <div 
          className={`${colorClasses[color].bg} ${sizeClasses[size]} rounded-full transition-all duration-500 ${animated ? 'ease-out' : ''}`}
          style={{ width: `${clampedProgress}%` }}
        />
      </div>
      {showPercentage && (
        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mt-1">
          <span>{Math.round(clampedProgress)}%</span>
        </div>
      )}
    </div>
  );
}

export function CircularProgress({ 
  progress, 
  size = 120, 
  strokeWidth = 8, 
  color = 'blue',
  className = "",
  showPercentage = true
}: CircularProgressProps) {
  const clampedProgress = Math.max(0, Math.min(100, progress));
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (clampedProgress / 100) * circumference;

  return (
    <div className={`relative inline-flex items-center justify-center ${className}`}>
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
      >
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          className="text-gray-200 dark:text-gray-700"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={strokeDasharray}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className={`${colorClasses[color].stroke} transition-all duration-500 ease-out`}
        />
      </svg>
      {showPercentage && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={`text-lg font-semibold ${colorClasses[color].text}`}>
            {Math.round(clampedProgress)}%
          </span>
        </div>
      )}
    </div>
  );
}

export function StepProgress({ steps, className = "" }: StepProgressProps) {
  return (
    <div className={`w-full ${className}`}>
      <div className="flex items-center justify-between">
        {steps.map((step, index) => (
          <React.Fragment key={index}>
            <div className="flex flex-col items-center">
              {/* Step indicator */}
              <div className={`
                w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
                ${step.status === 'completed' 
                  ? 'bg-green-600 text-white' 
                  : step.status === 'current'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
                }
              `}>
                {step.status === 'completed' ? (
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <span>{index + 1}</span>
                )}
              </div>
              
              {/* Step label */}
              <div className="mt-2 text-center">
                <p className={`text-sm font-medium ${
                  step.status === 'completed' || step.status === 'current'
                    ? 'text-gray-900 dark:text-gray-100'
                    : 'text-gray-500 dark:text-gray-400'
                }`}>
                  {step.label}
                </p>
                {step.description && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {step.description}
                  </p>
                )}
              </div>
            </div>
            
            {/* Connector line */}
            {index < steps.length - 1 && (
              <div className={`flex-1 h-0.5 mx-4 ${
                steps[index + 1].status === 'completed' || 
                (step.status === 'completed' && steps[index + 1].status === 'current')
                  ? 'bg-green-600'
                  : 'bg-gray-200 dark:bg-gray-700'
              }`} />
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

// Animated progress indicator for indeterminate progress
export function IndeterminateProgress({ 
  className = "",
  color = 'blue'
}: { 
  className?: string;
  color?: 'blue' | 'green' | 'red' | 'yellow' | 'purple';
}) {
  return (
    <div className={`w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden ${className}`}>
      <div className={`h-full ${colorClasses[color].bg} rounded-full animate-pulse`} 
           style={{
             animation: 'indeterminate 2s ease-in-out infinite',
             width: '30%'
           }} />
      <style jsx>{`
        @keyframes indeterminate {
          0% {
            transform: translateX(-100%);
          }
          50% {
            transform: translateX(300%);
          }
          100% {
            transform: translateX(-100%);
          }
        }
      `}</style>
    </div>
  );
}

// Multi-stage progress with different colors for each stage
export function MultiStageProgress({ 
  stages, 
  className = "" 
}: { 
  stages: Array<{
    label: string;
    progress: number;
    color?: 'blue' | 'green' | 'red' | 'yellow' | 'purple';
  }>;
  className?: string;
}) {
  const totalProgress = stages.reduce((sum, stage) => sum + stage.progress, 0) / stages.length;

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex justify-between items-center">
        <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
          Overall Progress
        </h3>
        <span className="text-sm text-gray-600 dark:text-gray-400">
          {Math.round(totalProgress)}%
        </span>
      </div>
      
      <ProgressBar progress={totalProgress} showPercentage={false} />
      
      <div className="space-y-2">
        {stages.map((stage, index) => (
          <div key={index} className="flex items-center justify-between">
            <span className="text-sm text-gray-700 dark:text-gray-300">
              {stage.label}
            </span>
            <div className="flex items-center space-x-2">
              <div className="w-24">
                <ProgressBar 
                  progress={stage.progress} 
                  color={stage.color || 'blue'}
                  size="sm"
                />
              </div>
              <span className="text-xs text-gray-500 dark:text-gray-400 w-8 text-right">
                {Math.round(stage.progress)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}