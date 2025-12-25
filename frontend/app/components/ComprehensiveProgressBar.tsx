'use client';

import React, { useState, useEffect } from 'react';

export interface ProgressStep {
  id: string;
  name: string;
  description: string;
  estimatedDuration: number; // in milliseconds
  status: 'pending' | 'active' | 'completed' | 'error';
  progress: number; // 0-100
  startTime?: number;
  endTime?: number;
  error?: string;
}

export interface ComprehensiveProgressProps {
  steps: ProgressStep[];
  onStepUpdate?: (stepId: string, update: Partial<ProgressStep>) => void;
  onComplete?: () => void;
  onError?: (error: string) => void;
  className?: string;
}

export const ComprehensiveProgressBar: React.FC<ComprehensiveProgressProps> = ({
  steps,
  onStepUpdate,
  onComplete,
  onError,
  className = ''
}) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [overallProgress, setOverallProgress] = useState(0);
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [startTime] = useState(Date.now());

  // Calculate overall progress
  useEffect(() => {
    const totalSteps = steps.length;
    const completedSteps = steps.filter(step => step.status === 'completed').length;
    const activeStep = steps.find(step => step.status === 'active');
    
    let progress = (completedSteps / totalSteps) * 100;
    
    if (activeStep) {
      progress += (activeStep.progress / 100) * (1 / totalSteps) * 100;
    }
    
    setOverallProgress(Math.min(progress, 100));
  }, [steps]);

  // Calculate time estimates
  useEffect(() => {
    const now = Date.now();
    setElapsedTime(now - startTime);

    const activeStepIndex = steps.findIndex(step => step.status === 'active');
    if (activeStepIndex === -1) return;

    const activeStep = steps[activeStepIndex];
    const remainingSteps = steps.slice(activeStepIndex);
    
    // Calculate remaining time for current step
    const currentStepRemaining = activeStep.estimatedDuration * (1 - activeStep.progress / 100);
    
    // Calculate time for pending steps
    const pendingStepsTime = remainingSteps
      .slice(1)
      .filter(step => step.status === 'pending')
      .reduce((total, step) => total + step.estimatedDuration, 0);
    
    setEstimatedTimeRemaining(currentStepRemaining + pendingStepsTime);
  }, [steps, startTime]);

  // Format time duration
  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    if (ms < 60000) return `${Math.round(ms / 1000)}s`;
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.round((ms % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  };

  // Get status color
  const getStatusColor = (status: ProgressStep['status']): string => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'active': return 'text-blue-600 bg-blue-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-gray-500 bg-gray-100';
    }
  };

  // Get status icon
  const getStatusIcon = (status: ProgressStep['status']): string => {
    switch (status) {
      case 'completed': return '✓';
      case 'active': return '⟳';
      case 'error': return '✗';
      default: return '○';
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg ${className}`}>
      {/* Overall Progress Header */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Processing Progress
          </h3>
          <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
            {Math.round(overallProgress)}%
          </span>
        </div>
        
        {/* Overall Progress Bar */}
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-4">
          <div 
            className="bg-blue-500 to-purple-600 h-3 rounded-full transition-all duration-300 ease-out"
            style={{ 
              width: `${overallProgress}%`,
              background: 'linear-gradient(to right, #3b82f6, #9333ea)'
            }}
          />
        </div>

        {/* Time Information */}
        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-300">
          <span>Elapsed: {formatDuration(elapsedTime)}</span>
          <span>Remaining: ~{formatDuration(estimatedTimeRemaining)}</span>
        </div>
      </div>

      {/* Detailed Steps */}
      <div className="space-y-4">
        {steps.map((step, index) => (
          <div key={step.id} className="border-l-4 border-gray-200 dark:border-gray-700 pl-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-3">
                <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-medium ${getStatusColor(step.status)}`}>
                  {getStatusIcon(step.status)}
                </span>
                <div>
                  <h4 className="font-medium text-gray-900 dark:text-white">
                    {step.name}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    {step.description}
                  </p>
                </div>
              </div>
              
              {step.status === 'active' && (
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-900 dark:text-white">
                    {Math.round(step.progress)}%
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    ~{formatDuration(step.estimatedDuration * (1 - step.progress / 100))} left
                  </div>
                </div>
              )}
            </div>

            {/* Step Progress Bar */}
            {step.status === 'active' && (
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 mb-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300 ease-out"
                  style={{ width: `${step.progress}%` }}
                />
              </div>
            )}

            {/* Error Message */}
            {step.status === 'error' && step.error && (
              <div className="mt-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
                <p className="text-sm text-red-700 dark:text-red-300">
                  {step.error}
                </p>
              </div>
            )}

            {/* Completion Time */}
            {step.status === 'completed' && step.startTime && step.endTime && (
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Completed in {formatDuration(step.endTime - step.startTime)}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Current Process Details */}
      {(() => {
        const activeStep = steps.find(step => step.status === 'active');
        if (!activeStep) return null;

        return (
          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
            <div className="flex items-center space-x-2 mb-2">
              <div className="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full" />
              <span className="font-medium text-blue-900 dark:text-blue-100">
                Currently Processing: {activeStep.name}
              </span>
            </div>
            <p className="text-sm text-blue-700 dark:text-blue-300">
              {activeStep.description}
            </p>
            <div className="mt-2 text-xs text-blue-600 dark:text-blue-400">
              Progress: {Math.round(activeStep.progress)}% • 
              Estimated time remaining: ~{formatDuration(activeStep.estimatedDuration * (1 - activeStep.progress / 100))}
            </div>
          </div>
        );
      })()}
    </div>
  );
};

// Hook for managing progress steps
export const useProgressSteps = (initialSteps: Omit<ProgressStep, 'progress' | 'status'>[]) => {
  const [steps, setSteps] = useState<ProgressStep[]>(
    initialSteps.map(step => ({
      ...step,
      status: 'pending' as const,
      progress: 0
    }))
  );

  const updateStep = (stepId: string, update: Partial<ProgressStep>) => {
    setSteps(prev => prev.map(step => 
      step.id === stepId 
        ? { ...step, ...update }
        : step
    ));
  };

  const startStep = (stepId: string) => {
    updateStep(stepId, { 
      status: 'active', 
      progress: 0, 
      startTime: Date.now() 
    });
  };

  const updateProgress = (stepId: string, progress: number) => {
    updateStep(stepId, { progress: Math.min(Math.max(progress, 0), 100) });
  };

  const completeStep = (stepId: string) => {
    updateStep(stepId, { 
      status: 'completed', 
      progress: 100, 
      endTime: Date.now() 
    });
  };

  const errorStep = (stepId: string, error: string) => {
    updateStep(stepId, { 
      status: 'error', 
      error,
      endTime: Date.now() 
    });
  };

  const resetSteps = () => {
    setSteps(prev => prev.map(step => ({
      ...step,
      status: 'pending' as const,
      progress: 0,
      startTime: undefined,
      endTime: undefined,
      error: undefined
    })));
  };

  return {
    steps,
    updateStep,
    startStep,
    updateProgress,
    completeStep,
    errorStep,
    resetSteps
  };
};

export default ComprehensiveProgressBar;