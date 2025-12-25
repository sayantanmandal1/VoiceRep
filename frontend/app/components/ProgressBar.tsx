'use client';

import React from 'react';
import { ProgressState, formatTime, getProgressColor } from '../lib/progress';

interface ProgressBarProps {
  progress: ProgressState;
  className?: string;
}

export default function ProgressBar({ progress, className = '' }: ProgressBarProps) {
  const currentStep = progress.steps.find(step => step.id === progress.currentStep);
  const progressColor = getProgressColor(progress.overallProgress);

  return (
    <div className={`w-full max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg ${className}`}>
      {/* Header */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">
          Voice Cloning Progress
        </h3>
        <div className="flex justify-between items-center text-sm text-gray-600">
          <span>
            {progress.status === 'completed' ? 'Completed' : 
             progress.status === 'failed' ? 'Failed' :
             progress.status === 'running' ? 'Processing...' : 'Ready'}
          </span>
          <span>{Math.round(progress.overallProgress)}%</span>
        </div>
      </div>

      {/* Main Progress Bar */}
      <div className="mb-6">
        <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
          <div 
            className="h-full transition-all duration-300 ease-out rounded-full"
            style={{ 
              width: `${progress.overallProgress}%`,
              backgroundColor: progressColor
            }}
          >
            {/* Animated shine effect */}
            <div className="h-full w-full bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-pulse"></div>
          </div>
        </div>
        
        {/* Percentage indicator */}
        <div className="flex justify-between items-center mt-2 text-xs text-gray-500">
          <span>0%</span>
          <span className="font-medium text-gray-700">
            {Math.round(progress.overallProgress)}%
          </span>
          <span>100%</span>
        </div>
      </div>

      {/* Current Step Information */}
      {currentStep && progress.status === 'running' && (
        <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-medium text-blue-900">{currentStep.name}</h4>
            <span className="text-sm text-blue-700">
              {Math.round(progress.currentStepProgress)}%
            </span>
          </div>
          
          <p className="text-sm text-blue-800 mb-3">{currentStep.description}</p>
          
          {/* Current Step Progress Bar */}
          <div className="w-full bg-blue-200 rounded-full h-2">
            <div 
              className="h-full bg-blue-500 rounded-full transition-all duration-300"
              style={{ width: `${progress.currentStepProgress}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Time Information */}
      <div className="grid grid-cols-2 gap-4 mb-6 text-sm">
        <div className="text-center p-3 bg-gray-50 rounded-lg">
          <div className="font-medium text-gray-700">Elapsed Time</div>
          <div className="text-lg font-semibold text-gray-900">
            {formatTime(progress.elapsedTime)}
          </div>
        </div>
        
        <div className="text-center p-3 bg-gray-50 rounded-lg">
          <div className="font-medium text-gray-700">
            {progress.status === 'completed' ? 'Total Time' : 'Time Remaining'}
          </div>
          <div className="text-lg font-semibold text-gray-900">
            {progress.status === 'completed' 
              ? formatTime(progress.elapsedTime)
              : formatTime(progress.estimatedTimeRemaining)
            }
          </div>
        </div>
      </div>

      {/* Steps Overview */}
      <div className="space-y-2">
        <h4 className="font-medium text-gray-700 mb-3">Process Steps</h4>
        {progress.steps.map((step, index) => (
          <div key={step.id} className="flex items-center space-x-3">
            {/* Step Status Icon */}
            <div className="flex-shrink-0">
              {step.status === 'completed' ? (
                <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              ) : step.status === 'active' ? (
                <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                </div>
              ) : step.status === 'failed' ? (
                <div className="w-6 h-6 bg-red-500 rounded-full flex items-center justify-center">
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </div>
              ) : (
                <div className="w-6 h-6 bg-gray-300 rounded-full flex items-center justify-center">
                  <span className="text-xs text-gray-600">{index + 1}</span>
                </div>
              )}
            </div>

            {/* Step Information */}
            <div className="flex-grow">
              <div className="flex items-center justify-between">
                <span className={`text-sm font-medium ${
                  step.status === 'completed' ? 'text-green-700' :
                  step.status === 'active' ? 'text-blue-700' :
                  step.status === 'failed' ? 'text-red-700' :
                  'text-gray-600'
                }`}>
                  {step.name}
                </span>
                
                {step.status === 'active' && (
                  <span className="text-xs text-blue-600">
                    {Math.round(progress.currentStepProgress)}%
                  </span>
                )}
              </div>
              
              {step.status === 'active' && (
                <div className="mt-1">
                  <div className="w-full bg-gray-200 rounded-full h-1">
                    <div 
                      className="h-full bg-blue-500 rounded-full transition-all duration-300"
                      style={{ width: `${progress.currentStepProgress}%` }}
                    ></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Error Display */}
      {progress.status === 'failed' && progress.error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h4 className="font-medium text-red-800">Error Occurred</h4>
          </div>
          <p className="mt-2 text-sm text-red-700">{progress.error}</p>
        </div>
      )}

      {/* Success Message */}
      {progress.status === 'completed' && (
        <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h4 className="font-medium text-green-800">Voice Cloning Completed!</h4>
          </div>
          <p className="mt-2 text-sm text-green-700">
            Your voice has been successfully cloned and the speech has been synthesized.
          </p>
        </div>
      )}
    </div>
  );
}