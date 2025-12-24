'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { apiClient, SynthesisRequest, SynthesisProgress, SynthesisResult } from '../lib/api';
import AudioPlayer from './AudioPlayer';
import { useNotifications } from './NotificationSystem';

interface SynthesisManagerProps {
  uploadedFile?: any;
  validatedText?: any;
  onError?: (error: string) => void;
}

interface SynthesisTask {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  message: string;
  estimated_completion?: string;
  queue_position?: number;
}

export default function SynthesisManager({ 
  uploadedFile, 
  validatedText, 
  onError 
}: SynthesisManagerProps) {
  const [synthesisTask, setSynthesisTask] = useState<SynthesisTask | null>(null);
  const [synthesisResult, setSynthesisResult] = useState<SynthesisResult | null>(null);
  const [progress, setProgress] = useState<SynthesisProgress | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string>('');
  const { addNotification } = useNotifications();

  // Start synthesis using the new API client
  const startSynthesis = useCallback(async () => {
    if (!uploadedFile || !validatedText) {
      setError('Missing required data for synthesis');
      return;
    }

    try {
      setIsProcessing(true);
      setError('');
      setSynthesisResult(null);
      setProgress(null);

      // Create synthesis request
      const synthesisRequest: SynthesisRequest = {
        text: validatedText.sanitized_text || validatedText.text,
        voice_model_id: `voice_model_${uploadedFile.id}`,
        language: validatedText.detected_language || 'english',
        voice_settings: {
          pitch_shift: 0.0,
          speed_factor: 1.0,
          emotion_intensity: 1.0,
          volume_gain: 0.0
        },
        output_format: 'wav',
        quality: 'high'
      };

      // Use the API client for synthesis
      const response = await apiClient.synthesizeSpeech(synthesisRequest);
      setSynthesisTask(response);

      addNotification({
        type: 'info',
        title: 'Synthesis Started',
        message: 'Voice synthesis has begun. This may take a few moments.',
        duration: 4000
      });

      // Start polling for progress
      pollProgress(response.task_id);

    } catch (error: any) {
      const errorMessage = error.message || 'Failed to start synthesis';
      setError(errorMessage);
      onError?.(errorMessage);
      setIsProcessing(false);
      
      addNotification({
        type: 'error',
        title: 'Synthesis Failed',
        message: errorMessage,
        duration: 6000
      });
    }
  }, [uploadedFile, validatedText, onError, addNotification]);

  // Poll for synthesis progress using API client
  const pollProgress = useCallback(async (taskId: string) => {
    const maxAttempts = 120; // 2 minutes with 1-second intervals
    let attempts = 0;

    const poll = async () => {
      try {
        attempts++;

        // Get progress using API client
        const progressData = await apiClient.getSynthesisStatus(taskId);
        setProgress(progressData);

        // Check if completed
        if (progressData.stage === 'completed') {
          // Get final result
          const result = await apiClient.getSynthesisResult(taskId);
          setSynthesisResult(result);
          setIsProcessing(false);
          
          addNotification({
            type: 'success',
            title: 'Synthesis Complete!',
            message: 'Your voice has been successfully synthesized and is ready for playback.',
            duration: 5000
          });
          return;
        }

        // Check if failed
        if (progressData.stage === 'failed') {
          setError(progressData.status);
          onError?.(progressData.status);
          setIsProcessing(false);
          
          addNotification({
            type: 'error',
            title: 'Synthesis Failed',
            message: progressData.status,
            duration: 6000
          });
          return;
        }

        // Continue polling if still processing and within limits
        if (attempts < maxAttempts && (progressData.stage === 'processing' || progressData.stage === 'queued')) {
          setTimeout(poll, 1000); // Poll every second
        } else if (attempts >= maxAttempts) {
          const timeoutMessage = 'Synthesis timeout - please try again';
          setError(timeoutMessage);
          onError?.(timeoutMessage);
          setIsProcessing(false);
          
          addNotification({
            type: 'error',
            title: 'Synthesis Timeout',
            message: 'The synthesis process took too long. Please try again.',
            duration: 6000
          });
        }

      } catch (error: any) {
        console.error('Error polling progress:', error);
        
        // If it's a 202 (still processing), continue polling
        if (error.message?.includes('202') && attempts < maxAttempts) {
          setTimeout(poll, 1000);
          return;
        }

        const errorMessage = error.message || 'Failed to get synthesis status';
        setError(errorMessage);
        onError?.(errorMessage);
        setIsProcessing(false);
        
        addNotification({
          type: 'error',
          title: 'Status Check Failed',
          message: errorMessage,
          duration: 6000
        });
      }
    };

    // Start polling
    setTimeout(poll, 1000); // Initial delay
  }, [onError, addNotification]);

  // Handle download completion
  const handleDownloadComplete = useCallback(() => {
    addNotification({
      type: 'success',
      title: 'Download Complete',
      message: 'Audio file has been downloaded successfully.',
      duration: 3000
    });
  }, [addNotification]);

  // Cancel synthesis
  const cancelSynthesis = useCallback(async () => {
    if (!synthesisTask) return;

    try {
      await apiClient.cancelSynthesis(synthesisTask.task_id);
      setSynthesisTask(null);
      setProgress(null);
      setIsProcessing(false);
      
      addNotification({
        type: 'info',
        title: 'Synthesis Cancelled',
        message: 'The synthesis task has been cancelled.',
        duration: 3000
      });
    } catch (error: any) {
      console.error('Failed to cancel synthesis:', error);
      addNotification({
        type: 'warning',
        title: 'Cancellation Failed',
        message: 'Could not cancel the synthesis task.',
        duration: 4000
      });
    }
  }, [synthesisTask, addNotification]);

  // Format estimated time
  const formatEstimatedTime = (isoString?: string) => {
    if (!isoString) return null;
    
    try {
      const estimatedTime = new Date(isoString);
      const now = new Date();
      const diffMs = estimatedTime.getTime() - now.getTime();
      const diffSeconds = Math.max(0, Math.floor(diffMs / 1000));
      
      if (diffSeconds < 60) {
        return `${diffSeconds}s`;
      } else {
        const minutes = Math.floor(diffSeconds / 60);
        const seconds = diffSeconds % 60;
        return `${minutes}m ${seconds}s`;
      }
    } catch {
      return null;
    }
  };

  // Auto-start synthesis when both file and text are ready
  useEffect(() => {
    if (uploadedFile && validatedText && !synthesisTask && !isProcessing) {
      // Small delay to ensure UI is ready
      const timer = setTimeout(() => {
        startSynthesis();
      }, 500);
      
      return () => clearTimeout(timer);
    }
  }, [uploadedFile, validatedText, synthesisTask, isProcessing, startSynthesis]);

  // Don't render if missing required data
  if (!uploadedFile || !validatedText) {
    return null;
  }

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 pt-8">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">
        Step 3: Voice Synthesis
      </h2>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5 text-red-600 dark:text-red-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <p className="text-red-700 dark:text-red-400">{error}</p>
          </div>
          <div className="mt-3 flex space-x-3">
            <button
              onClick={startSynthesis}
              className="px-4 py-2 text-sm font-medium text-red-700 dark:text-red-400 bg-transparent border border-red-300 dark:border-red-600 rounded-md hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
            >
              Try Again
            </button>
            <button
              onClick={() => setError('')}
              className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-400 bg-transparent border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-900/30 transition-colors"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Processing Status */}
      {isProcessing && (
        <div className="mb-6 p-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-600 border-t-transparent"></div>
              <div>
                <h3 className="font-medium text-blue-900 dark:text-blue-100">
                  Synthesizing Voice...
                </h3>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  {progress?.status || 'Preparing synthesis...'}
                </p>
              </div>
            </div>
            
            {/* Cancel Button */}
            <button
              onClick={cancelSynthesis}
              className="px-3 py-1 text-sm text-blue-700 dark:text-blue-300 hover:text-blue-900 dark:hover:text-blue-100 transition-colors"
            >
              Cancel
            </button>
          </div>

          {/* Progress Bar */}
          {progress && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-blue-700 dark:text-blue-300">
                <span>Progress: {progress.progress}%</span>
                {progress.stage && (
                  <span className="capitalize">{progress.stage.replace('_', ' ')}</span>
                )}
              </div>
              <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${progress.progress}%` }}
                />
              </div>
              {progress.estimated_remaining && (
                <p className="text-xs text-blue-600 dark:text-blue-400">
                  Estimated remaining: {Math.ceil(progress.estimated_remaining)}s
                </p>
              )}
            </div>
          )}

          {/* Task Info */}
          {synthesisTask && (
            <div className="mt-4 text-sm text-blue-700 dark:text-blue-300">
              <p>Task ID: {synthesisTask.task_id}</p>
              {synthesisTask.queue_position && (
                <p>Queue position: {synthesisTask.queue_position}</p>
              )}
              {synthesisTask.estimated_completion && (
                <p>
                  Estimated completion: {formatEstimatedTime(synthesisTask.estimated_completion)}
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Synthesis Result - Audio Player */}
      {synthesisResult && synthesisResult.status === 'completed' && synthesisResult.output_url && (
        <div className="space-y-6">
          <div className="flex items-center space-x-3 text-green-600 dark:text-green-400 mb-4">
            <div className="w-8 h-8 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <h3 className="font-medium">Synthesis Complete!</h3>
              <p className="text-sm">Your voice has been successfully cloned and synthesized.</p>
            </div>
          </div>

          {/* Audio Player */}
          <AudioPlayer
            audioUrl={apiClient.getAudioStreamUrl(synthesisResult.task_id)}
            title="Synthesized Speech"
            onDownload={handleDownloadComplete}
            className="mb-6"
          />

          {/* Synthesis Metadata */}
          {synthesisResult.metadata && (
            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-3">
                Synthesis Details
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Duration:</span>
                  <span className="ml-2 text-gray-900 dark:text-gray-100">
                    {synthesisResult.metadata.duration.toFixed(1)}s
                  </span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Sample Rate:</span>
                  <span className="ml-2 text-gray-900 dark:text-gray-100">
                    {synthesisResult.metadata.sample_rate} Hz
                  </span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Language:</span>
                  <span className="ml-2 text-gray-900 dark:text-gray-100 capitalize">
                    {synthesisResult.metadata.language}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Quality:</span>
                  <span className="ml-2 text-gray-900 dark:text-gray-100">
                    {Math.round(synthesisResult.metadata.quality_score * 100)}%
                  </span>
                </div>
                {synthesisResult.processing_time && (
                  <div>
                    <span className="text-gray-500 dark:text-gray-400">Processing Time:</span>
                    <span className="ml-2 text-gray-900 dark:text-gray-100">
                      {synthesisResult.processing_time.toFixed(1)}s
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* New Synthesis Button */}
          <div className="flex justify-center">
            <button
              onClick={() => {
                setSynthesisTask(null);
                setSynthesisResult(null);
                setProgress(null);
                setError('');
              }}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
            >
              Create New Synthesis
            </button>
          </div>
        </div>
      )}

      {/* Failed Result */}
      {synthesisResult && synthesisResult.status === 'failed' && (
        <div className="p-6 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-8 h-8 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center">
              <svg className="w-5 h-5 text-red-600 dark:text-red-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div>
              <h3 className="font-medium text-red-900 dark:text-red-100">
                Synthesis Failed
              </h3>
              <p className="text-sm text-red-700 dark:text-red-300">
                {synthesisResult.error_message || 'An unknown error occurred during synthesis.'}
              </p>
            </div>
          </div>
          
          <button
            onClick={startSynthesis}
            className="px-4 py-2 text-sm font-medium text-red-700 dark:text-red-400 bg-transparent border border-red-300 dark:border-red-600 rounded-md hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}