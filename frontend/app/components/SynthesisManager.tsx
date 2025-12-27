'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../lib/api';
import AudioPlayer from './AudioPlayer';
import { useNotifications } from './NotificationSystem';
import { ComprehensiveProgressBar, useProgressSteps, ProgressStep } from './ComprehensiveProgressBar';

// Local interfaces for this component
interface SynthesisRequest {
  text: string;
  voice_model_id: string;
  language: string;
  voice_settings?: {
    pitch_shift?: number;
    speed_factor?: number;
    emotion_intensity?: number;
    volume_gain?: number;
  };
  output_format?: 'wav' | 'mp3' | 'flac';
  quality?: 'standard' | 'high' | 'premium';
}

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

interface SynthesisProgress {
  task_id: string;
  progress: number;
  status: string;
  stage?: string;
  estimated_remaining?: number;
  quality_metrics?: {
    current_similarity?: number;
    confidence_score?: number;
    processing_stage?: string;
  };
  recommendations?: string[];
}

interface QualityMetrics {
  overall_similarity?: number;
  pitch_similarity?: number;
  timbre_similarity?: number;
  prosody_similarity?: number;
  spectral_similarity?: number;
  confidence_score?: number;
  quality_level?: string;
}

interface SynthesisResult {
  task_id: string;
  status: 'completed' | 'failed';
  output_url?: string;
  output_path?: string;
  metadata?: {
    duration?: number;
    sample_rate?: number;
    language?: string;
    quality_score?: number;
    processing_time?: number;
    quality_metrics?: QualityMetrics;
    recommendations?: string[];
    synthesis_method?: string;
    error_details?: {
      error_id?: string;
      error_type?: string;
      error_message?: string;
      error_category?: string;
      is_retryable?: boolean;
      recovery_suggestions?: string[];
    };
  };
  error_message?: string;
  processing_time?: number;
  created_at: string;
  completed_at?: string;
}

export default function SynthesisManager({ 
  uploadedFile, 
  validatedText, 
  onError 
}: SynthesisManagerProps) {
  const [synthesisTask, setSynthesisTask] = useState<SynthesisTask | null>(null);
  const [synthesisResult, setSynthesisResult] = useState<SynthesisResult | null>(null);
  const [progress, setProgress] = useState<SynthesisProgress | null>(null);
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics | null>(null);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string>('');
  const [showProgress, setShowProgress] = useState(false);
  
  // Regeneration state
  const [showRegenerationPanel, setShowRegenerationPanel] = useState(false);
  const [customText, setCustomText] = useState('');
  const [voiceSettings, setVoiceSettings] = useState({
    pitch_shift: 0.0,
    speed_factor: 1.0,
    emotion_intensity: 1.0,
    volume_gain: 0.0
  });
  const [selectedLanguage, setSelectedLanguage] = useState('english');
  
  const { addNotification } = useNotifications();

  // Progress steps for comprehensive synthesis tracking
  const synthesisProgressSteps: Omit<ProgressStep, 'progress' | 'status'>[] = [
    {
      id: 'initialization',
      name: 'Synthesis Initialization',
      description: 'Preparing voice model and text for synthesis',
      estimatedDuration: 2000
    },
    {
      id: 'voice_analysis',
      name: 'Voice Analysis',
      description: 'Analyzing reference voice characteristics',
      estimatedDuration: 5000
    },
    {
      id: 'text_processing',
      name: 'Text Processing',
      description: 'Processing and normalizing input text',
      estimatedDuration: 1500
    },
    {
      id: 'model_loading',
      name: 'Model Loading',
      description: 'Loading and preparing synthesis models',
      estimatedDuration: 3000
    },
    {
      id: 'synthesis',
      name: 'Voice Synthesis',
      description: 'Generating synthetic speech with cloned voice',
      estimatedDuration: 8000
    },
    {
      id: 'post_processing',
      name: 'Audio Post-Processing',
      description: 'Enhancing and finalizing audio output',
      estimatedDuration: 2000
    },
    {
      id: 'finalization',
      name: 'Finalization',
      description: 'Preparing audio for download and playback',
      estimatedDuration: 1000
    }
  ];

  const {
    steps,
    startStep,
    updateProgress,
    completeStep,
    errorStep,
    resetSteps
  } = useProgressSteps(synthesisProgressSteps);

  // Cleanup function to stop all audio atomically
  const stopAllAudio = useCallback(async (): Promise<void> => {
    return new Promise((resolve) => {
      const existingAudio = document.querySelectorAll('audio');
      let stoppedCount = 0;
      const totalAudio = existingAudio.length;
      
      if (totalAudio === 0) {
        resolve();
        return;
      }
      
      existingAudio.forEach(audio => {
        audio.pause();
        audio.currentTime = 0;
        stoppedCount++;
        if (stoppedCount === totalAudio) {
          resolve();
        }
      });
      
      // Fallback timeout to prevent hanging
      setTimeout(resolve, 100);
    });
  }, []);

  // Start synthesis using the new API client
  const startSynthesis = useCallback(async (useCustomSettings = false) => {
    if (!uploadedFile || !validatedText) {
      setError('Missing required data for synthesis');
      return;
    }

    try {
      // Atomically stop any currently playing audio before starting new synthesis
      await stopAllAudio();

      setIsProcessing(true);
      setError('');
      setSynthesisResult(null);
      setProgress(null);
      setShowProgress(true);
      resetSteps();

      // Step 1: Initialization
      startStep('initialization');
      updateProgress('initialization', 30);

      // Determine text and settings to use
      const textToUse = useCustomSettings && customText.trim() 
        ? customText.trim() 
        : (validatedText.sanitized_text || validatedText.text);
      
      const languageToUse = useCustomSettings 
        ? selectedLanguage 
        : (validatedText.detected_language || 'english');
      
      const settingsToUse = useCustomSettings 
        ? voiceSettings 
        : {
            pitch_shift: 0.0,
            speed_factor: 1.0,
            emotion_intensity: 1.0,
            volume_gain: 0.0
          };

      // Create synthesis request
      const synthesisRequest: SynthesisRequest = {
        text: textToUse,
        voice_model_id: `voice_model_${uploadedFile.id}`,
        language: languageToUse,
        voice_settings: settingsToUse,
        output_format: 'wav',
        quality: 'high'
      };

      updateProgress('initialization', 70);
      
      // Small delay to show initialization
      await new Promise(resolve => setTimeout(resolve, 500));
      
      updateProgress('initialization', 100);
      completeStep('initialization');

      // Step 2: Voice Analysis
      startStep('voice_analysis');
      updateProgress('voice_analysis', 20);

      // Use the API client for synthesis
      const response = await apiClient.synthesizeSpeech(synthesisRequest);
      setSynthesisTask(response);

      updateProgress('voice_analysis', 100);
      completeStep('voice_analysis');

      addNotification({
        type: 'info',
        title: 'Synthesis Started',
        message: useCustomSettings 
          ? 'Voice synthesis with custom settings has begun.' 
          : 'Voice synthesis has begun. This may take a few moments.',
        duration: 4000
      });

      // Start polling for progress
      pollProgress(response.task_id);

    } catch (error: any) {
      console.error('Synthesis start error:', error);
      
      // Log detailed error information for debugging
      console.log('Synthesis error details:', {
        message: error.message,
        name: error.name,
        stack: error.stack,
        response: error.response,
        request: error.request
      });
      
      const errorMessage = error.message || 'Failed to start synthesis';
      
      // Mark current step as failed
      const activeStep = steps.find(step => step.status === 'active');
      if (activeStep) {
        errorStep(activeStep.id, errorMessage);
      }
      
      setError(errorMessage);
      onError?.(errorMessage);
      setIsProcessing(false);
      setShowProgress(false);
      
      addNotification({
        type: 'error',
        title: 'Synthesis Failed',
        message: errorMessage,
        duration: 6000
      });
    }
  }, [uploadedFile, validatedText, onError, addNotification, steps, startStep, updateProgress, completeStep, errorStep, resetSteps, customText, voiceSettings, selectedLanguage, stopAllAudio]);

  // Cleanup audio on component unmount
  useEffect(() => {
    return () => {
      // Atomically stop all audio when component unmounts
      stopAllAudio();
    };
  }, [stopAllAudio]);

  // Poll for synthesis progress using API client
  const pollProgress = useCallback(async (taskId: string) => {
    const maxAttempts = 300; // 5 minutes with 1-second intervals
    let attempts = 0;

    // Start text processing step
    startStep('text_processing');
    updateProgress('text_processing', 50);

    const poll = async () => {
      try {
        attempts++;

        // Get progress using API client
        const progressData = await apiClient.getSynthesisStatus(taskId);
        setProgress(progressData);
        
        // Update quality metrics and recommendations if available
        if (progressData.quality_metrics) {
          setQualityMetrics(progressData.quality_metrics);
        }
        if (progressData.recommendations) {
          setRecommendations(progressData.recommendations);
        }

        // Map progress to our detailed steps
        if (progressData.stage === 'queued') {
          if (steps.find(s => s.id === 'text_processing')?.status === 'active') {
            updateProgress('text_processing', 100);
            completeStep('text_processing');
            startStep('model_loading');
            updateProgress('model_loading', 30);
          }
        } else if (progressData.stage === 'processing') {
          // Map backend progress to our steps
          const backendProgress = progressData.progress || 0;
          
          if (backendProgress < 20) {
            // Still in model loading
            if (steps.find(s => s.id === 'text_processing')?.status === 'active') {
              updateProgress('text_processing', 100);
              completeStep('text_processing');
            }
            if (steps.find(s => s.id === 'model_loading')?.status !== 'active') {
              startStep('model_loading');
            }
            updateProgress('model_loading', Math.min(100, backendProgress * 5));
          } else if (backendProgress < 80) {
            // Main synthesis phase
            if (steps.find(s => s.id === 'model_loading')?.status === 'active') {
              updateProgress('model_loading', 100);
              completeStep('model_loading');
            }
            if (steps.find(s => s.id === 'synthesis')?.status !== 'active') {
              startStep('synthesis');
            }
            const synthesisProgress = ((backendProgress - 20) / 60) * 100;
            updateProgress('synthesis', Math.min(100, synthesisProgress));
          } else {
            // Post-processing phase
            if (steps.find(s => s.id === 'synthesis')?.status === 'active') {
              updateProgress('synthesis', 100);
              completeStep('synthesis');
            }
            if (steps.find(s => s.id === 'post_processing')?.status !== 'active') {
              startStep('post_processing');
            }
            const postProcessProgress = ((backendProgress - 80) / 20) * 100;
            updateProgress('post_processing', Math.min(100, postProcessProgress));
          }
        }

        // Check if completed
        if (progressData.stage === 'completed') {
          // Complete any remaining steps
          ['text_processing', 'model_loading', 'synthesis', 'post_processing'].forEach(stepId => {
            const step = steps.find(s => s.id === stepId);
            if (step && step.status !== 'completed') {
              if (step.status === 'active') {
                updateProgress(stepId, 100);
              }
              completeStep(stepId);
            }
          });

          // Start and complete finalization
          startStep('finalization');
          updateProgress('finalization', 50);
          
          // Get final result
          const result = await apiClient.getSynthesisResult(taskId);
          
          // Extract quality metrics from result
          if (result.metadata?.quality_metrics) {
            setQualityMetrics(result.metadata.quality_metrics);
          }
          if (result.metadata?.recommendations) {
            setRecommendations(result.metadata.recommendations);
          }
          
          updateProgress('finalization', 100);
          completeStep('finalization');
          
          setSynthesisResult(result);
          setIsProcessing(false);
          setShowProgress(false);
          
          // Enhanced notification with quality info
          const qualityScore = result.metadata?.quality_metrics?.overall_similarity;
          const qualityMessage = qualityScore 
            ? `Quality score: ${(qualityScore * 100).toFixed(1)}%`
            : 'Your voice has been successfully synthesized and is ready for playback.';
          
          addNotification({
            type: 'success',
            title: 'Synthesis Complete!',
            message: qualityMessage,
            duration: 5000
          });
          return;
        }

        // Check if failed
        if (progressData.stage === 'failed') {
          const activeStep = steps.find(step => step.status === 'active');
          if (activeStep) {
            errorStep(activeStep.id, progressData.status);
          }
          
          setError(progressData.status);
          onError?.(progressData.status);
          setIsProcessing(false);
          setShowProgress(false);
          
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
          
          const activeStep = steps.find(step => step.status === 'active');
          if (activeStep) {
            errorStep(activeStep.id, timeoutMessage);
          }
          
          setError(timeoutMessage);
          onError?.(timeoutMessage);
          setIsProcessing(false);
          setShowProgress(false);
          
          addNotification({
            type: 'error',
            title: 'Synthesis Timeout',
            message: 'The synthesis process took too long. Please try again.',
            duration: 6000
          });
        }

      } catch (error: any) {
        console.error('Error polling progress:', error);
        
        // Log detailed error information for debugging
        console.log('Error details:', {
          message: error.message,
          name: error.name,
          stack: error.stack,
          response: error.response,
          request: error.request
        });
        
        // Handle timeout errors specifically
        if (error.message?.includes('timeout') || error.code === 'ECONNABORTED') {
          console.warn('Request timeout, but synthesis may still be processing. Continuing to poll...');
          if (attempts < maxAttempts) {
            setTimeout(poll, 2000); // Wait a bit longer before retrying
            return;
          }
        }
        
        // If it's a 202 (still processing), continue polling
        if (error.message?.includes('202') && attempts < maxAttempts) {
          setTimeout(poll, 1000);
          return;
        }

        // Check if this might be a successful response that's being mishandled
        if (error.response && error.response.status >= 200 && error.response.status < 300) {
          console.warn('Received successful response but treated as error:', error.response);
          // Try to continue polling as this might be a parsing issue
          if (attempts < maxAttempts) {
            setTimeout(poll, 1000);
            return;
          }
        }

        const errorMessage = error.message || 'Failed to get synthesis status';
        
        const activeStep = steps.find(step => step.status === 'active');
        if (activeStep) {
          errorStep(activeStep.id, errorMessage);
        }
        
        setError(errorMessage);
        onError?.(errorMessage);
        setIsProcessing(false);
        setShowProgress(false);
        
        addNotification({
          type: 'error',
          title: 'Status Check Failed',
          message: errorMessage,
          duration: 6000
        });
      }
    };

    // Complete text processing and start polling
    setTimeout(() => {
      updateProgress('text_processing', 100);
      completeStep('text_processing');
      poll();
    }, 1000);
  }, [onError, addNotification, steps, startStep, updateProgress, completeStep, errorStep]);

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
      setShowProgress(false);
      resetSteps();
      
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
  }, [synthesisTask, addNotification, resetSteps]);

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

  // Auto-start synthesis when both file and text are ready - DISABLED for manual control
  // useEffect(() => {
  //   if (uploadedFile && validatedText && !synthesisTask && !isProcessing) {
  //     // Small delay to ensure UI is ready
  //     const timer = setTimeout(() => {
  //       startSynthesis();
  //     }, 500);
      
  //     return () => clearTimeout(timer);
  //   }
  // }, [uploadedFile, validatedText, synthesisTask, isProcessing, startSynthesis]);

  // Don't render if missing required data
  if (!uploadedFile || !validatedText) {
    return null;
  }

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 pt-8">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">
        Step 3: Voice Synthesis
      </h2>

      {/* Manual Synthesis Trigger Button */}
      {!isProcessing && !synthesisResult && (
        <div className="mb-6 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="text-center">
            <div className="mb-4">
              <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.617.816L4.5 13.5H2a1 1 0 01-1-1V7.5a1 1 0 011-1h2.5l3.883-3.316a1 1 0 011.617.816zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.983 5.983 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.984 3.984 0 00-1.172-2.828 1 1 0 010-1.415z" clipRule="evenodd" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                Ready to Generate Your Voice Clone
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6 max-w-md mx-auto">
                Your audio file and text have been processed. Click the button below to start the voice synthesis process and create your personalized synthetic voice.
              </p>
            </div>
            
            <button
              onClick={() => startSynthesis()}
              disabled={isProcessing}
              className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 disabled:cursor-not-allowed disabled:transform-none"
            >
              <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 14.142M9 9a3 3 0 000 6v-6a3 3 0 000-6zm0 0V7a2 2 0 012-2h4a2 2 0 012 2v2M9 9a3 3 0 000 6v-6a3 3 0 000-6z" />
              </svg>
              {isProcessing ? 'Generating Voice...' : 'Generate Synthetic Voice'}
            </button>
            
            <div className="mt-4 text-sm text-gray-500 dark:text-gray-400">
              <p>This process typically takes 30-60 seconds</p>
            </div>
          </div>
        </div>
      )}

      {/* Comprehensive Progress Bar */}
      {showProgress && (
        <div className="mb-6">
          <ComprehensiveProgressBar 
            steps={steps}
            className="mb-4"
          />
        </div>
      )}

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
              onClick={() => startSynthesis()}
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
            key={synthesisResult.task_id} // Force remount on new synthesis
            audioUrl={apiClient.getAudioStreamUrl(synthesisResult.task_id)}
            title="Synthesized Speech"
            onDownload={handleDownloadComplete}
            className="mb-6"
          />

          {/* Quality Metrics Display */}
          {qualityMetrics && (
            <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center">
                <svg className="w-5 h-5 mr-2 text-green-600 dark:text-green-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Voice Quality Analysis
              </h4>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {qualityMetrics.overall_similarity && (
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Overall Similarity</span>
                      <span className={`text-lg font-bold ${
                        qualityMetrics.overall_similarity >= 0.95 ? 'text-green-600 dark:text-green-400' :
                        qualityMetrics.overall_similarity >= 0.85 ? 'text-yellow-600 dark:text-yellow-400' :
                        'text-red-600 dark:text-red-400'
                      }`}>
                        {(qualityMetrics.overall_similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          qualityMetrics.overall_similarity >= 0.95 ? 'bg-green-500' :
                          qualityMetrics.overall_similarity >= 0.85 ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${qualityMetrics.overall_similarity * 100}%` }}
                      />
                    </div>
                  </div>
                )}
                
                {qualityMetrics.pitch_similarity && (
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Pitch Match</span>
                      <span className="text-lg font-bold text-blue-600 dark:text-blue-400">
                        {(qualityMetrics.pitch_similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full"
                        style={{ width: `${qualityMetrics.pitch_similarity * 100}%` }}
                      />
                    </div>
                  </div>
                )}
                
                {qualityMetrics.timbre_similarity && (
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Voice Timbre</span>
                      <span className="text-lg font-bold text-purple-600 dark:text-purple-400">
                        {(qualityMetrics.timbre_similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-purple-500 h-2 rounded-full"
                        style={{ width: `${qualityMetrics.timbre_similarity * 100}%` }}
                      />
                    </div>
                  </div>
                )}
                
                {qualityMetrics.prosody_similarity && (
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Speech Rhythm</span>
                      <span className="text-lg font-bold text-indigo-600 dark:text-indigo-400">
                        {(qualityMetrics.prosody_similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-indigo-500 h-2 rounded-full"
                        style={{ width: `${qualityMetrics.prosody_similarity * 100}%` }}
                      />
                    </div>
                  </div>
                )}
                
                {qualityMetrics.confidence_score && (
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Confidence</span>
                      <span className="text-lg font-bold text-teal-600 dark:text-teal-400">
                        {(qualityMetrics.confidence_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-teal-500 h-2 rounded-full"
                        style={{ width: `${qualityMetrics.confidence_score * 100}%` }}
                      />
                    </div>
                  </div>
                )}
                
                {qualityMetrics.quality_level && (
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Quality Level</span>
                      <span className={`text-lg font-bold capitalize ${
                        qualityMetrics.quality_level === 'excellent' ? 'text-green-600 dark:text-green-400' :
                        qualityMetrics.quality_level === 'good' ? 'text-blue-600 dark:text-blue-400' :
                        qualityMetrics.quality_level === 'acceptable' ? 'text-yellow-600 dark:text-yellow-400' :
                        'text-red-600 dark:text-red-400'
                      }`}>
                        {qualityMetrics.quality_level}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Recommendations Display */}
          {recommendations && recommendations.length > 0 && (
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center">
                <svg className="w-5 h-5 mr-2 text-yellow-600 dark:text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                Quality Improvement Recommendations
              </h4>
              
              <div className="space-y-3">
                {recommendations.map((recommendation, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 bg-white dark:bg-gray-800 rounded-lg border border-yellow-200 dark:border-yellow-700">
                    <div className="flex-shrink-0 w-6 h-6 bg-yellow-100 dark:bg-yellow-900 rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-xs font-medium text-yellow-800 dark:text-yellow-200">
                        {index + 1}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      {recommendation}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

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
                    {synthesisResult.metadata?.duration?.toFixed(1) || 'N/A'}s
                  </span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Sample Rate:</span>
                  <span className="ml-2 text-gray-900 dark:text-gray-100">
                    {synthesisResult.metadata?.sample_rate || 'N/A'} Hz
                  </span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Language:</span>
                  <span className="ml-2 text-gray-900 dark:text-gray-100 capitalize">
                    {synthesisResult.metadata?.language || 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Quality:</span>
                  <span className="ml-2 text-gray-900 dark:text-gray-100">
                    {synthesisResult.metadata?.quality_score ? Math.round(synthesisResult.metadata.quality_score * 100) : 'N/A'}%
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
          <div className="flex justify-center space-x-4">
            <button
              onClick={async () => {
                // Atomically stop any playing audio
                await stopAllAudio();
                
                setSynthesisTask(null);
                setSynthesisResult(null);
                setProgress(null);
                setError('');
                setShowProgress(false);
                setShowRegenerationPanel(false);
                resetSteps();
              }}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
            >
              Create New Synthesis
            </button>
            
            {/* Regenerate with Same Settings Button */}
            <button
              onClick={() => startSynthesis(false)}
              disabled={isProcessing}
              className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-medium rounded-lg transition-colors flex items-center space-x-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <span>Regenerate Voice</span>
            </button>
            
            {/* Advanced Regeneration Button */}
            <button
              onClick={() => setShowRegenerationPanel(!showRegenerationPanel)}
              className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white font-medium rounded-lg transition-colors flex items-center space-x-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
              </svg>
              <span>Advanced Options</span>
            </button>
          </div>

          {/* Advanced Regeneration Panel */}
          {showRegenerationPanel && (
            <div className="mt-6 p-6 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
              <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center">
                <svg className="w-5 h-5 mr-2 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
                </svg>
                Advanced Synthesis Options
              </h4>
              
              <div className="space-y-6">
                {/* Custom Text Input */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Custom Text (Optional)
                  </label>
                  <textarea
                    value={customText}
                    onChange={(e) => setCustomText(e.target.value)}
                    placeholder="Enter different text to synthesize with the same voice..."
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-purple-500 focus:border-purple-500 dark:bg-gray-700 dark:text-gray-100 resize-none"
                    rows={3}
                  />
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    Leave empty to use the original text. Maximum 500 characters recommended.
                  </p>
                </div>

                {/* Language Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Language
                  </label>
                  <select
                    value={selectedLanguage}
                    onChange={(e) => setSelectedLanguage(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-2 focus:ring-purple-500 focus:border-purple-500 dark:bg-gray-700 dark:text-gray-100"
                  >
                    <option value="english">English</option>
                    <option value="spanish">Spanish</option>
                    <option value="french">French</option>
                    <option value="german">German</option>
                    <option value="italian">Italian</option>
                    <option value="portuguese">Portuguese</option>
                    <option value="russian">Russian</option>
                    <option value="chinese">Chinese</option>
                    <option value="japanese">Japanese</option>
                    <option value="korean">Korean</option>
                  </select>
                </div>

                {/* Voice Settings */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Pitch Shift */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Pitch Shift: {voiceSettings.pitch_shift > 0 ? '+' : ''}{voiceSettings.pitch_shift.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      min="-0.5"
                      max="0.5"
                      step="0.1"
                      value={voiceSettings.pitch_shift}
                      onChange={(e) => setVoiceSettings(prev => ({
                        ...prev,
                        pitch_shift: parseFloat(e.target.value)
                      }))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                    />
                    <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                      <span>Lower</span>
                      <span>Normal</span>
                      <span>Higher</span>
                    </div>
                  </div>

                  {/* Speed Factor */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Speed: {voiceSettings.speed_factor.toFixed(1)}x
                    </label>
                    <input
                      type="range"
                      min="0.5"
                      max="2.0"
                      step="0.1"
                      value={voiceSettings.speed_factor}
                      onChange={(e) => setVoiceSettings(prev => ({
                        ...prev,
                        speed_factor: parseFloat(e.target.value)
                      }))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                    />
                    <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                      <span>0.5x</span>
                      <span>1.0x</span>
                      <span>2.0x</span>
                    </div>
                  </div>

                  {/* Emotion Intensity */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Emotion Intensity: {voiceSettings.emotion_intensity.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      min="0.0"
                      max="2.0"
                      step="0.1"
                      value={voiceSettings.emotion_intensity}
                      onChange={(e) => setVoiceSettings(prev => ({
                        ...prev,
                        emotion_intensity: parseFloat(e.target.value)
                      }))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                    />
                    <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                      <span>Flat</span>
                      <span>Normal</span>
                      <span>Expressive</span>
                    </div>
                  </div>

                  {/* Volume Gain */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Volume: {voiceSettings.volume_gain > 0 ? '+' : ''}{voiceSettings.volume_gain.toFixed(1)} dB
                    </label>
                    <input
                      type="range"
                      min="-10.0"
                      max="10.0"
                      step="1.0"
                      value={voiceSettings.volume_gain}
                      onChange={(e) => setVoiceSettings(prev => ({
                        ...prev,
                        volume_gain: parseFloat(e.target.value)
                      }))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                    />
                    <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                      <span>-10dB</span>
                      <span>0dB</span>
                      <span>+10dB</span>
                    </div>
                  </div>
                </div>

                {/* Reset and Generate Buttons */}
                <div className="flex justify-between items-center pt-4 border-t border-gray-200 dark:border-gray-600">
                  <button
                    onClick={() => {
                      setCustomText('');
                      setVoiceSettings({
                        pitch_shift: 0.0,
                        speed_factor: 1.0,
                        emotion_intensity: 1.0,
                        volume_gain: 0.0
                      });
                      setSelectedLanguage(validatedText?.detected_language || 'english');
                    }}
                    className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-transparent border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                  >
                    Reset to Defaults
                  </button>
                  
                  <button
                    onClick={() => {
                      setShowRegenerationPanel(false);
                      startSynthesis(true);
                    }}
                    disabled={isProcessing}
                    className="px-6 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white font-medium rounded-md transition-colors flex items-center space-x-2"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 14.142M9 9a3 3 0 000 6v-6a3 3 0 000-6zm0 0V7a2 2 0 012-2h4a2 2 0 012 2v2M9 9a3 3 0 000 6v-6a3 3 0 000-6z" />
                    </svg>
                    <span>Generate with Custom Settings</span>
                  </button>
                </div>
              </div>
            </div>
          )}
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
          
          {/* Error Details */}
          {synthesisResult.metadata?.error_details && (
            <div className="mb-4 p-4 bg-red-100 dark:bg-red-900/30 rounded-lg">
              <h4 className="font-medium text-red-900 dark:text-red-100 mb-2">Error Details</h4>
              <div className="text-sm text-red-800 dark:text-red-200 space-y-1">
                {synthesisResult.metadata.error_details.error_id && (
                  <p><span className="font-medium">Error ID:</span> {synthesisResult.metadata.error_details.error_id}</p>
                )}
                {synthesisResult.metadata.error_details.error_category && (
                  <p><span className="font-medium">Category:</span> {synthesisResult.metadata.error_details.error_category}</p>
                )}
                {synthesisResult.metadata.error_details.is_retryable !== undefined && (
                  <p><span className="font-medium">Retryable:</span> {synthesisResult.metadata.error_details.is_retryable ? 'Yes' : 'No'}</p>
                )}
              </div>
            </div>
          )}
          
          {/* Recovery Suggestions */}
          {synthesisResult.metadata?.error_details?.recovery_suggestions && (
            <div className="mb-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded-lg">
              <h4 className="font-medium text-yellow-900 dark:text-yellow-100 mb-2 flex items-center">
                <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                Recovery Suggestions
              </h4>
              <ul className="text-sm text-yellow-800 dark:text-yellow-200 space-y-1">
                {synthesisResult.metadata.error_details.recovery_suggestions.map((suggestion: string, index: number) => (
                  <li key={index} className="flex items-start">
                    <span className="w-2 h-2 bg-yellow-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                    {suggestion}
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          <div className="flex space-x-3">
            {synthesisResult.metadata?.error_details?.is_retryable && (
              <button
                onClick={() => startSynthesis()}
                className="px-4 py-2 text-sm font-medium text-red-700 dark:text-red-400 bg-transparent border border-red-300 dark:border-red-600 rounded-md hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
              >
                Try Again
              </button>
            )}
            <button
              onClick={async () => {
                // Atomically stop any playing audio
                await stopAllAudio();
                
                setSynthesisTask(null);
                setSynthesisResult(null);
                setProgress(null);
                setError('');
                setShowProgress(false);
                resetSteps();
              }}
              className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-400 bg-transparent border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-900/30 transition-colors"
            >
              Start Over
            </button>
          </div>
        </div>
      )}
    </div>
  );
}