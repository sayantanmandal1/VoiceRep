'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { apiClient } from '../lib/api';
import { ComprehensiveProgressBar, useProgressSteps, ProgressStep } from './ComprehensiveProgressBar';

interface FileUploadProps {
  onFileUploaded?: (fileData: any) => void;
  onError?: (error: string) => void;
}

interface UploadState {
  file: File | null;
  progress: number;
  status: 'idle' | 'uploading' | 'processing' | 'complete' | 'error';
  error?: string;
  fileData?: any;
  showProgress?: boolean;
}

const SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.m4a', '.mp4', '.avi', '.mov', '.mkv'];
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

export default function FileUpload({ onFileUploaded, onError }: FileUploadProps) {
  const [uploadState, setUploadState] = useState<UploadState>({
    file: null,
    progress: 0,
    status: 'idle'
  });

  // Progress steps for comprehensive tracking
  const uploadProgressSteps: Omit<ProgressStep, 'progress' | 'status'>[] = [
    {
      id: 'validation',
      name: 'File Validation',
      description: 'Validating file format, size, and structure',
      estimatedDuration: 1000
    },
    {
      id: 'preparation',
      name: 'Upload Preparation',
      description: 'Preparing file for secure upload',
      estimatedDuration: 500
    },
    {
      id: 'upload',
      name: 'File Upload',
      description: 'Uploading file to server',
      estimatedDuration: 5000
    },
    {
      id: 'processing',
      name: 'Server Processing',
      description: 'Extracting metadata and analyzing audio',
      estimatedDuration: 3000
    },
    {
      id: 'finalization',
      name: 'Finalization',
      description: 'Completing upload and preparing for analysis',
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
  } = useProgressSteps(uploadProgressSteps);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Reset progress steps
    resetSteps();
    
    // Start validation step
    startStep('validation');
    setUploadState(prev => ({ 
      ...prev, 
      file, 
      status: 'uploading', 
      progress: 0, 
      error: undefined,
      showProgress: true
    }));

    try {
      // Step 1: File Validation
      updateProgress('validation', 30);
      
      // Validate file size
      if (file.size > MAX_FILE_SIZE) {
        const error = `File size exceeds maximum limit of 100MB`;
        errorStep('validation', error);
        setUploadState(prev => ({ ...prev, status: 'error', error, showProgress: false }));
        onError?.(error);
        return;
      }

      updateProgress('validation', 60);

      // Validate file format
      const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!SUPPORTED_FORMATS.includes(fileExt)) {
        const error = `Unsupported file format. Supported formats: ${SUPPORTED_FORMATS.join(', ')}`;
        errorStep('validation', error);
        setUploadState(prev => ({ ...prev, status: 'error', error, showProgress: false }));
        onError?.(error);
        return;
      }

      updateProgress('validation', 100);
      completeStep('validation');

      // Step 2: Upload Preparation
      startStep('preparation');
      updateProgress('preparation', 50);
      
      // Small delay to show preparation step
      await new Promise(resolve => setTimeout(resolve, 300));
      
      updateProgress('preparation', 100);
      completeStep('preparation');

      // Step 3: File Upload
      startStep('upload');
      
      try {
        const response = await apiClient.uploadFile(file, (progress) => {
          updateProgress('upload', progress);
          setUploadState(prev => ({ ...prev, progress }));
        });

        completeStep('upload');

        // Step 4: Server Processing
        startStep('processing');
        updateProgress('processing', 20);

        // Poll for file status to track server processing
        let processingAttempts = 0;
        const maxProcessingAttempts = 30; // 15 seconds with 500ms intervals

        while (processingAttempts < maxProcessingAttempts) {
          try {
            const fileStatus = await apiClient.getFileStatus(response.id);
            
            if (fileStatus.status === 'ready') {
              updateProgress('processing', 100);
              completeStep('processing');
              break;
            } else if (fileStatus.status === 'failed') {
              throw new Error('Server processing failed');
            } else {
              // Update progress based on processing status
              const progressValue = Math.min(95, 20 + (processingAttempts / maxProcessingAttempts) * 75);
              updateProgress('processing', progressValue);
            }
          } catch (statusError) {
            // If status check fails, continue with estimated progress
            const progressValue = Math.min(95, 20 + (processingAttempts / maxProcessingAttempts) * 75);
            updateProgress('processing', progressValue);
          }

          await new Promise(resolve => setTimeout(resolve, 500));
          processingAttempts++;
        }

        // Complete processing step even if we couldn't get exact status
        if (processingAttempts >= maxProcessingAttempts) {
          updateProgress('processing', 100);
          completeStep('processing');
        }

        // Step 5: Finalization
        startStep('finalization');
        updateProgress('finalization', 50);
        
        // Small delay for finalization
        await new Promise(resolve => setTimeout(resolve, 500));
        
        updateProgress('finalization', 100);
        completeStep('finalization');

        setUploadState(prev => ({ 
          ...prev, 
          status: 'complete', 
          progress: 100,
          fileData: response,
          showProgress: false
        }));
        
        onFileUploaded?.(response);

      } catch (uploadError: any) {
        const errorMessage = uploadError.message || 'Upload failed';
        
        // Determine which step failed
        const activeStep = steps.find(step => step.status === 'active');
        if (activeStep) {
          errorStep(activeStep.id, errorMessage);
        }
        
        setUploadState(prev => ({ 
          ...prev, 
          status: 'error', 
          error: errorMessage,
          showProgress: false
        }));
        
        onError?.(errorMessage);
      }

    } catch (error: any) {
      const errorMessage = error.message || 'Validation failed';
      
      // Mark validation as failed if we're still in that step
      const activeStep = steps.find(step => step.status === 'active');
      if (activeStep) {
        errorStep(activeStep.id, errorMessage);
      }
      
      setUploadState(prev => ({ 
        ...prev, 
        status: 'error', 
        error: errorMessage,
        showProgress: false
      }));
      
      onError?.(errorMessage);
    }
  }, [onFileUploaded, onError, steps, startStep, updateProgress, completeStep, errorStep, resetSteps]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.flac', '.m4a'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv']
    },
    maxFiles: 1,
    maxSize: MAX_FILE_SIZE
  });

  const resetUpload = () => {
    setUploadState({
      file: null,
      progress: 0,
      status: 'idle',
      showProgress: false
    });
    resetSteps();
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="w-full max-w-md mx-auto">
      {/* Comprehensive Progress Bar */}
      {uploadState.showProgress && (
        <div className="mb-6">
          <ComprehensiveProgressBar 
            steps={steps}
            className="mb-4"
          />
        </div>
      )}

      {uploadState.status === 'idle' && (
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
            ${isDragActive 
              ? 'border-blue-400 bg-blue-50 dark:bg-blue-900/20' 
              : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
            }
          `}
        >
          <input {...getInputProps()} />
          <div className="space-y-4">
            <div className="text-4xl">🎵</div>
            <div>
              <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
                {isDragActive ? 'Drop your file here' : 'Upload audio or video file'}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                Drag & drop or click to select
              </p>
            </div>
            <div className="text-xs text-gray-400 dark:text-gray-500">
              <p>Supported formats: {SUPPORTED_FORMATS.join(', ')}</p>
              <p>Maximum size: 100MB</p>
            </div>
          </div>
        </div>
      )}

      {uploadState.status === 'uploading' && (
        <div className="space-y-4">
          <div className="flex items-center space-x-3">
            <div className="text-2xl">📁</div>
            <div className="flex-1">
              <p className="font-medium text-gray-900 dark:text-gray-100">
                {uploadState.file?.name}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {uploadState.file && formatFileSize(uploadState.file.size)}
              </p>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-600 dark:text-gray-400">Uploading...</span>
              <span className="text-gray-600 dark:text-gray-400">{uploadState.progress}%</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadState.progress}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {uploadState.status === 'complete' && uploadState.fileData && (
        <div className="space-y-4">
          <div className="flex items-center space-x-3">
            <div className="text-2xl">✅</div>
            <div className="flex-1">
              <p className="font-medium text-gray-900 dark:text-gray-100">
                Upload successful!
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {uploadState.fileData.filename}
              </p>
            </div>
          </div>
          
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 space-y-2">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-500 dark:text-gray-400">Size:</span>
                <span className="ml-2 text-gray-900 dark:text-gray-100">
                  {formatFileSize(uploadState.fileData.file_size)}
                </span>
              </div>
              {uploadState.fileData.duration && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Duration:</span>
                  <span className="ml-2 text-gray-900 dark:text-gray-100">
                    {formatDuration(uploadState.fileData.duration)}
                  </span>
                </div>
              )}
              {uploadState.fileData.sample_rate && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Sample Rate:</span>
                  <span className="ml-2 text-gray-900 dark:text-gray-100">
                    {uploadState.fileData.sample_rate} Hz
                  </span>
                </div>
              )}
              <div>
                <span className="text-gray-500 dark:text-gray-400">Status:</span>
                <span className="ml-2 text-green-600 dark:text-green-400">
                  {uploadState.fileData.status}
                </span>
              </div>
            </div>
          </div>
          
          <button
            onClick={resetUpload}
            className="w-full px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
          >
            Upload Another File
          </button>
        </div>
      )}

      {uploadState.status === 'error' && (
        <div className="space-y-4">
          <div className="flex items-center space-x-3">
            <div className="text-2xl">❌</div>
            <div className="flex-1">
              <p className="font-medium text-red-600 dark:text-red-400">
                Upload failed
              </p>
              <p className="text-sm text-red-500 dark:text-red-400">
                {uploadState.error}
              </p>
            </div>
          </div>
          
          <button
            onClick={resetUpload}
            className="w-full px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}