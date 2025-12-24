'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

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
}

const SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac', '.m4a', '.mp4', '.avi', '.mov', '.mkv'];
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

export default function FileUpload({ onFileUploaded, onError }: FileUploadProps) {
  const [uploadState, setUploadState] = useState<UploadState>({
    file: null,
    progress: 0,
    status: 'idle'
  });

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      const error = `File size exceeds maximum limit of 100MB`;
      setUploadState(prev => ({ ...prev, status: 'error', error }));
      onError?.(error);
      return;
    }

    // Validate file format
    const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!SUPPORTED_FORMATS.includes(fileExt)) {
      const error = `Unsupported file format. Supported formats: ${SUPPORTED_FORMATS.join(', ')}`;
      setUploadState(prev => ({ ...prev, status: 'error', error }));
      onError?.(error);
      return;
    }

    setUploadState(prev => ({ 
      ...prev, 
      file, 
      status: 'uploading', 
      progress: 0, 
      error: undefined 
    }));

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:8000/api/v1/files/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.total 
            ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
            : 0;
          setUploadState(prev => ({ ...prev, progress }));
        },
      });

      setUploadState(prev => ({ 
        ...prev, 
        status: 'complete', 
        progress: 100,
        fileData: response.data
      }));
      
      onFileUploaded?.(response.data);
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail?.message || 
                          error.response?.data?.detail || 
                          error.message || 
                          'Upload failed';
      
      setUploadState(prev => ({ 
        ...prev, 
        status: 'error', 
        error: errorMessage 
      }));
      
      onError?.(errorMessage);
    }
  }, [onFileUploaded, onError]);

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
      status: 'idle'
    });
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