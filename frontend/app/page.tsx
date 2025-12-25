'use client';

import { useState } from 'react';
import ResponsiveLayout from './components/ResponsiveLayout';
import NotificationSystem, { useNotifications } from './components/NotificationSystem';
import ErrorBoundary from './components/ErrorBoundary';
import FileUpload from './components/FileUpload';
import TextInput from './components/TextInput';
import SynthesisManager from './components/SynthesisManager';
import { StepProgress } from './components/ProgressVisualization';
import { ErrorDisplay } from './components/ErrorBoundary';

export default function Home() {
  const [uploadedFile, setUploadedFile] = useState<any>(null);
  const [validatedText, setValidatedText] = useState<any>(null);
  const [error, setError] = useState<string>('');
  const { addNotification } = useNotifications();

  const handleFileUploaded = (fileData: any) => {
    setUploadedFile(fileData);
    setError('');
    addNotification({
      type: 'success',
      title: 'File Uploaded Successfully',
      message: `${fileData.filename} is ready for voice analysis`,
      duration: 4000
    });
  };

  const handleTextValidated = (textData: any) => {
    setValidatedText(textData);
    setError('');
    addNotification({
      type: 'info',
      title: 'Text Validated',
      message: `Ready to synthesize ${textData.character_count} characters in ${textData.detected_language || 'detected language'}`,
      duration: 3000
    });
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    addNotification({
      type: 'error',
      title: 'Error Occurred',
      message: errorMessage,
      duration: 6000
    });
  };

  // Progress steps for the workflow
  const getProgressSteps = (): { label: string; status: "completed" | "pending" | "current"; description?: string }[] => [
    {
      label: 'Upload Audio',
      status: uploadedFile ? 'completed' : 'current',
      description: 'Upload reference voice file'
    },
    {
      label: 'Enter Text',
      status: validatedText ? 'completed' : uploadedFile ? 'current' : 'pending',
      description: 'Input text to synthesize'
    },
    {
      label: 'Voice Synthesis',
      status: (uploadedFile && validatedText) ? 'current' : 'pending',
      description: 'Generate cloned voice'
    }
  ];

  return (
    <ErrorBoundary>
      <ResponsiveLayout showSystemStatus={true} showPerformanceMonitor={false}>
        <NotificationSystem />
        
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Voice Style Replication
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
              Upload audio or video files to clone voices with high-fidelity synthesis. 
              Preserve timbre, pitch, prosody, and emotional tone.
            </p>
          </div>

          {/* Progress Indicator */}
          <div className="mb-8">
            <StepProgress steps={getProgressSteps()} />
          </div>

          {/* Main Content */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
            <div className="mb-8">
              <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
                Step 1: Upload Reference Audio
              </h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Upload an audio or video file containing the voice you want to replicate. 
                The system will analyze voice characteristics for synthesis.
              </p>
              
              <FileUpload 
                onFileUploaded={handleFileUploaded}
                onError={handleError}
              />
            </div>

            {/* Error Display */}
            {error && (
              <div className="mb-6">
                <ErrorDisplay 
                  error={error} 
                  onRetry={() => setError('')}
                />
              </div>
            )}

            {/* Text Input Step */}
            {uploadedFile && (
              <div className="border-t border-gray-200 dark:border-gray-700 pt-8">
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
                  Step 2: Enter Text to Synthesize
                </h2>
                <p className="text-gray-600 dark:text-gray-300 mb-6">
                  Enter the text you want to synthesize using the uploaded voice. 
                  The system supports Unicode text in any language with automatic language detection.
                </p>
                
                <TextInput 
                  onTextValidated={handleTextValidated}
                  onError={handleError}
                />
              </div>
            )}

            {/* Synthesis Manager */}
            <SynthesisManager 
              uploadedFile={uploadedFile}
              validatedText={validatedText}
              onError={handleError}
            />
          </div>

          {/* Features */}
          <div className="mt-12 grid md:grid-cols-3 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow">
              <div className="text-2xl mb-3">🎯</div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">High Fidelity</h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Preserves timbre, pitch, prosody, and emotional characteristics
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow">
              <div className="text-2xl mb-3">🌍</div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Multi-Language</h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Cross-language synthesis while maintaining voice identity
              </p>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow">
              <div className="text-2xl mb-3">⚡</div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Fast Processing</h3>
              <p className="text-gray-600 dark:text-gray-300 text-sm">
                Quick analysis and synthesis with real-time progress tracking
              </p>
            </div>
          </div>
        </div>
      </ResponsiveLayout>
    </ErrorBoundary>
  );
}