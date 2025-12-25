'use client';

import React, { useState, useCallback } from 'react';
import { ProgressTracker, VOICE_CLONING_STEPS, ProgressState } from '../lib/progress';
import ProgressBar from './ProgressBar';
import FileUpload from './FileUpload';
import TextInput from './TextInput';
import AudioPlayer from './AudioPlayer';
import { apiClient, SynthesisResult } from '../lib/api';

interface VoiceCloningWorkflowProps {
  className?: string;
}

export default function VoiceCloningWorkflow({ className = '' }: VoiceCloningWorkflowProps) {
  const [progressTracker] = useState(() => new ProgressTracker(VOICE_CLONING_STEPS));
  const [progressState, setProgressState] = useState<ProgressState>(progressTracker.getState());
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedFileData, setUploadedFileData] = useState<any>(null);
  const [text, setText] = useState('');
  const [isTextValid, setIsTextValid] = useState(false);
  const [synthesisResult, setSynthesisResult] = useState<SynthesisResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Subscribe to progress updates
  React.useEffect(() => {
    const unsubscribe = progressTracker.subscribe(setProgressState);
    return unsubscribe;
  }, [progressTracker]);

  const handleFileUpload = useCallback((file: File) => {
    setUploadedFile(file);
  }, []);

  const handleTextValidated = useCallback((textData: any) => {
    setText(textData.sanitized_text || textData.text);
    setIsTextValid(textData.is_valid);
  }, []);

  const handleTextError = useCallback((error: string) => {
    setIsTextValid(false);
    console.error('Text validation error:', error);
  }, []);

  const startVoiceCloning = async () => {
    if (!uploadedFileData || !text.trim() || !isTextValid) {
      alert('Please upload a file and enter valid text to synthesize.');
      return;
    }

    setIsProcessing(true);
    setSynthesisResult(null);
    progressTracker.start();

    try {
      // Step 1: Initialize Session
      progressTracker.setCurrentStep('session-init', 0);
      await new Promise(resolve => setTimeout(resolve, 500)); // Simulate initialization
      
      let sessionInfo;
      try {
        sessionInfo = await apiClient.createSession();
        progressTracker.updateStepProgress('session-init', 100);
        progressTracker.completeStep('session-init');
      } catch (error) {
        progressTracker.failStep('session-init', `Failed to create session: ${error}`);
        throw error;
      }

      // Step 2: Upload File - Skip since file is already uploaded
      progressTracker.setCurrentStep('file-upload', 0);
      // File is already uploaded by FileUpload component
      const uploadResult = uploadedFileData;
      progressTracker.updateStepProgress('file-upload', 100);
      progressTracker.completeStep('file-upload');

      // Step 3: Validate File
      progressTracker.setCurrentStep('file-validation', 0);
      try {
        // Simulate validation process
        for (let i = 0; i <= 100; i += 20) {
          progressTracker.updateStepProgress('file-validation', i);
          await new Promise(resolve => setTimeout(resolve, 200));
        }
        
        const fileStatus = await apiClient.getFileStatus(uploadResult.id);
        progressTracker.completeStep('file-validation');
      } catch (error) {
        progressTracker.failStep('file-validation', `File validation failed: ${error}`);
        throw error;
      }

      // Step 4: Audio Extraction (if video file)
      // Check file type from uploaded file data
      const isVideoFile = uploadedFileData.filename?.toLowerCase().includes('.mp4') || 
                         uploadedFileData.filename?.toLowerCase().includes('.avi') || 
                         uploadedFileData.filename?.toLowerCase().includes('.mov') || 
                         uploadedFileData.filename?.toLowerCase().includes('.mkv');
      
      if (isVideoFile) {
        progressTracker.setCurrentStep('audio-extraction', 0);
        try {
          // Simulate audio extraction
          for (let i = 0; i <= 100; i += 10) {
            progressTracker.updateStepProgress('audio-extraction', i);
            await new Promise(resolve => setTimeout(resolve, 500));
          }
          progressTracker.completeStep('audio-extraction');
        } catch (error) {
          progressTracker.failStep('audio-extraction', `Audio extraction failed: ${error}`);
          throw error;
        }
      } else {
        // Skip audio extraction for audio files
        progressTracker.completeStep('audio-extraction');
      }

      // Step 5: Voice Analysis
      progressTracker.setCurrentStep('voice-analysis', 0);
      try {
        // Sub-step: Feature Extraction
        progressTracker.setCurrentStep('voice-analysis-features', 0);
        for (let i = 0; i <= 100; i += 5) {
          progressTracker.updateStepProgress('voice-analysis-features', i);
          await new Promise(resolve => setTimeout(resolve, 150));
        }
        progressTracker.completeStep('voice-analysis-features');

        // Sub-step: Prosody Analysis
        progressTracker.setCurrentStep('voice-analysis-prosody', 0);
        for (let i = 0; i <= 100; i += 8) {
          progressTracker.updateStepProgress('voice-analysis-prosody', i);
          await new Promise(resolve => setTimeout(resolve, 120));
        }
        progressTracker.completeStep('voice-analysis-prosody');

        // Sub-step: Model Generation
        progressTracker.setCurrentStep('voice-analysis-model', 0);
        const analysisResult = await apiClient.analyzeVoice(uploadResult.id);
        
        // Poll for analysis completion
        let attempts = 0;
        while (attempts < 30) {
          const status = await apiClient.getVoiceAnalysisStatus(analysisResult.voice_profile_id);
          const progress = Math.min(100, (attempts / 30) * 100);
          progressTracker.updateStepProgress('voice-analysis-model', progress);
          
          if (status.status === 'completed') {
            progressTracker.completeStep('voice-analysis-model');
            progressTracker.completeStep('voice-analysis');
            break;
          } else if (status.status === 'failed') {
            throw new Error('Voice analysis failed');
          }
          
          await new Promise(resolve => setTimeout(resolve, 1000));
          attempts++;
        }
        
        if (attempts >= 30) {
          throw new Error('Voice analysis timeout');
        }
      } catch (error) {
        progressTracker.failStep('voice-analysis', `Voice analysis failed: ${error}`);
        throw error;
      }

      // Step 6: Text Processing
      progressTracker.setCurrentStep('text-processing', 0);
      try {
        const textValidation = await apiClient.validateText(text);
        progressTracker.updateStepProgress('text-processing', 50);
        
        if (!textValidation.is_valid) {
          throw new Error('Text validation failed');
        }
        
        progressTracker.updateStepProgress('text-processing', 100);
        progressTracker.completeStep('text-processing');
      } catch (error) {
        progressTracker.failStep('text-processing', `Text processing failed: ${error}`);
        throw error;
      }

      // Step 7: Speech Synthesis
      progressTracker.setCurrentStep('speech-synthesis', 0);
      try {
        // Sub-step: Preparation
        progressTracker.setCurrentStep('synthesis-preparation', 0);
        await new Promise(resolve => setTimeout(resolve, 1000));
        progressTracker.updateStepProgress('synthesis-preparation', 100);
        progressTracker.completeStep('synthesis-preparation');

        // Sub-step: Generation
        progressTracker.setCurrentStep('synthesis-generation', 0);
        const synthesisRequest = {
          text: text,
          voice_model_id: uploadResult.id, // Using id as voice_model_id for now
          language: 'auto',
          output_format: 'wav' as const
        };
        
        const synthesisResponse = await apiClient.synthesizeSpeech(synthesisRequest);
        
        // Poll for synthesis completion
        let attempts = 0;
        while (attempts < 60) {
          const progress = await apiClient.getSynthesisStatus(synthesisResponse.task_id);
          const progressPercent = Math.min(100, (attempts / 60) * 100);
          progressTracker.updateStepProgress('synthesis-generation', progressPercent);
          
          if (progress.stage === 'completed') {
            progressTracker.completeStep('synthesis-generation');
            
            // Sub-step: Post-processing
            progressTracker.setCurrentStep('synthesis-postprocessing', 0);
            const result = await apiClient.getSynthesisResult(synthesisResponse.task_id);
            
            for (let i = 0; i <= 100; i += 25) {
              progressTracker.updateStepProgress('synthesis-postprocessing', i);
              await new Promise(resolve => setTimeout(resolve, 200));
            }
            
            progressTracker.completeStep('synthesis-postprocessing');
            progressTracker.completeStep('speech-synthesis');
            setSynthesisResult(result);
            break;
          } else if (progress.stage === 'failed') {
            throw new Error('Speech synthesis failed');
          }
          
          await new Promise(resolve => setTimeout(resolve, 1000));
          attempts++;
        }
        
        if (attempts >= 60) {
          throw new Error('Speech synthesis timeout');
        }
      } catch (error) {
        progressTracker.failStep('speech-synthesis', `Speech synthesis failed: ${error}`);
        throw error;
      }

      // Step 8: Finalization
      progressTracker.setCurrentStep('finalization', 0);
      try {
        // Simulate finalization
        for (let i = 0; i <= 100; i += 50) {
          progressTracker.updateStepProgress('finalization', i);
          await new Promise(resolve => setTimeout(resolve, 300));
        }
        progressTracker.completeStep('finalization');
        progressTracker.complete();
      } catch (error) {
        progressTracker.failStep('finalization', `Finalization failed: ${error}`);
        throw error;
      }

    } catch (error) {
      console.error('Voice cloning failed:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const resetWorkflow = () => {
    setUploadedFile(null);
    setUploadedFileData(null);
    setText('');
    setIsTextValid(false);
    setSynthesisResult(null);
    setIsProcessing(false);
    // Create new progress tracker
    const newTracker = new ProgressTracker(VOICE_CLONING_STEPS);
    setProgressState(newTracker.getState());
  };

  return (
    <div className={`max-w-4xl mx-auto p-6 space-y-8 ${className}`}>
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Voice Style Replication
        </h1>
        <p className="text-gray-600">
          Upload an audio or video file and enter text to clone the voice
        </p>
      </div>

      {/* Progress Bar - Always visible when processing */}
      {(isProcessing || progressState.status === 'completed' || progressState.status === 'failed') && (
        <ProgressBar progress={progressState} />
      )}

      {/* Input Section - Hidden during processing */}
      {!isProcessing && progressState.status !== 'completed' && (
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h2 className="text-xl font-semibold mb-4">1. Upload Reference Audio/Video</h2>
            <FileUpload onFileUploaded={(fileData) => {
              setUploadedFileData(fileData);
              // We'll need to track the original file for type checking
              // For now, we'll create a mock file object from the data
              const mockFile = new File([''], fileData.filename, { type: 'audio/wav' });
              setUploadedFile(mockFile);
            }} />
            {uploadedFileData && (
              <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-sm text-green-800">
                  ✓ File uploaded: {uploadedFileData.filename}
                </p>
              </div>
            )}
          </div>

          <div>
            <h2 className="text-xl font-semibold mb-4">2. Enter Text to Synthesize</h2>
            <TextInput
              onTextValidated={handleTextValidated}
              onError={handleTextError}
              placeholder="Enter the text you want to synthesize in the cloned voice..."
              maxLength={1000}
            />
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex justify-center space-x-4">
        {!isProcessing && progressState.status !== 'completed' && (
          <button
            onClick={startVoiceCloning}
            disabled={!uploadedFileData || !text.trim() || !isTextValid}
            className="px-8 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            Start Voice Cloning
          </button>
        )}

        {(progressState.status === 'completed' || progressState.status === 'failed') && (
          <button
            onClick={resetWorkflow}
            className="px-8 py-3 bg-gray-600 text-white rounded-lg font-medium hover:bg-gray-700 transition-colors"
          >
            Start New Cloning
          </button>
        )}
      </div>

      {/* Results Section */}
      {synthesisResult && progressState.status === 'completed' && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4">3. Your Cloned Voice</h2>
          <AudioPlayer
            audioUrl={apiClient.getDownloadUrl(synthesisResult.task_id)}
            title="Synthesized Speech"
            onDownload={() => {
              const url = apiClient.getDownloadUrl(synthesisResult.task_id);
              window.open(url, '_blank');
            }}
          />
        </div>
      )}
    </div>
  );
}