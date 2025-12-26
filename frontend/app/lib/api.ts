/**
 * API client for Voice Style Replication backend
 * Provides centralized API communication with error handling and type safety
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_VERSION = '/api/v1';

// Types for API responses
export interface ApiResponse<T = any> {
  data: T;
  status: number;
  message?: string;
}

export interface ApiError {
  detail: string;
  status_code: number;
  type?: string;
}

// File Upload Types
export interface FileUploadResponse {
  id: string;
  filename: string;
  file_size: number;
  duration?: number;
  sample_rate?: number;
  status: 'uploaded' | 'extracting' | 'analyzing' | 'ready' | 'failed';
  upload_timestamp: string;
  processing_status?: string;
}

export interface FileValidationResponse {
  is_valid: boolean;
  file_type: string;
  duration?: number;
  sample_rate?: number;
  error_message?: string;
}

// Text Processing Types
export interface TextValidationResponse {
  is_valid: boolean;
  sanitized_text: string;
  character_count: number;
  detected_language: string;
  language_confidence: number;
  error_message?: string;
}

// Voice Analysis Types
export interface VoiceAnalysisResponse {
  voice_profile_id: string;
  voice_model_id: string;
  characteristics: {
    fundamental_frequency_range: { min: number; max: number; mean: number };
    formant_frequencies: number[];
    spectral_characteristics: { centroid: number; rolloff: number };
    prosody_parameters: { speech_rate: number; pause_frequency: number };
  };
  quality_score: number;
  status: 'analyzing' | 'completed' | 'failed';
  processing_time?: number;
}

// Synthesis Types
export interface SynthesisRequest {
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

export interface SynthesisResponse {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  message: string;
  estimated_completion?: string;
  queue_position?: number;
}

export interface SynthesisProgress {
  task_id: string;
  progress: number;
  status: string;
  stage?: 'queued' | 'processing' | 'completed' | 'failed';
  estimated_remaining?: number;
  quality_metrics?: {
    current_similarity?: number;
    confidence_score?: number;
    processing_stage?: string;
  };
  recommendations?: string[];
}

export interface SynthesisResult {
  task_id: string;
  status: 'completed' | 'failed';
  output_url?: string;
  output_path?: string;
  metadata?: {
    duration: number;
    sample_rate: number;
    language: string;
    quality_score: number;
    processing_time: number;
    quality_metrics?: {
      overall_similarity?: number;
      pitch_similarity?: number;
      timbre_similarity?: number;
      prosody_similarity?: number;
      spectral_similarity?: number;
      confidence_score?: number;
      quality_level?: string;
    };
    recommendations?: string[];
    synthesis_method?: string;
  };
  error_message?: string;
  processing_time?: number;
  created_at: string;
  completed_at?: string;
}

// Performance Types
export interface PerformanceMetrics {
  processing_time: number;
  queue_length: number;
  system_load: number;
  memory_usage: number;
  active_tasks: number;
}

// Session Types
export interface SessionInfo {
  id: string;
  session_token: string;
  user_identifier: string;
  is_active: boolean;
  expires_at: string;
  last_activity: string;
  data_namespace: string;
  created_at: string;
}

/**
 * API Client Class
 */
class VoiceReplicationAPI {
  private client: AxiosInstance;
  private sessionCreationPromise: Promise<SessionInfo> | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}${API_VERSION}`,
      timeout: 600000, // 10 minutes default timeout (matches backend TASK_TIMEOUT)
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for adding session info
    this.client.interceptors.request.use(
      async (config) => {
        // Add session token if available
        let sessionToken = this.getSessionToken();
        
        // If no session token, try to create a session (avoid concurrent creation)
        if (!sessionToken) {
          try {
            if (!this.sessionCreationPromise) {
              this.sessionCreationPromise = this.createSessionInternal();
            }
            const sessionResponse = await this.sessionCreationPromise;
            sessionToken = sessionResponse.session_token;
            this.sessionCreationPromise = null; // Reset for future use
          } catch (error) {
            console.warn('Failed to create session automatically:', error);
            this.sessionCreationPromise = null; // Reset on error
          }
        }
        
        if (sessionToken) {
          config.headers['X-Session-Token'] = sessionToken;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling and session token capture
    this.client.interceptors.response.use(
      (response) => {
        // Capture session token from response headers (try both cases)
        const sessionToken = response.headers['x-session-token'] || response.headers['X-Session-Token'];
        if (sessionToken) {
          this.setSessionToken(sessionToken);
        }
        return response;
      },
      (error: AxiosError<ApiError>) => {
        // Also capture session token from error responses (try both cases)
        const sessionToken = error.response?.headers['x-session-token'] || error.response?.headers['X-Session-Token'];
        if (sessionToken) {
          this.setSessionToken(sessionToken);
        }
        
        const apiError = this.handleApiError(error);
        return Promise.reject(apiError);
      }
    );
  }

  private getSessionToken(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('voice_replication_session_token');
    }
    return null;
  }

  private setSessionToken(sessionToken: string): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('voice_replication_session_token', sessionToken);
    }
  }

  private clearSessionToken(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('voice_replication_session_token');
    }
  }

  private handleApiError(error: AxiosError<ApiError>): Error {
    console.error('API Error Details:', {
      message: error.message,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      hasRequest: !!error.request,
      hasResponse: !!error.response
    });

    if (error.response) {
      const { data, status } = error.response;
      
      // Clear session token on authentication errors
      if (status === 401) {
        console.log('Authentication error - clearing session token');
        this.clearSessionToken();
      }
      
      const message = data?.detail || `HTTP ${status}: ${error.message}`;
      return new Error(message);
    } else if (error.request) {
      // Check if this is actually a network error or a parsing issue
      console.warn('Request made but no response received:', error.request);
      
      // If we have a request but no response, it might be a CORS or parsing issue
      // rather than the server being unreachable
      if (error.code === 'ECONNREFUSED' || error.code === 'NETWORK_ERROR') {
        return new Error('Network error: Unable to reach the server. Please check if the backend is running on http://localhost:8001');
      } else {
        // For other cases, provide a more specific error message
        return new Error(`Request failed: ${error.message}. The server may be processing your request.`);
      }
    } else {
      return new Error(`Request error: ${error.message}`);
    }
  }

  // Session Management
  private async createSessionInternal(): Promise<SessionInfo> {
    try {
      console.log('Creating session internally...');
      const response = await axios.post<SessionInfo>(`${API_BASE_URL}${API_VERSION}/session/create`, {});
      console.log('Session created successfully:', response.data);
      this.setSessionToken(response.data.session_token);
      return response.data;
    } catch (error) {
      console.error('Failed to create session internally:', error);
      throw error;
    }
  }

  async createSession(): Promise<SessionInfo> {
    try {
      console.log('Creating session via API client...');
      const response = await this.client.post<SessionInfo>('/session/create', {});
      console.log('Session created via API client:', response.data);
      this.setSessionToken(response.data.session_token);
      return response.data;
    } catch (error) {
      console.error('Failed to create session via API client:', error);
      throw error;
    }
  }

  async getSessionInfo(): Promise<SessionInfo> {
    const response = await this.client.get<SessionInfo>('/session/info');
    return response.data;
  }

  // File Operations
  async uploadFile(file: File, onProgress?: (progress: number) => void): Promise<FileUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await this.client.post<FileUploadResponse>('/files/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes for file upload
        onUploadProgress: (progressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onProgress(progress);
          }
        },
      });

      return response.data;
    } catch (error) {
      console.error('File upload error:', error);
      throw error;
    }
  }

  async validateFile(fileId: string): Promise<FileValidationResponse> {
    const response = await this.client.get<FileValidationResponse>(`/files/${fileId}/validate`);
    return response.data;
  }

  async getFileStatus(fileId: string): Promise<FileUploadResponse> {
    const response = await this.client.get<FileUploadResponse>(`/files/${fileId}/status`);
    return response.data;
  }

  // Text Processing
  async validateText(text: string, language?: string): Promise<TextValidationResponse> {
    const response = await this.client.post<TextValidationResponse>('/text/validate', {
      text,
      language,
    });
    return response.data;
  }

  async detectLanguage(text: string): Promise<{ language: string; confidence: number }> {
    const response = await this.client.post<{ language: string; confidence: number }>('/text/detect-language', {
      text,
    });
    return response.data;
  }

  // Voice Analysis
  async analyzeVoice(fileId: string): Promise<VoiceAnalysisResponse> {
    const response = await this.client.post<VoiceAnalysisResponse>('/voice/analyze', {
      reference_audio_id: fileId,
    });
    return response.data;
  }

  async getVoiceAnalysisStatus(voiceProfileId: string): Promise<VoiceAnalysisResponse> {
    const response = await this.client.get<VoiceAnalysisResponse>(`/voice/analysis/${voiceProfileId}/status`);
    return response.data;
  }

  // Speech Synthesis
  async synthesizeSpeech(request: SynthesisRequest): Promise<SynthesisResponse> {
    try {
      console.log('Starting synthesis request:', request);
      const response = await this.client.post<SynthesisResponse>('/synthesis/synthesize', request, {
        timeout: 600000, // 10 minutes timeout for synthesis requests (matches backend TASK_TIMEOUT)
      });
      console.log('Synthesis request successful:', response.data);
      return response.data;
    } catch (error) {
      console.error('Synthesis request failed:', error);
      throw error;
    }
  }

  async getSynthesisStatus(taskId: string): Promise<SynthesisProgress> {
    try {
      // Use longer timeout for synthesis status requests since synthesis can take time
      const response = await this.client.get<SynthesisProgress>(`/synthesis/status/${taskId}`, {
        timeout: 120000 // 2 minutes timeout for status checks
      });
      console.log(`Status for task ${taskId}:`, response.data);
      return response.data;
    } catch (error) {
      console.error(`Failed to get status for task ${taskId}:`, error);
      throw error;
    }
  }

  async getSynthesisResult(taskId: string): Promise<SynthesisResult> {
    // Use longer timeout for synthesis result requests
    const response = await this.client.get<SynthesisResult>(`/synthesis/result/${taskId}`, {
      timeout: 120000 // 2 minutes timeout
    });
    return response.data;
  }

  async cancelSynthesis(taskId: string): Promise<{ status: string; message: string }> {
    const response = await this.client.delete<{ status: string; message: string }>(`/synthesis/task/${taskId}`);
    return response.data;
  }

  // Cross-language Synthesis
  async synthesizeCrossLanguage(
    text: string,
    sourceVoiceModelId: string,
    targetLanguage: string
  ): Promise<SynthesisResponse> {
    const response = await this.client.post<SynthesisResponse>('/synthesis/synthesize/cross-language', {
      text,
      source_voice_model_id: sourceVoiceModelId,
      target_language: targetLanguage,
    });
    return response.data;
  }

  // Performance Monitoring
  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    const response = await this.client.get<PerformanceMetrics>('/performance/metrics');
    return response.data;
  }

  async getSystemStatus(): Promise<{ status: string; uptime: number; version: string }> {
    const response = await this.client.get<{ status: string; uptime: number; version: string }>('/performance/status');
    return response.data;
  }

  // Quality Metrics and Feedback
  async getQualityMetrics(taskId: string): Promise<{
    task_id: string;
    status: string;
    current_metrics: any;
    final_metrics: any;
    recommendations: string[];
    similarity_breakdown: any;
    confidence_scores: any;
    processing_info: any;
  }> {
    const response = await this.client.get(`/synthesis/quality/${taskId}`);
    return response.data;
  }

  async submitQualityFeedback(taskId: string, feedback: {
    rating?: number;
    issues?: string[];
    comments?: string;
    similarity_rating?: number;
  }): Promise<{
    task_id: string;
    feedback_received: boolean;
    improvement_suggestions: string[];
    next_steps: string[];
  }> {
    const response = await this.client.post(`/synthesis/feedback/${taskId}`, feedback);
    return response.data;
  }

  // Utility Methods
  getDownloadUrl(taskId: string, format: 'wav' | 'mp3' | 'flac' = 'wav'): string {
    return `${API_BASE_URL}${API_VERSION}/synthesis/download/${taskId}?format=${format}`;
  }

  getAudioStreamUrl(taskId: string): string {
    return `${API_BASE_URL}${API_VERSION}/synthesis/download/${taskId}`;
  }

  // Health Check with detailed logging
  async healthCheck(): Promise<{ status: string }> {
    try {
      console.log('Performing health check to:', `${API_BASE_URL}${API_VERSION}/`);
      const response = await this.client.get<{ status: string }>('/');
      console.log('Health check successful:', response.data);
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  // Test connectivity method
  async testConnectivity(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch (error) {
      console.error('Connectivity test failed:', error);
      return false;
    }
  }
}

// Create singleton instance
export const apiClient = new VoiceReplicationAPI();

// Export default instance
export default apiClient;

// Utility functions for common operations
export const uploadAndAnalyzeFile = async (
  file: File,
  onUploadProgress?: (progress: number) => void,
  onAnalysisProgress?: (status: string) => void
): Promise<VoiceAnalysisResponse> => {
  // Upload file
  const uploadResult = await apiClient.uploadFile(file, onUploadProgress);
  
  // Start voice analysis
  const analysisResult = await apiClient.analyzeVoice(uploadResult.id);
  
  // Poll for analysis completion
  let attempts = 0;
  const maxAttempts = 60; // 1 minute with 1-second intervals
  
  while (attempts < maxAttempts) {
    const status = await apiClient.getVoiceAnalysisStatus(analysisResult.voice_profile_id);
    
    if (onAnalysisProgress) {
      onAnalysisProgress(status.status);
    }
    
    if (status.status === 'completed') {
      return status;
    } else if (status.status === 'failed') {
      throw new Error('Voice analysis failed');
    }
    
    await new Promise(resolve => setTimeout(resolve, 1000));
    attempts++;
  }
  
  throw new Error('Voice analysis timeout');
};

export const synthesizeWithPolling = async (
  request: SynthesisRequest,
  onProgress?: (progress: SynthesisProgress) => void
): Promise<SynthesisResult> => {
  // Start synthesis
  const synthesisResponse = await apiClient.synthesizeSpeech(request);
  
  // Poll for completion
  let attempts = 0;
  const maxAttempts = 600; // 10 minutes with 1-second intervals (matches backend TASK_TIMEOUT)
  
  while (attempts < maxAttempts) {
    const progress = await apiClient.getSynthesisStatus(synthesisResponse.task_id);
    
    if (onProgress) {
      onProgress(progress);
    }
    
    if (progress.stage === 'completed') {
      return await apiClient.getSynthesisResult(synthesisResponse.task_id);
    } else if (progress.stage === 'failed') {
      const result = await apiClient.getSynthesisResult(synthesisResponse.task_id);
      throw new Error(result.error_message || 'Synthesis failed');
    }
    
    await new Promise(resolve => setTimeout(resolve, 1000));
    attempts++;
  }
  
  throw new Error('Synthesis timeout');
};