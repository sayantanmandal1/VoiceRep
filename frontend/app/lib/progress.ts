/**
 * Comprehensive Progress Tracking System
 * Tracks micro-processes and provides detailed progress information
 */

export interface ProgressStep {
  id: string;
  name: string;
  description: string;
  weight: number; // Relative weight for progress calculation
  status: 'pending' | 'active' | 'completed' | 'failed';
  startTime?: number;
  endTime?: number;
  estimatedDuration?: number; // in milliseconds
  subSteps?: ProgressStep[];
}

export interface ProgressState {
  currentStep: string;
  overallProgress: number; // 0-100
  currentStepProgress: number; // 0-100
  estimatedTimeRemaining: number; // in milliseconds
  elapsedTime: number; // in milliseconds
  status: 'idle' | 'running' | 'completed' | 'failed';
  error?: string;
  steps: ProgressStep[];
}

export class ProgressTracker {
  private state: ProgressState;
  private listeners: ((state: ProgressState) => void)[] = [];
  private startTime: number = 0;

  constructor(steps: ProgressStep[]) {
    this.state = {
      currentStep: '',
      overallProgress: 0,
      currentStepProgress: 0,
      estimatedTimeRemaining: 0,
      elapsedTime: 0,
      status: 'idle',
      steps: steps
    };
  }

  // Subscribe to progress updates
  subscribe(listener: (state: ProgressState) => void): () => void {
    this.listeners.push(listener);
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  // Start tracking
  start(): void {
    this.startTime = Date.now();
    this.state.status = 'running';
    this.state.elapsedTime = 0;
    this.notifyListeners();
  }

  // Update current step
  setCurrentStep(stepId: string, progress: number = 0): void {
    const step = this.findStep(stepId);
    if (step) {
      // Mark previous steps as completed
      this.markPreviousStepsCompleted(stepId);
      
      // Update current step
      step.status = 'active';
      step.startTime = Date.now();
      
      this.state.currentStep = stepId;
      this.state.currentStepProgress = progress;
      this.updateOverallProgress();
      this.updateTimeEstimates();
      this.notifyListeners();
    }
  }

  // Update progress of current step
  updateStepProgress(stepId: string, progress: number): void {
    const step = this.findStep(stepId);
    if (step && step.status === 'active') {
      this.state.currentStepProgress = Math.min(100, Math.max(0, progress));
      this.updateOverallProgress();
      this.updateTimeEstimates();
      this.notifyListeners();
    }
  }

  // Complete a step
  completeStep(stepId: string): void {
    const step = this.findStep(stepId);
    if (step) {
      step.status = 'completed';
      step.endTime = Date.now();
      this.state.currentStepProgress = 100;
      this.updateOverallProgress();
      this.updateTimeEstimates();
      this.notifyListeners();
    }
  }

  // Fail a step
  failStep(stepId: string, error: string): void {
    const step = this.findStep(stepId);
    if (step) {
      step.status = 'failed';
      step.endTime = Date.now();
      this.state.status = 'failed';
      this.state.error = error;
      this.notifyListeners();
    }
  }

  // Complete all tracking
  complete(): void {
    this.state.status = 'completed';
    this.state.overallProgress = 100;
    this.state.currentStepProgress = 100;
    this.state.estimatedTimeRemaining = 0;
    this.state.elapsedTime = Date.now() - this.startTime;
    
    // Mark all steps as completed
    this.state.steps.forEach(step => {
      if (step.status !== 'failed') {
        step.status = 'completed';
        step.endTime = Date.now();
      }
    });
    
    this.notifyListeners();
  }

  // Get current state
  getState(): ProgressState {
    this.state.elapsedTime = Date.now() - this.startTime;
    return { ...this.state };
  }

  // Private methods
  private findStep(stepId: string): ProgressStep | undefined {
    for (const step of this.state.steps) {
      if (step.id === stepId) return step;
      if (step.subSteps) {
        const subStep = step.subSteps.find(s => s.id === stepId);
        if (subStep) return subStep;
      }
    }
    return undefined;
  }

  private markPreviousStepsCompleted(currentStepId: string): void {
    let foundCurrent = false;
    for (const step of this.state.steps) {
      if (step.id === currentStepId) {
        foundCurrent = true;
        break;
      }
      if (!foundCurrent && step.status === 'pending') {
        step.status = 'completed';
        step.endTime = Date.now();
      }
    }
  }

  private updateOverallProgress(): void {
    let totalWeight = 0;
    let completedWeight = 0;

    for (const step of this.state.steps) {
      totalWeight += step.weight;
      
      if (step.status === 'completed') {
        completedWeight += step.weight;
      } else if (step.status === 'active') {
        completedWeight += (step.weight * this.state.currentStepProgress / 100);
      }
    }

    this.state.overallProgress = totalWeight > 0 ? (completedWeight / totalWeight) * 100 : 0;
  }

  private updateTimeEstimates(): void {
    const elapsedTime = Date.now() - this.startTime;
    const progress = this.state.overallProgress;
    
    if (progress > 0) {
      const totalEstimatedTime = (elapsedTime / progress) * 100;
      this.state.estimatedTimeRemaining = Math.max(0, totalEstimatedTime - elapsedTime);
    } else {
      // Use step estimates if available
      let totalEstimated = 0;
      for (const step of this.state.steps) {
        if (step.estimatedDuration) {
          totalEstimated += step.estimatedDuration;
        }
      }
      this.state.estimatedTimeRemaining = totalEstimated;
    }
  }

  private notifyListeners(): void {
    this.listeners.forEach(listener => listener(this.getState()));
  }
}

// Predefined progress steps for common operations
export const VOICE_CLONING_STEPS: ProgressStep[] = [
  {
    id: 'session-init',
    name: 'Initializing Session',
    description: 'Creating secure session and preparing workspace',
    weight: 5,
    status: 'pending',
    estimatedDuration: 2000
  },
  {
    id: 'file-upload',
    name: 'Uploading File',
    description: 'Transferring audio/video file to server',
    weight: 15,
    status: 'pending',
    estimatedDuration: 10000
  },
  {
    id: 'file-validation',
    name: 'Validating File',
    description: 'Checking file format, quality, and extracting metadata',
    weight: 10,
    status: 'pending',
    estimatedDuration: 5000
  },
  {
    id: 'audio-extraction',
    name: 'Extracting Audio',
    description: 'Converting video to audio and optimizing quality',
    weight: 20,
    status: 'pending',
    estimatedDuration: 15000
  },
  {
    id: 'voice-analysis',
    name: 'Analyzing Voice',
    description: 'Extracting voice characteristics and building voice model',
    weight: 30,
    status: 'pending',
    estimatedDuration: 45000,
    subSteps: [
      {
        id: 'voice-analysis-features',
        name: 'Feature Extraction',
        description: 'Analyzing pitch, timbre, and spectral characteristics',
        weight: 40,
        status: 'pending',
        estimatedDuration: 15000
      },
      {
        id: 'voice-analysis-prosody',
        name: 'Prosody Analysis',
        description: 'Analyzing rhythm, stress, and intonation patterns',
        weight: 30,
        status: 'pending',
        estimatedDuration: 12000
      },
      {
        id: 'voice-analysis-model',
        name: 'Model Generation',
        description: 'Creating neural voice model from analyzed features',
        weight: 30,
        status: 'pending',
        estimatedDuration: 18000
      }
    ]
  },
  {
    id: 'text-processing',
    name: 'Processing Text',
    description: 'Validating and preparing text for synthesis',
    weight: 5,
    status: 'pending',
    estimatedDuration: 3000
  },
  {
    id: 'speech-synthesis',
    name: 'Synthesizing Speech',
    description: 'Generating speech in the cloned voice',
    weight: 25,
    status: 'pending',
    estimatedDuration: 30000,
    subSteps: [
      {
        id: 'synthesis-preparation',
        name: 'Synthesis Preparation',
        description: 'Loading voice model and preparing synthesis pipeline',
        weight: 20,
        status: 'pending',
        estimatedDuration: 5000
      },
      {
        id: 'synthesis-generation',
        name: 'Audio Generation',
        description: 'Generating raw audio with voice characteristics',
        weight: 60,
        status: 'pending',
        estimatedDuration: 20000
      },
      {
        id: 'synthesis-postprocessing',
        name: 'Post-processing',
        description: 'Enhancing audio quality and applying final touches',
        weight: 20,
        status: 'pending',
        estimatedDuration: 5000
      }
    ]
  },
  {
    id: 'finalization',
    name: 'Finalizing',
    description: 'Preparing download and cleaning up temporary files',
    weight: 5,
    status: 'pending',
    estimatedDuration: 2000
  }
];

// Utility functions
export function formatTime(milliseconds: number): string {
  if (milliseconds < 1000) {
    return '< 1 second';
  }
  
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  
  if (hours > 0) {
    return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}

export function getProgressColor(progress: number): string {
  if (progress < 25) return '#ef4444'; // red
  if (progress < 50) return '#f97316'; // orange
  if (progress < 75) return '#eab308'; // yellow
  return '#22c55e'; // green
}