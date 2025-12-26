/**
 * Frontend progress tracking utilities for comprehensive micro-process monitoring
 */

export interface MicroProcess {
  id: string;
  name: string;
  description: string;
  estimatedDuration: number;
  startTime?: number;
  endTime?: number;
  progress: number;
  status: 'pending' | 'active' | 'completed' | 'error';
  error?: string;
}

export interface ProcessGroup {
  id: string;
  name: string;
  description: string;
  processes: MicroProcess[];
  overallProgress: number;
  status: 'pending' | 'active' | 'completed' | 'error';
}

export class ProgressTracker {
  private processGroups: Map<string, ProcessGroup> = new Map();
  private listeners: Set<(groups: ProcessGroup[]) => void> = new Set();

  // Add a listener for progress updates
  addListener(listener: (groups: ProcessGroup[]) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  // Notify all listeners of progress changes
  private notifyListeners(): void {
    const groups = Array.from(this.processGroups.values());
    this.listeners.forEach(listener => listener(groups));
  }

  // Create a new process group
  createProcessGroup(
    groupId: string,
    name: string,
    description: string,
    processes: Omit<MicroProcess, 'progress' | 'status'>[]
  ): void {
    const processGroup: ProcessGroup = {
      id: groupId,
      name,
      description,
      processes: processes.map(p => ({
        ...p,
        progress: 0,
        status: 'pending'
      })),
      overallProgress: 0,
      status: 'pending'
    };

    this.processGroups.set(groupId, processGroup);
    this.notifyListeners();
  }

  // Start a specific process
  startProcess(groupId: string, processId: string): void {
    const group = this.processGroups.get(groupId);
    if (!group) return;

    const process = group.processes.find(p => p.id === processId);
    if (!process) return;

    process.status = 'active';
    process.startTime = Date.now();
    process.progress = 0;

    // Update group status
    if (group.status === 'pending') {
      group.status = 'active';
    }

    this.updateOverallProgress(groupId);
    this.notifyListeners();
  }

  // Update process progress
  updateProcessProgress(
    groupId: string,
    processId: string,
    progress: number,
    description?: string
  ): void {
    const group = this.processGroups.get(groupId);
    if (!group) return;

    const process = group.processes.find(p => p.id === processId);
    if (!process) return;

    process.progress = Math.min(Math.max(progress, 0), 100);
    
    if (description) {
      process.description = description;
    }

    this.updateOverallProgress(groupId);
    this.notifyListeners();
  }

  // Complete a process
  completeProcess(groupId: string, processId: string): void {
    const group = this.processGroups.get(groupId);
    if (!group) return;

    const process = group.processes.find(p => p.id === processId);
    if (!process) return;

    process.status = 'completed';
    process.progress = 100;
    process.endTime = Date.now();

    this.updateOverallProgress(groupId);
    
    // Check if all processes are completed
    const allCompleted = group.processes.every(p => p.status === 'completed');
    if (allCompleted) {
      group.status = 'completed';
    }

    this.notifyListeners();
  }

  // Mark a process as failed
  failProcess(groupId: string, processId: string, error: string): void {
    const group = this.processGroups.get(groupId);
    if (!group) return;

    const process = group.processes.find(p => p.id === processId);
    if (!process) return;

    process.status = 'error';
    process.error = error;
    process.endTime = Date.now();

    // Mark group as failed
    group.status = 'error';

    this.updateOverallProgress(groupId);
    this.notifyListeners();
  }

  // Update overall progress for a group
  private updateOverallProgress(groupId: string): void {
    const group = this.processGroups.get(groupId);
    if (!group) return;

    const totalProgress = group.processes.reduce((sum, process) => sum + process.progress, 0);
    group.overallProgress = totalProgress / group.processes.length;
  }

  // Get current state of a process group
  getProcessGroup(groupId: string): ProcessGroup | undefined {
    return this.processGroups.get(groupId);
  }

  // Get all process groups
  getAllProcessGroups(): ProcessGroup[] {
    return Array.from(this.processGroups.values());
  }

  // Reset a process group
  resetProcessGroup(groupId: string): void {
    const group = this.processGroups.get(groupId);
    if (!group) return;

    group.status = 'pending';
    group.overallProgress = 0;
    group.processes.forEach(process => {
      process.status = 'pending';
      process.progress = 0;
      process.startTime = undefined;
      process.endTime = undefined;
      process.error = undefined;
    });

    this.notifyListeners();
  }

  // Remove a process group
  removeProcessGroup(groupId: string): void {
    this.processGroups.delete(groupId);
    this.notifyListeners();
  }

  // Get estimated time remaining for a process
  getEstimatedTimeRemaining(groupId: string, processId: string): number | null {
    const group = this.processGroups.get(groupId);
    if (!group) return null;

    const process = group.processes.find(p => p.id === processId);
    if (!process || process.status !== 'active' || !process.startTime) return null;

    const elapsed = Date.now() - process.startTime;
    const progressRatio = process.progress / 100;
    
    if (progressRatio <= 0) {
      return process.estimatedDuration;
    }

    const estimatedTotal = elapsed / progressRatio;
    return Math.max(0, estimatedTotal - elapsed);
  }

  // Get estimated time remaining for entire group
  getGroupEstimatedTimeRemaining(groupId: string): number | null {
    const group = this.processGroups.get(groupId);
    if (!group) return null;

    let totalRemaining = 0;
    let hasActiveProcess = false;

    for (const process of group.processes) {
      if (process.status === 'pending') {
        totalRemaining += process.estimatedDuration;
      } else if (process.status === 'active') {
        hasActiveProcess = true;
        const remaining = this.getEstimatedTimeRemaining(groupId, process.id);
        if (remaining !== null) {
          totalRemaining += remaining;
        }
      }
    }

    return hasActiveProcess || totalRemaining > 0 ? totalRemaining : null;
  }
}

// Global progress tracker instance
export const progressTracker = new ProgressTracker();

// Predefined process templates for common operations
export const PROCESS_TEMPLATES = {
  FILE_UPLOAD: [
    {
      id: 'validation',
      name: 'File Validation',
      description: 'Validating file format and size',
      estimatedDuration: 1000
    },
    {
      id: 'preparation',
      name: 'Upload Preparation',
      description: 'Preparing file for upload',
      estimatedDuration: 500
    },
    {
      id: 'upload',
      name: 'File Transfer',
      description: 'Uploading file to server',
      estimatedDuration: 5000
    },
    {
      id: 'processing',
      name: 'Server Processing',
      description: 'Processing and analyzing file',
      estimatedDuration: 3000
    },
    {
      id: 'finalization',
      name: 'Finalization',
      description: 'Completing upload process',
      estimatedDuration: 1000
    }
  ],

  VOICE_SYNTHESIS: [
    {
      id: 'initialization',
      name: 'Initialization',
      description: 'Preparing synthesis environment',
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
      description: 'Processing and normalizing text',
      estimatedDuration: 1500
    },
    {
      id: 'model_loading',
      name: 'Model Loading',
      description: 'Loading synthesis models',
      estimatedDuration: 3000
    },
    {
      id: 'synthesis',
      name: 'Voice Synthesis',
      description: 'Generating synthetic speech',
      estimatedDuration: 8000
    },
    {
      id: 'post_processing',
      name: 'Post-Processing',
      description: 'Enhancing audio quality',
      estimatedDuration: 2000
    },
    {
      id: 'finalization',
      name: 'Finalization',
      description: 'Preparing final output',
      estimatedDuration: 1000
    }
  ],

  TEXT_VALIDATION: [
    {
      id: 'sanitization',
      name: 'Text Sanitization',
      description: 'Cleaning and normalizing text',
      estimatedDuration: 500
    },
    {
      id: 'language_detection',
      name: 'Language Detection',
      description: 'Detecting text language',
      estimatedDuration: 1000
    },
    {
      id: 'validation',
      name: 'Content Validation',
      description: 'Validating text content',
      estimatedDuration: 800
    },
    {
      id: 'preparation',
      name: 'Preparation',
      description: 'Preparing text for synthesis',
      estimatedDuration: 300
    }
  ]
};

// Utility function to format duration
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${Math.round(ms / 1000)}s`;
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.round((ms % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
}

// Utility function to get progress color
export function getProgressColor(progress: number): string {
  if (progress < 25) return 'bg-red-500';
  if (progress < 50) return 'bg-yellow-500';
  if (progress < 75) return 'bg-blue-500';
  return 'bg-green-500';
}