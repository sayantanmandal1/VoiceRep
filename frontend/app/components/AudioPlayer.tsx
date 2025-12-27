'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';

interface AudioPlayerProps {
  audioUrl: string;
  title?: string;
  onDownload?: () => void;
  className?: string;
}

interface PlayerState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  isLoading: boolean;
  error?: string;
  isBuffering: boolean;
}

interface AudioMetadata {
  duration: number;
  sample_rate: number;
  language?: string;
  quality_score?: number;
  processing_time?: number;
}

export default function AudioPlayer({ 
  audioUrl, 
  title = "Synthesized Audio", 
  onDownload,
  className = ""
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const [playerState, setPlayerState] = useState<PlayerState>({
    isPlaying: false,
    currentTime: 0,
    duration: 0,
    volume: 1.0,
    isLoading: true,
    isBuffering: false
  });

  // Format time display
  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Handle play/pause
  const togglePlayPause = useCallback(async () => {
    if (!audioRef.current) return;

    try {
      if (playerState.isPlaying) {
        audioRef.current.pause();
      } else {
        // Stop all other audio elements before playing this one
        const allAudio = document.querySelectorAll('audio');
        allAudio.forEach(audio => {
          if (audio !== audioRef.current) {
            audio.pause();
          }
        });
        
        await audioRef.current.play();
      }
    } catch (error) {
      console.error('Error playing audio:', error);
      setPlayerState(prev => ({ 
        ...prev, 
        error: 'Failed to play audio',
        isPlaying: false 
      }));
    }
  }, [playerState.isPlaying]);

  // Handle seek
  const handleSeek = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || !progressRef.current) return;

    const rect = progressRef.current.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const width = rect.width;
    const percentage = clickX / width;
    const newTime = percentage * playerState.duration;

    audioRef.current.currentTime = newTime;
    setPlayerState(prev => ({ ...prev, currentTime: newTime }));
  }, [playerState.duration]);

  // Handle volume change
  const handleVolumeChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(event.target.value);
    if (audioRef.current) {
      audioRef.current.volume = newVolume;
    }
    setPlayerState(prev => ({ ...prev, volume: newVolume }));
  }, []);

  // Handle download
  const handleDownload = useCallback(async () => {
    try {
      setPlayerState(prev => ({ ...prev, isLoading: true }));
      
      const response = await axios.get(audioUrl, {
        responseType: 'blob',
        headers: {
          'Accept': 'audio/wav,audio/mpeg,audio/*'
        }
      });

      // Create download link
      const blob = new Blob([response.data], { type: response.headers['content-type'] || 'audio/wav' });
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      
      // Generate filename with timestamp
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const filename = `synthesized_${timestamp}.wav`;
      
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Cleanup
      window.URL.revokeObjectURL(downloadUrl);
      
      onDownload?.();
    } catch (error) {
      console.error('Download failed:', error);
      setPlayerState(prev => ({ 
        ...prev, 
        error: 'Download failed. Please try again.' 
      }));
    } finally {
      setPlayerState(prev => ({ ...prev, isLoading: false }));
    }
  }, [audioUrl, onDownload]);

  // Audio event handlers
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadStart = () => {
      setPlayerState(prev => ({ ...prev, isLoading: true, error: undefined }));
    };

    const handleLoadedMetadata = () => {
      setPlayerState(prev => ({ 
        ...prev, 
        duration: audio.duration,
        isLoading: false 
      }));
    };

    const handleTimeUpdate = () => {
      setPlayerState(prev => ({ 
        ...prev, 
        currentTime: audio.currentTime 
      }));
    };

    const handlePlay = () => {
      setPlayerState(prev => ({ ...prev, isPlaying: true }));
    };

    const handlePause = () => {
      setPlayerState(prev => ({ ...prev, isPlaying: false }));
    };

    const handleEnded = () => {
      setPlayerState(prev => ({ 
        ...prev, 
        isPlaying: false, 
        currentTime: 0 
      }));
      audio.currentTime = 0;
    };

    const handleError = () => {
      setPlayerState(prev => ({ 
        ...prev, 
        error: 'Failed to load audio',
        isLoading: false,
        isPlaying: false 
      }));
    };

    const handleWaiting = () => {
      setPlayerState(prev => ({ ...prev, isBuffering: true }));
    };

    const handleCanPlay = () => {
      setPlayerState(prev => ({ ...prev, isBuffering: false }));
    };

    // Add event listeners
    audio.addEventListener('loadstart', handleLoadStart);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('error', handleError);
    audio.addEventListener('waiting', handleWaiting);
    audio.addEventListener('canplay', handleCanPlay);

    return () => {
      // Cleanup event listeners
      audio.removeEventListener('loadstart', handleLoadStart);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('error', handleError);
      audio.removeEventListener('waiting', handleWaiting);
      audio.removeEventListener('canplay', handleCanPlay);
    };
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.target !== document.body) return; // Only when not focused on input
      
      switch (event.code) {
        case 'Space':
          event.preventDefault();
          togglePlayPause();
          break;
        case 'ArrowLeft':
          if (audioRef.current) {
            audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - 10);
          }
          break;
        case 'ArrowRight':
          if (audioRef.current) {
            audioRef.current.currentTime = Math.min(playerState.duration, audioRef.current.currentTime + 10);
          }
          break;
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => document.removeEventListener('keydown', handleKeyPress);
  }, [togglePlayPause, playerState.duration]);

  // Cleanup audio when component unmounts
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
    };
  }, []);

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 ${className}`}>
      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        src={audioUrl}
        preload="metadata"
        crossOrigin="anonymous"
      />

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
            <svg className="w-5 h-5 text-blue-600 dark:text-blue-400" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.617.816L4.5 13.5H2a1 1 0 01-1-1V7.5a1 1 0 011-1h2.5l3.883-3.316a1 1 0 011.617.816zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.983 5.983 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.984 3.984 0 00-1.172-2.828 1 1 0 010-1.415z" clipRule="evenodd" />
            </svg>
          </div>
          <div>
            <h3 className="font-medium text-gray-900 dark:text-gray-100">{title}</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {playerState.duration > 0 ? formatTime(playerState.duration) : 'Loading...'}
            </p>
          </div>
        </div>

        {/* Download button */}
        <button
          onClick={handleDownload}
          disabled={playerState.isLoading}
          className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {playerState.isLoading ? (
            <div className="animate-spin rounded-full h-4 w-4 border-2 border-gray-400 border-t-transparent"></div>
          ) : (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          )}
          <span>Download</span>
        </button>
      </div>

      {/* Error display */}
      {playerState.error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
          <p className="text-sm text-red-700 dark:text-red-400">{playerState.error}</p>
        </div>
      )}

      {/* Progress bar */}
      <div className="mb-4">
        <div 
          ref={progressRef}
          className="relative h-2 bg-gray-200 dark:bg-gray-700 rounded-full cursor-pointer"
          onClick={handleSeek}
        >
          <div 
            className="absolute top-0 left-0 h-full bg-blue-600 rounded-full transition-all duration-100"
            style={{ 
              width: playerState.duration > 0 
                ? `${(playerState.currentTime / playerState.duration) * 100}%` 
                : '0%' 
            }}
          />
          {/* Buffering indicator */}
          {playerState.isBuffering && (
            <div className="absolute top-0 left-0 h-full bg-blue-300 dark:bg-blue-700 rounded-full animate-pulse"
                 style={{ width: '100%' }} />
          )}
        </div>
        
        {/* Time display */}
        <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
          <span>{formatTime(playerState.currentTime)}</span>
          <span>{formatTime(playerState.duration)}</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        {/* Play/Pause and Skip buttons */}
        <div className="flex items-center space-x-3">
          {/* Skip backward */}
          <button
            onClick={() => {
              if (audioRef.current) {
                audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - 10);
              }
            }}
            className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
            title="Skip backward 10s"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M15.707 15.707a1 1 0 01-1.414 0l-5-5a1 1 0 010-1.414l5-5a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 010 1.414zm-6 0a1 1 0 01-1.414 0l-5-5a1 1 0 010-1.414l5-5a1 1 0 011.414 1.414L5.414 10l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
            </svg>
          </button>

          {/* Play/Pause button */}
          <button
            onClick={togglePlayPause}
            disabled={playerState.isLoading}
            className="flex items-center justify-center w-12 h-12 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-full transition-colors disabled:cursor-not-allowed"
            title={playerState.isPlaying ? "Pause" : "Play"}
          >
            {playerState.isLoading ? (
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
            ) : playerState.isPlaying ? (
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg className="w-5 h-5 ml-1" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
              </svg>
            )}
          </button>

          {/* Skip forward */}
          <button
            onClick={() => {
              if (audioRef.current) {
                audioRef.current.currentTime = Math.min(playerState.duration, audioRef.current.currentTime + 10);
              }
            }}
            className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
            title="Skip forward 10s"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414zm6 0a1 1 0 011.414 0l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414-1.414L14.586 10l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>

        {/* Volume control */}
        <div className="flex items-center space-x-2">
          <svg className="w-4 h-4 text-gray-600 dark:text-gray-400" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.617.816L4.5 13.5H2a1 1 0 01-1-1V7.5a1 1 0 011-1h2.5l3.883-3.316a1 1 0 011.617.816zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={playerState.volume}
            onChange={handleVolumeChange}
            className="w-20 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
            title={`Volume: ${Math.round(playerState.volume * 100)}%`}
          />
        </div>
      </div>

      {/* Keyboard shortcuts hint */}
      <div className="mt-4 text-xs text-gray-500 dark:text-gray-400 text-center">
        <p>Keyboard shortcuts: Space (play/pause), ← (skip back 10s), → (skip forward 10s)</p>
      </div>
    </div>
  );
}