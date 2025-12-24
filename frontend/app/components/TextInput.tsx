'use client';

import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

interface TextInputProps {
  onTextValidated?: (textData: any) => void;
  onError?: (error: string) => void;
  maxLength?: number;
  placeholder?: string;
  disabled?: boolean;
}

interface TextState {
  text: string;
  characterCount: number;
  isValid: boolean;
  validationErrors: string[];
  detectedLanguage?: string;
  languageConfidence?: number;
  isValidating: boolean;
}

interface ValidationResponse {
  is_valid: boolean;
  text: string;
  character_count: number;
  detected_language?: string;
  language_confidence?: number;
  sanitized_text: string;
  validation_errors?: string[];
}

const MAX_TEXT_LENGTH = 1000;

export default function TextInput({ 
  onTextValidated, 
  onError, 
  maxLength = MAX_TEXT_LENGTH,
  placeholder = "Enter text to synthesize in the cloned voice...",
  disabled = false
}: TextInputProps) {
  const [textState, setTextState] = useState<TextState>({
    text: '',
    characterCount: 0,
    isValid: false,
    validationErrors: [],
    isValidating: false
  });

  // Debounced validation
  const [validationTimeout, setValidationTimeout] = useState<NodeJS.Timeout | null>(null);

  const validateText = useCallback(async (text: string) => {
    if (!text.trim()) {
      setTextState(prev => ({
        ...prev,
        isValid: false,
        validationErrors: [],
        detectedLanguage: undefined,
        languageConfidence: undefined,
        isValidating: false
      }));
      return;
    }

    setTextState(prev => ({ ...prev, isValidating: true }));

    try {
      const response = await axios.post<ValidationResponse>(
        'http://localhost:8000/api/v1/text/validate',
        { text }
      );

      const validationResult = response.data;
      
      setTextState(prev => ({
        ...prev,
        isValid: validationResult.is_valid,
        characterCount: validationResult.character_count,
        validationErrors: validationResult.validation_errors || [],
        detectedLanguage: validationResult.detected_language,
        languageConfidence: validationResult.language_confidence,
        isValidating: false
      }));

      if (validationResult.is_valid) {
        onTextValidated?.(validationResult);
      }

    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || 
                          error.message || 
                          'Text validation failed';
      
      setTextState(prev => ({
        ...prev,
        isValid: false,
        validationErrors: [errorMessage],
        isValidating: false
      }));
      
      onError?.(errorMessage);
    }
  }, [onTextValidated, onError]);

  const handleTextChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newText = event.target.value;
    
    setTextState(prev => ({
      ...prev,
      text: newText,
      characterCount: newText.length
    }));

    // Clear previous timeout
    if (validationTimeout) {
      clearTimeout(validationTimeout);
    }

    // Set new timeout for debounced validation
    const timeout = setTimeout(() => {
      validateText(newText);
    }, 500); // 500ms debounce

    setValidationTimeout(timeout);
  };

  const clearText = () => {
    setTextState({
      text: '',
      characterCount: 0,
      isValid: false,
      validationErrors: [],
      isValidating: false
    });
    
    if (validationTimeout) {
      clearTimeout(validationTimeout);
    }
  };

  const getCharacterCountColor = () => {
    const ratio = textState.characterCount / maxLength;
    if (ratio >= 1) return 'text-red-600 dark:text-red-400';
    if (ratio >= 0.8) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-gray-500 dark:text-gray-400';
  };

  const getLanguageDisplay = () => {
    if (!textState.detectedLanguage) return null;
    
    const confidence = textState.languageConfidence 
      ? Math.round(textState.languageConfidence * 100) 
      : 0;
    
    return (
      <div className="flex items-center space-x-2 text-sm">
        <span className="text-gray-500 dark:text-gray-400">Detected:</span>
        <span className="font-medium text-blue-600 dark:text-blue-400">
          {textState.detectedLanguage.toUpperCase()}
        </span>
        <span className="text-gray-400 dark:text-gray-500">
          ({confidence}% confidence)
        </span>
      </div>
    );
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (validationTimeout) {
        clearTimeout(validationTimeout);
      }
    };
  }, [validationTimeout]);

  return (
    <div className="w-full space-y-4">
      {/* Text Input Area */}
      <div className="relative">
        <textarea
          value={textState.text}
          onChange={handleTextChange}
          placeholder={placeholder}
          disabled={disabled}
          maxLength={maxLength}
          rows={6}
          className={`
            w-full px-4 py-3 border rounded-lg resize-none transition-colors
            ${textState.validationErrors.length > 0 
              ? 'border-red-300 dark:border-red-600 focus:border-red-500 dark:focus:border-red-400' 
              : 'border-gray-300 dark:border-gray-600 focus:border-blue-500 dark:focus:border-blue-400'
            }
            bg-white dark:bg-gray-800 
            text-gray-900 dark:text-gray-100
            placeholder-gray-500 dark:placeholder-gray-400
            focus:outline-none focus:ring-2 focus:ring-blue-500/20 dark:focus:ring-blue-400/20
            disabled:bg-gray-50 dark:disabled:bg-gray-700 disabled:cursor-not-allowed
          `}
        />
        
        {/* Validation Spinner */}
        {textState.isValidating && (
          <div className="absolute top-3 right-3">
            <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-500 border-t-transparent"></div>
          </div>
        )}
      </div>

      {/* Character Count and Language Detection */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          {getLanguageDisplay()}
          
          {textState.isValid && (
            <div className="flex items-center space-x-1 text-sm text-green-600 dark:text-green-400">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
              <span>Valid</span>
            </div>
          )}
        </div>
        
        <div className="flex items-center space-x-3">
          <span className={`text-sm ${getCharacterCountColor()}`}>
            {textState.characterCount}/{maxLength}
          </span>
          
          {textState.text && (
            <button
              onClick={clearText}
              className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Validation Errors */}
      {textState.validationErrors.length > 0 && (
        <div className="space-y-2">
          {textState.validationErrors.map((error, index) => (
            <div key={index} className="flex items-center space-x-2 text-sm text-red-600 dark:text-red-400">
              <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span>{error}</span>
            </div>
          ))}
        </div>
      )}

      {/* Unicode Support Notice */}
      <div className="text-xs text-gray-500 dark:text-gray-400">
        <p>✓ Supports all Unicode characters including special characters and punctuation</p>
        <p>✓ Automatic language detection for cross-language synthesis</p>
      </div>
    </div>
  );
}