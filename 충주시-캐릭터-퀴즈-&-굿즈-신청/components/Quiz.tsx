
import React, { useState, useEffect } from 'react';
import { QuizQuestion } from '../types';
import { CheckIcon } from './icons/CheckIcon';
import { XIcon } from './icons/XIcon';

interface QuizProps {
  question: QuizQuestion;
  questionNumber: number;
  totalQuestions: number;
  onAnswerSubmit: (selectedAnswer: string) => void;
}

const Quiz: React.FC<QuizProps> = ({ question, questionNumber, totalQuestions, onAnswerSubmit }) => {
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<'correct' | 'incorrect' | null>(null);
  const [isAnswered, setIsAnswered] = useState<boolean>(false);

  useEffect(() => {
    setSelectedOption(null);
    setFeedback(null);
    setIsAnswered(false);
  }, [question]);

  const handleOptionClick = (option: string) => {
    if (isAnswered) return;

    setSelectedOption(option);
    setIsAnswered(true);
    const isCorrect = option === question.correctAnswer;
    setFeedback(isCorrect ? 'correct' : 'incorrect');

    setTimeout(() => {
      onAnswerSubmit(option);
    }, 1500); // Wait for feedback animation
  };

  const getOptionClasses = (option: string) => {
    let baseClasses = "w-full text-left p-4 my-2 rounded-lg border-2 transition-all duration-300 cursor-pointer text-sm sm:text-base";
    if (!isAnswered) {
      return `${baseClasses} border-sky-600 hover:bg-sky-700 hover:border-sky-500 bg-sky-800 text-slate-100`;
    }
    if (option === selectedOption) {
      if (feedback === 'correct') {
        return `${baseClasses} border-green-500 bg-green-700 text-white animate-pulseCorrect`;
      }
      if (feedback === 'incorrect') {
        return `${baseClasses} border-red-500 bg-red-700 text-white animate-pulseIncorrect`;
      }
    }
    if (option === question.correctAnswer) {
        return `${baseClasses} border-green-500 bg-green-700 text-white`;
    }
    return `${baseClasses} border-slate-600 bg-slate-700 text-slate-400 cursor-not-allowed`;
  };

  return (
    <div className="w-full animate-fadeIn">
      <div className="mb-4 text-right text-yellow-400 font-semibold">
        질문 {questionNumber} / {totalQuestions}
      </div>
      {question.characterImage && (
         <img 
            src={question.characterImage} 
            alt="Quiz context image" 
            className="w-full max-w-xs h-auto mx-auto mb-6 rounded-lg shadow-md object-contain"
            style={{maxHeight: '200px'}}
        />
      )}
      <h2 className="text-xl sm:text-2xl font-semibold mb-6 text-slate-100 leading-relaxed">{question.questionText}</h2>
      <div className="space-y-3">
        {question.options.map((option, index) => (
          <button
            key={index}
            onClick={() => handleOptionClick(option)}
            disabled={isAnswered}
            className={`${getOptionClasses(option)} flex items-center justify-between group`}
          >
            <span>{option}</span>
            {isAnswered && option === selectedOption && feedback === 'correct' && <CheckIcon className="w-6 h-6 text-green-300" />}
            {isAnswered && option === selectedOption && feedback === 'incorrect' && <XIcon className="w-6 h-6 text-red-300" />}
            {isAnswered && option !== selectedOption && option === question.correctAnswer && <CheckIcon className="w-6 h-6 text-green-300 opacity-70" />}
          </button>
        ))}
      </div>
       {/* Tailwind CSS for animations - needs to be in a real CSS file or configured in tailwind.config.js usually.
           For CDN, we can define simple keyframes using a style tag or rely on simple class changes.
           Here, we'll imply animations with text like animate-pulseCorrect etc. and expect them to be defined elsewhere, or use simple transitions.
           Let's add simple keyframes in index.html for this demo.
      */}
      <style>{`
        @keyframes pulseCorrect { 0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); } 50% { transform: scale(1.02); box-shadow: 0 0 0 10px rgba(74, 222, 128, 0); } }
        .animate-pulseCorrect { animation: pulseCorrect 1.5s ease-out; }
        @keyframes pulseIncorrect { 0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); } 50% { transform: scale(1.02); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); } }
        .animate-pulseIncorrect { animation: pulseIncorrect 1.5s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .animate-fadeIn { animation: fadeIn 0.5s ease-out forwards; }
      `}</style>
    </div>
  );
};

export default Quiz;
