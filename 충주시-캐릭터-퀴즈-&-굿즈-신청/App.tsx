
import React, { useState, useCallback, useEffect } from 'react';
import { AppPhase, QuizQuestion, MerchandiseFormData } from './types';
import { QUIZ_QUESTIONS, PASSING_SCORE_PERCENTAGE, CHARACTER_NAME, NIGHT_MARKET_NAME, MAIN_CHARACTER_IMAGE_URL } from './constants';
import Header from './components/Header';
import Footer from './components/Footer';
import Quiz from './components/Quiz';
import QuizResult from './components/QuizResult';
import MerchandiseForm from './components/MerchandiseForm';
import { WelcomeScreen } from './components/WelcomeScreen';
import { SubmissionSuccessScreen } from './components/SubmissionSuccessScreen';

const App: React.FC = () => {
  const [appPhase, setAppPhase] = useState<AppPhase>(AppPhase.WELCOME);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState<number>(0);
  const [score, setScore] = useState<number>(0);
  const [quizPassed, setQuizPassed] = useState<boolean>(false);

  const totalQuestions = QUIZ_QUESTIONS.length;
  const passingScore = Math.ceil(totalQuestions * PASSING_SCORE_PERCENTAGE);

  const resetQuiz = useCallback(() => {
    setCurrentQuestionIndex(0);
    setScore(0);
    setQuizPassed(false);
  }, []);

  const handleStartQuiz = useCallback(() => {
    resetQuiz();
    setAppPhase(AppPhase.QUIZ);
  }, [resetQuiz]);

  const handleAnswerSubmit = useCallback((selectedAnswer: string) => {
    if (QUIZ_QUESTIONS[currentQuestionIndex].correctAnswer === selectedAnswer) {
      setScore(prevScore => prevScore + 1);
    }
    
    if (currentQuestionIndex < totalQuestions - 1) {
      setCurrentQuestionIndex(prevIndex => prevIndex + 1);
    } else {
      setAppPhase(AppPhase.RESULT);
    }
  }, [currentQuestionIndex, totalQuestions]);
  
  useEffect(() => {
    if (appPhase === AppPhase.RESULT) {
      setQuizPassed(score >= passingScore);
    }
  }, [appPhase, score, passingScore]);

  const handleProceedToForm = useCallback(() => {
    setAppPhase(AppPhase.FORM);
  }, []);

  const handleRetryQuiz = useCallback(() => {
    resetQuiz();
    setAppPhase(AppPhase.QUIZ);
  }, [resetQuiz]);

  const handleFormSubmit = useCallback((formData: MerchandiseFormData) => {
    console.log("Merchandise Form Submitted:", formData); // In a real app, send this to a server
    setAppPhase(AppPhase.SUBMITTED);
  }, []);

  const handleStartOver = useCallback(() => {
    resetQuiz();
    setAppPhase(AppPhase.WELCOME);
  }, [resetQuiz]);

  const renderContent = () => {
    switch (appPhase) {
      case AppPhase.WELCOME:
        return (
          <WelcomeScreen
            characterName={CHARACTER_NAME}
            nightMarketName={NIGHT_MARKET_NAME}
            characterImageUrl={MAIN_CHARACTER_IMAGE_URL}
            onStartQuiz={handleStartQuiz}
          />
        );
      case AppPhase.QUIZ:
        if (currentQuestionIndex < totalQuestions) {
          return (
            <Quiz
              question={QUIZ_QUESTIONS[currentQuestionIndex]}
              questionNumber={currentQuestionIndex + 1}
              totalQuestions={totalQuestions}
              onAnswerSubmit={handleAnswerSubmit}
            />
          );
        }
        return null; // Should transition to RESULT before this
      case AppPhase.RESULT:
        return (
          <QuizResult
            score={score}
            totalQuestions={totalQuestions}
            passed={quizPassed}
            passingScore={passingScore}
            onProceedToForm={handleProceedToForm}
            onRetryQuiz={handleRetryQuiz}
          />
        );
      case AppPhase.FORM:
        return <MerchandiseForm onSubmit={handleFormSubmit} />;
      case AppPhase.SUBMITTED:
        return <SubmissionSuccessScreen characterName={CHARACTER_NAME} onStartOver={handleStartOver} />;
      default:
        return <p>오류가 발생했습니다. (An error occurred.)</p>;
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-between bg-gradient-to-br from-slate-900 via-purple-900 to-sky-800 p-4 selection:bg-yellow-400 selection:text-slate-900">
      <Header characterName={CHARACTER_NAME} />
      <main className="flex-grow flex items-center justify-center w-full max-w-2xl py-8">
        <div className="bg-slate-800 bg-opacity-80 backdrop-blur-md shadow-2xl rounded-xl p-6 md:p-10 w-full transition-all duration-500 ease-in-out">
          {renderContent()}
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default App;
