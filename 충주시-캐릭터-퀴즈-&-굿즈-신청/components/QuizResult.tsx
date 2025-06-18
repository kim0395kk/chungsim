
import React from 'react';

interface QuizResultProps {
  score: number;
  totalQuestions: number;
  passed: boolean;
  passingScore: number;
  onProceedToForm: () => void;
  onRetryQuiz: () => void;
}

const QuizResult: React.FC<QuizResultProps> = ({
  score,
  totalQuestions,
  passed,
  passingScore,
  onProceedToForm,
  onRetryQuiz,
}) => {
  return (
    <div className="text-center animate-fadeIn">
      <h2 className={`text-3xl font-bold mb-4 ${passed ? 'text-green-400' : 'text-red-400'}`}>
        {passed ? '🎉 퀴즈 통과! 🎉' : '아쉬워요! 😥'}
      </h2>
      <p className="text-xl text-slate-200 mb-2">
        총 <span className="font-semibold text-yellow-400">{totalQuestions}</span>문제 중 <span className="font-semibold text-yellow-400">{score}</span>문제를 맞추셨습니다!
      </p>
      <p className="text-slate-300 mb-8">
        (통과 기준: {passingScore}개 이상 정답)
      </p>

      {passed ? (
        <>
          <p className="text-lg text-green-300 mb-6">축하드립니다! 충심이 굿즈 신청 자격이 주어졌습니다.</p>
          <button
            onClick={onProceedToForm}
            className="px-8 py-3 bg-gradient-to-r from-green-500 to-teal-500 hover:from-green-600 hover:to-teal-600 text-white font-bold rounded-lg shadow-md hover:shadow-lg transform hover:scale-105 transition-all duration-300 text-lg"
          >
            굿즈 신청하러 가기
          </button>
        </>
      ) : (
        <>
          <p className="text-lg text-red-300 mb-6">아쉽지만 다음 기회에 도전해주세요! 충심이에 대해 조금 더 알아보고 다시 도전해보세요!</p>
          <button
            onClick={onRetryQuiz}
            className="px-8 py-3 bg-gradient-to-r from-orange-500 to-yellow-500 hover:from-orange-600 hover:to-yellow-600 text-slate-900 font-bold rounded-lg shadow-md hover:shadow-lg transform hover:scale-105 transition-all duration-300 text-lg"
          >
            퀴즈 다시 풀기
          </button>
        </>
      )}
    </div>
  );
};

export default QuizResult;
