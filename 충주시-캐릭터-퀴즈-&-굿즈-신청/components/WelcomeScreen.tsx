
import React from 'react';

interface WelcomeScreenProps {
  characterName: string;
  nightMarketName: string;
  characterImageUrl: string;
  onStartQuiz: () => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  characterName,
  nightMarketName,
  characterImageUrl,
  onStartQuiz,
}) => {
  return (
    <div className="text-center animate-fadeIn">
      <img 
        src={characterImageUrl} 
        alt={characterName} 
        className="w-48 h-48 sm:w-64 sm:h-64 mx-auto mb-6 rounded-full shadow-lg border-4 border-yellow-400 object-cover" 
      />
      <h2 className="text-2xl sm:text-3xl font-bold text-yellow-400 mb-2">어서오세요!</h2>
      <p className="text-slate-200 mb-6 text-base sm:text-lg">
        {nightMarketName}의 귀염둥이 마스코트, <span className="font-semibold text-orange-400">{characterName}</span>에 대해 얼마나 알고 계신가요? <br/>
        퀴즈를 풀고 특별한 {characterName} 굿즈의 주인공이 되어보세요!
      </p>
      <button
        onClick={onStartQuiz}
        className="px-8 py-3 bg-gradient-to-r from-orange-500 to-yellow-500 hover:from-orange-600 hover:to-yellow-600 text-slate-900 font-bold rounded-lg shadow-md hover:shadow-lg transform hover:scale-105 transition-all duration-300 text-lg"
      >
        퀴즈 시작하기!
      </button>
    </div>
  );
};
