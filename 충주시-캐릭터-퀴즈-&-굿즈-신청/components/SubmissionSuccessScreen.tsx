
import React from 'react';

interface SubmissionSuccessScreenProps {
  characterName: string;
  onStartOver: () => void;
}

export const SubmissionSuccessScreen: React.FC<SubmissionSuccessScreenProps> = ({ characterName, onStartOver }) => {
  return (
    <div className="text-center animate-fadeIn">
      <h2 className="text-3xl font-bold text-green-400 mb-4">🎉 신청 완료! 🎉</h2>
      <p className="text-xl text-slate-200 mb-6">
        {characterName} 굿즈 신청이 성공적으로 완료되었습니다! <br/>
        참여해주셔서 감사합니다. 빠른 시일 내에 연락드리겠습니다.
      </p>
      <div className="my-8">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-24 w-24 text-green-400 mx-auto animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      <button
        onClick={onStartOver}
        className="px-8 py-3 bg-gradient-to-r from-sky-500 to-indigo-500 hover:from-sky-600 hover:to-indigo-600 text-white font-bold rounded-lg shadow-md hover:shadow-lg transform hover:scale-105 transition-all duration-300 text-lg"
      >
        처음으로 돌아가기
      </button>
    </div>
  );
};
