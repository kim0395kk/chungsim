
import React from 'react';

interface HeaderProps {
  characterName: string;
}

const Header: React.FC<HeaderProps> = ({ characterName }) => {
  return (
    <header className="w-full max-w-2xl text-center py-6">
      <h1 className="text-3xl sm:text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 via-orange-500 to-red-500">
        {characterName} 캐릭터 퀴즈 & 굿즈샵
      </h1>
      <p className="text-slate-300 mt-2 text-sm sm:text-base">누리야시장의 마스코트 {characterName}와 함께 즐거운 퀴즈도 풀고 한정판 굿즈도 받아가세요!</p>
    </header>
  );
};

export default Header;
