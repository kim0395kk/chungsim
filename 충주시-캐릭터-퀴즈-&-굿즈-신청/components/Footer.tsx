
import React from 'react';
import { NIGHT_MARKET_NAME } from '../constants';

const Footer: React.FC = () => {
  return (
    <footer className="w-full max-w-2xl text-center py-6 mt-8">
      <p className="text-sm text-slate-400">
        &copy; {new Date().getFullYear()} {NIGHT_MARKET_NAME} &amp; 충주시. All rights reserved.
      </p>
      <p className="text-xs text-slate-500 mt-1">
        본 이벤트는 {NIGHT_MARKET_NAME} 활성화를 위해 제작되었습니다.
      </p>
    </footer>
  );
};

export default Footer;
