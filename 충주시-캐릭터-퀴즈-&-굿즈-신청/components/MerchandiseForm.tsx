
import React, { useState } from 'react';
import { MerchandiseFormData } from '../types';

interface MerchandiseFormProps {
  onSubmit: (formData: MerchandiseFormData) => void;
}

// Define a type for form errors where each key from MerchandiseFormData can have a string error message
type FormErrors = {
  [K in keyof MerchandiseFormData]?: string;
};

const MerchandiseForm: React.FC<MerchandiseFormProps> = ({ onSubmit }) => {
  const [formData, setFormData] = useState<MerchandiseFormData>({
    name: '',
    phone: '',
    address: '',
    zipCode: '',
    agreedToTerms: false,
  });
  const [errors, setErrors] = useState<FormErrors>({}); // Use the new FormErrors type

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    if (type === 'checkbox') {
        const {checked} = e.target as HTMLInputElement;
        setFormData(prev => ({ ...prev, [name]: checked }));
    } else {
        setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {}; // Use the new FormErrors type
    if (!formData.name.trim()) newErrors.name = '이름을 입력해주세요.';
    if (!formData.phone.match(/^010-\d{4}-\d{4}$/) && !formData.phone.match(/^010\d{8}$/)) {
        newErrors.phone = '올바른 휴대폰 번호 형식(010-1234-5678 또는 01012345678)으로 입력해주세요.';
    }
    if (!formData.address.trim()) newErrors.address = '주소를 입력해주세요.';
    if (!formData.zipCode.match(/^\d{5}$/)) newErrors.zipCode = '올바른 우편번호(5자리)를 입력해주세요.';
    if (!formData.agreedToTerms) newErrors.agreedToTerms = '개인정보 수집 및 이용에 동의해주세요.';
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit(formData);
    }
  };

  const inputClass = "w-full p-3 bg-slate-700 border border-slate-600 rounded-md focus:ring-2 focus:ring-yellow-500 focus:border-yellow-500 outline-none transition-colors text-slate-100 placeholder-slate-400";
  const errorClass = "text-red-400 text-sm mt-1";

  return (
    <div className="animate-fadeIn w-full">
      <h2 className="text-2xl sm:text-3xl font-bold text-center text-yellow-400 mb-8">✨ 충심이 굿즈 신청 ✨</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="name" className="block text-sm font-medium text-slate-300 mb-1">이름</label>
          <input type="text" name="name" id="name" value={formData.name} onChange={handleChange} className={inputClass} placeholder="홍길동" />
          {errors.name && <p className={errorClass}>{errors.name}</p>}
        </div>
        <div>
          <label htmlFor="phone" className="block text-sm font-medium text-slate-300 mb-1">휴대폰 번호</label>
          <input type="tel" name="phone" id="phone" value={formData.phone} onChange={handleChange} className={inputClass} placeholder="010-1234-5678" />
          {errors.phone && <p className={errorClass}>{errors.phone}</p>}
        </div>
        <div>
          <label htmlFor="zipCode" className="block text-sm font-medium text-slate-300 mb-1">우편번호</label>
          <input type="text" name="zipCode" id="zipCode" value={formData.zipCode} onChange={handleChange} className={inputClass} placeholder="12345 (5자리)" />
          {errors.zipCode && <p className={errorClass}>{errors.zipCode}</p>}
        </div>
        <div>
          <label htmlFor="address" className="block text-sm font-medium text-slate-300 mb-1">주소</label>
          <input type="text" name="address" id="address" value={formData.address} onChange={handleChange} className={inputClass} placeholder="상세 주소를 입력해주세요." />
          {errors.address && <p className={errorClass}>{errors.address}</p>}
        </div>
        <div className="flex items-start">
            <div className="flex items-center h-5">
                 <input
                    id="agreedToTerms"
                    name="agreedToTerms"
                    type="checkbox"
                    checked={formData.agreedToTerms}
                    onChange={handleChange}
                    className="focus:ring-yellow-500 h-4 w-4 text-yellow-600 border-slate-500 rounded bg-slate-700"
                />
            </div>
            <div className="ml-3 text-sm">
                <label htmlFor="agreedToTerms" className="font-medium text-slate-300">
                    개인정보 수집 및 이용에 동의합니다.
                </label>
                <p className="text-slate-400 text-xs">(굿즈 발송 목적으로만 사용되며, 이벤트 종료 후 즉시 파기됩니다.)</p>
            </div>
        </div>
        {errors.agreedToTerms && <p className={errorClass}>{errors.agreedToTerms}</p>}

        <button 
          type="submit" 
          className="w-full px-8 py-3 bg-gradient-to-r from-orange-500 to-yellow-500 hover:from-orange-600 hover:to-yellow-600 text-slate-900 font-bold rounded-lg shadow-md hover:shadow-lg transform hover:scale-105 transition-all duration-300 text-lg"
        >
          굿즈 신청 완료하기
        </button>
      </form>
    </div>
  );
};

export default MerchandiseForm;
