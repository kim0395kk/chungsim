
export interface QuizQuestion {
  id: number;
  questionText: string;
  options: string[];
  correctAnswer: string;
  characterImage?: string;
}

export interface MerchandiseFormData {
  name: string;
  phone: string;
  address: string;
  zipCode: string;
  agreedToTerms: boolean;
}

export enum AppPhase {
  WELCOME,
  QUIZ,
  RESULT,
  FORM,
  SUBMITTED,
}
