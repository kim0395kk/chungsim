
import { QuizQuestion } from './types';

export const CHARACTER_NAME = "충심이";
export const NIGHT_MARKET_NAME = "누리야시장";
export const MAIN_CHARACTER_IMAGE_URL = "https://picsum.photos/seed/chungsimiMain/400/400"; // A cute, friendly character

export const QUIZ_QUESTIONS: QuizQuestion[] = [
  {
    id: 1,
    questionText: `${CHARACTER_NAME}의 주 활동 무대는 어디일까요?`,
    options: [`충주호`, `${NIGHT_MARKET_NAME}`, `수안보온천`, `탄금대`],
    correctAnswer: `${NIGHT_MARKET_NAME}`,
    characterImage: "https://picsum.photos/seed/quiz1Chungsimi/350/250"
  },
  {
    id: 2,
    questionText: `${CHARACTER_NAME}가 가장 좋아하는 ${NIGHT_MARKET_NAME} 간식은 무엇일까요?`,
    options: [`사과파이`, `매콤 닭꼬치`, `달콤 솜사탕`, `고소한 붕어빵`],
    correctAnswer: `매콤 닭꼬치`,
    characterImage: "https://picsum.photos/seed/quiz2Chungsimi/350/250"
  },
  {
    id: 3,
    questionText: `${CHARACTER_NAME} 캐릭터의 성격으로 가장 어울리는 것은 무엇일까요?`,
    options: [`활발하고 친근함`, `조용하고 신비로움`, `시크하고 도도함`, `장난기 넘치는 악동`],
    correctAnswer: `활발하고 친근함`,
    characterImage: "https://picsum.photos/seed/quiz3Chungsimi/350/250"
  },
  {
    id: 4,
    questionText: `${NIGHT_MARKET_NAME}에서 ${CHARACTER_NAME}을 만나면 무엇을 해야 할까요?`,
    options: [`숨는다`, `반갑게 인사한다`, `간식을 사준다 (캐릭터에게는 마음만!)`, `사진을 찍어 SNS에 올린다`],
    correctAnswer: `반갑게 인사한다`,
    characterImage: "https://picsum.photos/seed/quiz4Chungsimi/350/250"
  }
];

export const PASSING_SCORE_PERCENTAGE = 0.75; // 3 out of 4 questions (75%) to pass
