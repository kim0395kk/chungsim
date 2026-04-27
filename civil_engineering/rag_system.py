# civil_engineering/rag_system.py
"""
토목직 특화 RAG (Retrieval-Augmented Generation) 시스템
"""

import os
import json
import pickle
import re
from typing import List, Dict, Optional, Tuple
import numpy as np

# 벡터 임베딩 (Optional)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available")

# 벡터 DB (Optional)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available")


class CivilEngineeringRAG:
    """
    토목직 산업단지 전문 RAG 시스템
    """
    
    def __init__(self, 
                 complexes_data: List[Dict],
                 embedding_model: str = "distiluse-base-multilingual-cased-v2",
                 vector_db_path: str = "data/vector_db"):
        """
        초기화
        
        Args:
            complexes_data: 파싱된 산업단지 데이터 리스트
            embedding_model: 임베딩 모델명
            vector_db_path: 벡터 DB 저장 경로
        """
        self.complexes_data = complexes_data
        self.vector_db_path = vector_db_path
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.model_name = embedding_model
        
        # 임베딩 모델 로드
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
            except Exception as e:
                print(f"Failed to load sentence-transformer: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            print("(!) 임베딩 모델 없음 - 키워드 검색만 사용")
        
        # 데이터 준비
        self._prepare_chunks()
        
        # 벡터 DB 로드 또는 생성
        if os.path.exists(f"{vector_db_path}/index.faiss"):
            self._load_index()
        else:
            self._build_index()
    
    def _prepare_chunks(self):
        """청크 생성"""
        from civil_engineering.data_parser import create_search_chunks
        
        for complex_data in self.complexes_data:
            chunks = create_search_chunks(complex_data)
            self.chunks.extend(chunks)
        
        print(f"(+) {len(self.chunks)}개 청크 생성 완료")
    
    def _build_index(self):
        """벡터 인덱스 구축"""
        if not self.embedding_model:
            print("(!) 임베딩 모델 없음 - 인덱스 구축 스킵")
            return
        
        print("(*) 벡터 인덱스 구축 중...")
        
        try:
            # 텍스트 임베딩 생성
            texts = [chunk['text'] for chunk in self.chunks]
            self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # FAISS 인덱스 생성
            if FAISS_AVAILABLE:
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(self.embeddings.astype('float32'))
                
                # 저장
                os.makedirs(self.vector_db_path, exist_ok=True)
                faiss.write_index(self.index, f"{self.vector_db_path}/index.faiss")
                
                with open(f"{self.vector_db_path}/chunks.pkl", "wb") as f:
                    pickle.dump(self.chunks, f)
                
                print(f"(+) 벡터 인덱스 저장 완료: {self.vector_db_path}")
            else:
                print("(!) FAISS 없음 - 인덱스 저장 스킵")
        except Exception as e:
            print(f"(!) 인덱스 구축 중 오류: {e}")
    
    def _load_index(self):
        """저장된 인덱스 로드"""
        if not FAISS_AVAILABLE:
            return
        
        try:
            self.index = faiss.read_index(f"{self.vector_db_path}/index.faiss")
            
            with open(f"{self.vector_db_path}/chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            
            print(f"(+) 벡터 인덱스 로드 완료: {len(self.chunks)}개 청크")
        except Exception as e:
            print(f"(!) 인덱스 로드 실패 (재구축 시도): {e}")
            self._build_index()
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        쿼리에 대한 관련 청크 검색
        
        Args:
            query: 검색 질문
            top_k: 반환할 상위 결과 개수
        
        Returns:
            [(청크, 유사도 점수), ...] 리스트
        """
        
        # 벡터 검색 (임베딩 모델 있을 때)
        if self.embedding_model and self.index:
            try:
                query_embedding = self.embedding_model.encode([query])
                distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.chunks) and idx >= 0:
                        chunk = self.chunks[idx]
                        score = 1 / (1 + distances[0][i])  # 거리를 유사도로 변환
                        results.append((chunk, score))
                
                return results
            except Exception as e:
                print(f"Vector search failed: {e}")
                return self._keyword_search(query, top_k)
        
        # Fallback: 키워드 검색
        else:
            return self._keyword_search(query, top_k)
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """키워드 기반 검색 (Fallback)"""
        keywords = query.split()
        
        scored_chunks = []
        for chunk in self.chunks:
            score = sum(1 for kw in keywords if kw in chunk['text'])
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # 점수 순 정렬
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # 점수 정규화
        max_score = scored_chunks[0][1] if scored_chunks else 1
        normalized = [(chunk, score / max_score) for chunk, score in scored_chunks[:top_k]]
        
        return normalized

    def _tokenize_query(self, question: str) -> List[str]:
        """질문에서 의미 있는 키워드를 추출."""
        stop = {
            "무엇", "어떻게", "관련", "내용", "문의", "질문", "기준", "확인",
            "해줘", "주세요", "있나요", "인가요", "대한", "에서", "으로", "하는",
            "토목", "특화", "규정", "매뉴얼"
        }
        raw = re.findall(r"[0-9A-Za-z가-힣]{2,}", question or "")
        toks: List[str] = []
        for t in raw:
            t = t.strip().lower()
            if not t or t in stop:
                continue
            toks.append(t)
        # 순서 보존 중복 제거
        out: List[str] = []
        seen = set()
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _extract_parsed_facts(
        self,
        search_results: List[Tuple[Dict, float]],
        question: str,
        max_items: int = 4,
    ) -> List[Dict]:
        """질문 연관도가 높은 청크에서만 핵심 정보를 구조화해 추출."""
        candidates: List[Tuple[float, Dict]] = []
        seen_keys = set()
        q_tokens = self._tokenize_query(question)

        for chunk, score in search_results:
            if score < 0.03:
                continue
            text = chunk.get("text", "") or ""
            source = chunk.get("display_name") or chunk.get("complex_name") or "출처 미상"

            def _find(pat: str) -> Optional[str]:
                m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    return (m.group(1) or "").strip()
                return None

            location = _find(r"(?:위치|소재지)\s*[:：]\s*([^\n]+)")
            status = _find(r"(?:상태|사업\s*상태)\s*[:：]\s*([^\n]+)")
            area = _find(r"(?:면적|부지면적)\s*[:：]\s*([^\n]+)")
            budget = _find(r"(?:예산|사업비|총사업비)\s*[:：]\s*([^\n]+)")
            period = _find(r"(?:기간|사업기간)\s*[:：]\s*([^\n]+)")
            developer = _find(r"(?:시행자|사업시행자)\s*[:：]\s*([^\n]+)")
            industries = _find(r"(?:유치업종|업종)\s*[:：]\s*([^\n]+)")

            if not any([location, status, area, budget, period, developer, industries]):
                continue

            dedup_key = (source, location, status, area, budget, period, developer, industries)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            doc_blob = " ".join(
                [
                    source,
                    text[:2000],
                    location or "",
                    status or "",
                    area or "",
                    budget or "",
                    period or "",
                    developer or "",
                    industries or "",
                ]
            ).lower()
            hit = sum(1 for tok in q_tokens if tok in doc_blob)

            # 질문 키워드와 매칭되는 항목만 우선 선별
            if q_tokens and hit == 0:
                continue

            relevance = float(score) + (0.05 * hit)
            candidates.append(
                (
                    relevance,
                    {
                        "source": source,
                        "location": location,
                        "status": status,
                        "area": area,
                        "budget": budget,
                        "period": period,
                        "developer": developer,
                        "industries": industries,
                    },
                )
            )

        # 키워드 매칭이 하나도 없으면 검색 점수 상위 1개만 최소 반환
        if not candidates and search_results:
            top_chunk, _top_score = search_results[0]
            source = top_chunk.get("display_name") or top_chunk.get("complex_name") or "출처 미상"
            text = top_chunk.get("text", "") or ""

            def _find(pat: str) -> Optional[str]:
                m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    return (m.group(1) or "").strip()
                return None

            candidates.append(
                (
                    0.0,
                    {
                        "source": source,
                        "location": _find(r"(?:위치|소재지)\s*[:：]\s*([^\n]+)"),
                        "status": _find(r"(?:상태|사업\s*상태)\s*[:：]\s*([^\n]+)"),
                        "area": _find(r"(?:면적|부지면적)\s*[:：]\s*([^\n]+)"),
                        "budget": _find(r"(?:예산|사업비|총사업비)\s*[:：]\s*([^\n]+)"),
                        "period": _find(r"(?:기간|사업기간)\s*[:：]\s*([^\n]+)"),
                        "developer": _find(r"(?:시행자|사업시행자)\s*[:：]\s*([^\n]+)"),
                        "industries": _find(r"(?:유치업종|업종)\s*[:：]\s*([^\n]+)"),
                    },
                )
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in candidates[:max_items]]

    def _facts_to_table_rows(self, parsed_facts: List[Dict], max_rows: int = 8) -> List[Dict]:
        """파싱 결과를 고정 스키마 테이블 행으로 변환."""
        rows: List[Dict] = []
        for fact in (parsed_facts or [])[:max_rows]:
            rows.append(
                {
                    "출처": fact.get("source") or "-",
                    "위치": fact.get("location") or "-",
                    "상태": fact.get("status") or "-",
                    "면적": fact.get("area") or "-",
                    "예산": fact.get("budget") or "-",
                    "기간": fact.get("period") or "-",
                    "시행자": fact.get("developer") or "-",
                    "유치업종": fact.get("industries") or "-",
                }
            )
        return rows

    def _enforce_non_location_only_answer(self, answer: str, parsed_facts: List[Dict]) -> Tuple[str, Dict]:
        """
        위치만 언급하는 답변을 방지하기 위한 후처리.
        파싱 가능한 필드가 충분한데 답변이 위치 중심이면 보강 요약을 자동 추가.
        """
        quality = {"location_only_risk": False, "augmented": False}
        if not parsed_facts:
            return answer, quality

        available_fields = 0
        for key in ["status", "area", "budget", "period", "developer", "industries"]:
            if any((f.get(key) or "").strip() for f in parsed_facts):
                available_fields += 1

        if available_fields < 1:
            return answer, quality

        # 답변 내 구조 신호 확인
        signal_keywords = ["면적", "예산", "기간", "상태", "시행자", "유치업종"]
        present_signals = sum(1 for kw in signal_keywords if kw in (answer or ""))

        if present_signals >= 2:
            return answer, quality

        quality["location_only_risk"] = True
        top = parsed_facts[0]
        supplement = [
            "",
            "### 🔧 자동 보강 요약",
            "- 위치 외 핵심 항목도 함께 확인해 주세요.",
        ]
        if top.get("status"):
            supplement.append(f"- 상태: {top['status']}")
        if top.get("area"):
            supplement.append(f"- 면적: {top['area']}")
        if top.get("budget"):
            supplement.append(f"- 예산: {top['budget']}")
        if top.get("period"):
            supplement.append(f"- 기간: {top['period']}")
        if top.get("developer"):
            supplement.append(f"- 시행자: {top['developer']}")
        if top.get("industries"):
            supplement.append(f"- 유치업종: {top['industries']}")

        quality["augmented"] = True
        return (answer or "").rstrip() + "\n" + "\n".join(supplement), quality
    
    def answer_question(self, question: str, llm_service, top_k: int = 3) -> Dict:
        """
        질문에 대한 답변 생성 (RAG)
        
        Args:
            question: 사용자 질문
            llm_service: LLM 서비스 인스턴스 (generate_text 메서드 보유)
            top_k: 컨텍스트로 사용할 청크 수
        
        Returns:
            {
                "answer": "답변 텍스트",
                "sources": ["출처1", "출처2"],
                "confidence": 0.0~1.0
            }
        """
        
        # 1. 관련 청크 검색
        search_results = self.search(question, top_k)
        
        if not search_results:
            # 검색 결과가 없으면 LLM 지식으로 답변 시도 (단, 경고 메시지 포함)
            print(f"(!) 검색 결과 없음 -> LLM 일반 지식 활용: {question}")
            
            fallback_prompt = f"""
당신은 토목 행정 전문가입니다. 
사용자의 질문에 대해 당신이 가진 일반적인 토목/행정 지식을 바탕으로 친절하게 답변해 주세요.

[중요 제약사항]
답변의 맨 앞부분에 반드시 다음 경고 문구를 포함해야 합니다.
"⚠️ **내부 규정이나 매뉴얼에서 관련 내용을 찾을 수 없습니다.** 아래 내용은 일반적인 토목 지식에 기반한 답변이므로, 정확한 업무 처리를 위해서는 반드시 관련 규정을 별도로 확인하시기 바랍니다."

[질문]
{question}
"""
            # LLM 호출
            try:
                llm_answer = llm_service.generate_text(fallback_prompt)
            except Exception as e:
                llm_answer = "죄송합니다. 관련 정보를 찾을 수 없으며, 일반 지식 답변 생성 중 오류가 발생했습니다."
                print(f"(!) LLM Fallback Error: {e}")

            return {
                "answer": llm_answer,
                "summary": llm_answer[:220],
                "sources": ["⚠️ 일반 지식 (내부 문서 없음)"],
                "confidence": 0.1,
                "raw_chunks": [],
                "parsed_facts": [],
                "fact_rows": [],
                "quality": {"location_only_risk": False, "augmented": False},
                "retrieval_meta": {
                    "fallback_general_knowledge": True,
                    "used_chunks": 0,
                    "source_count": 1,
                    "fact_count": 0,
                },
            }
        
        # 2. 컨텍스트 구성
        context_parts = []
        sources = []
        
        for chunk, score in search_results:
            # display_name 우선 사용 (섹션 정보 포함)
            source_name = chunk.get('display_name', '')
            if not source_name:
                if chunk.get('metadata', {}).get('type') == 'manual':
                    source_name = f"{chunk['complex_name']} (매뉴얼)"
                else:
                    source_name = f"{chunk['complex_name']} ({chunk.get('type', 'general')})"
            
            context_parts.append(f"[출처: {source_name}]\n{chunk['text']}")
            sources.append(source_name)
        
        context = "\n\n---\n\n".join(context_parts)
        parsed_facts = self._extract_parsed_facts(search_results, question=question)
        fact_rows = self._facts_to_table_rows(parsed_facts)
        
        # 3. LLM에 질문
        prompt = f"""당신은 토목 행정 전문가입니다. 다음 [참고 자료]를 바탕으로 공무원의 질문에 답변하세요.

[참고 자료]
{context}

[질문]
{question}

[답변 규칙]
- 우선적으로 [참고 자료]에 있는 정보를 사용해 답변하세요.
- 질문에 답할 때 위치만 말하지 말고, 가능하면 면적/예산/기간/상태/시행자/유치업종도 함께 정리하세요.
- 만약 [참고 자료]에 해당 내용이 없다면 "자료에 없음"이라고 하지 말고, 당신의 일반적인 토목/건설 지식을 활용해 답변하세요.
- 단, 일반 지식을 사용할 경우 반드시 답변 시작 부분에 "⚠️ **[일반 지식] 제공된 문서에 해당 내용이 없어 일반적인 토목 지식으로 답변합니다.**" 라는 경고 문구를 붙이세요.
- 구체적인 숫자, 날짜, 명칭 정확히 인용
- 출처 문서명을 반드시 언급
- 간결하고 명확하게 답변
- 한국어로 답변

답변:
"""
        
        try:
            # llm_service가 generate_text 메서드를 가지고 있다고 가정
            answer = llm_service.generate_text(prompt)
            answer, quality = self._enforce_non_location_only_answer(answer, parsed_facts)
            
            # 평균 신뢰도 계산
            avg_confidence = sum(score for _, score in search_results) / len(search_results)
            
            return {
                "answer": answer,
                "summary": (answer or "").strip().split("\n")[0][:220],
                "sources": list(set(sources)),  # 중복 제거
                "confidence": avg_confidence,
                "raw_chunks": [chunk for chunk, _ in search_results],
                "parsed_facts": parsed_facts,
                "fact_rows": fact_rows,
                "quality": quality,
                "retrieval_meta": {
                    "fallback_general_knowledge": False,
                    "used_chunks": len(search_results),
                    "source_count": len(list(set(sources))),
                    "fact_count": len(parsed_facts),
                },
            }
            
        except Exception as e:
            return {
                "answer": f"답변 생성 중 오류: {str(e)}",
                "summary": "답변 생성 중 오류",
                "sources": sources,
                "confidence": 0.0,
                "raw_chunks": [],
                "parsed_facts": parsed_facts,
                "fact_rows": fact_rows,
                "quality": {"location_only_risk": False, "augmented": False},
                "retrieval_meta": {
                    "fallback_general_knowledge": False,
                    "used_chunks": len(search_results),
                    "source_count": len(list(set(sources))),
                    "fact_count": len(parsed_facts),
                },
            }


# ===== 편의 함수 =====

def load_rag_system(data_path: str = "data/parsed_complexes.json",
                    vector_db_path: str = "data/vector_db") -> Optional[CivilEngineeringRAG]:
    """
    RAG 시스템 로드
    
    Args:
        data_path: 파싱된 데이터 JSON 경로
        vector_db_path: 벡터 DB 경로
    
    Returns:
        CivilEngineeringRAG 인스턴스 또는 None
    """
    
    try:
        # 1. JSON 데이터 로드 시도
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                complexes_data = json.load(f)
        else:
            # JSON 없으면 MD 파일들 찾아서 파싱
            import glob
            from civil_engineering.data_parser import parse_all_md_files
            
            # llm_ready_docs 폴더 사용
            md_files = glob.glob(r"c:\Users\Mr Kim\Desktop\chungsim\llm_ready_docs\*.md")
            if not md_files:
                # 상대 경로 fallback
                md_files = glob.glob("llm_ready_docs/*.md")
            
            if not md_files:
                print("(!) MD 파일을 찾을 수 없습니다.")
                return None
                
            print(f"(+) MD 파일 {len(md_files)}개 파싱 시작...")
            complexes_data = parse_all_md_files(md_files)
            
            # 파싱 결과 저장 (캐싱)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(complexes_data, f, ensure_ascii=False, indent=2)
            print("(+) 파싱 데이터 캐싱 완료")
        
        # 2. RAG 시스템 초기화
        rag = CivilEngineeringRAG(complexes_data, vector_db_path=vector_db_path)
        
        return rag
        
    except Exception as e:
        print(f"(!) RAG 시스템 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
