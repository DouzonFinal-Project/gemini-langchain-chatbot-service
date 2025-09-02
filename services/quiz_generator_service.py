# services/quiz_generator_service.py
from typing import List, Dict, Optional
import re
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage


class QuizGeneratorService:
    def __init__(self):
        """Quiz Generator Service 초기화"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.7,
            max_output_tokens=3000
        )
        
        # 문제 유형 정의
        self.PROBLEM_TYPES = {
            "content_understanding": {
                "name": "내용 이해",
                "description": "지문의 내용을 정확히 이해했는지 확인",
                "prompt": "지문의 내용을 바탕으로 사실 확인이나 내용 이해를 묻는 문제"
            },
            "main_idea": {
                "name": "주제 파악",
                "description": "글의 중심 내용이나 주제 찾기",
                "prompt": "지문의 중심 내용, 주제, 글쓴이의 의도를 파악하는 문제"
            },
            "vocabulary": {
                "name": "어휘 의미",
                "description": "지문에 나온 어휘의 뜻이나 쓰임 이해",
                "prompt": "지문에 사용된 단어나 표현의 의미, 쓰임을 묻는 문제"
            },
            "text_structure": {
                "name": "글의 구조",
                "description": "글의 전개 방식이나 구성 파악",
                "prompt": "글의 구성, 전개 방식, 문단 간의 관계를 파악하는 문제"
            },
            "inference": {
                "name": "추론 및 상상",
                "description": "글 내용을 바탕으로 추론하거나 상상하기",
                "prompt": "지문 내용을 바탕으로 추론하거나 앞뒤 상황을 상상하는 문제"
            },
            "expression_technique": {
                "name": "표현 기법",
                "description": "글에 사용된 표현 방법이나 기법 파악",
                "prompt": "지문에 사용된 표현 기법, 문체, 서술 방법을 파악하는 문제"
            }
        }
    
    def get_problem_types(self) -> Dict[str, Dict]:
        """문제 유형 목록 반환"""
        return self.PROBLEM_TYPES
    
    def validate_text_input(self, text: str) -> Dict[str, any]:
        """지문 유효성 검사"""
        if not text or not text.strip():
            return {"valid": False, "error": "지문을 입력해주세요."}
        
        if len(text.strip()) < 50:
            return {"valid": False, "error": "더 긴 지문을 입력해주세요. (최소 50자 이상)"}
        
        return {"valid": True, "char_count": len(text.strip())}
    
    def validate_problem_types(self, selected_types: List[str]) -> Dict[str, any]:
        """선택된 문제 유형 유효성 검사"""
        if not selected_types:
            return {"valid": False, "error": "최소 하나의 문제 유형을 선택해주세요."}
        
        invalid_types = [t for t in selected_types if t not in self.PROBLEM_TYPES]
        if invalid_types:
            return {"valid": False, "error": f"유효하지 않은 문제 유형: {', '.join(invalid_types)}"}
        
        return {"valid": True}
    
    async def generate_quiz_problems(
        self, 
        text: str, 
        selected_types: List[str], 
        num_problems: int = 3
    ) -> Dict[str, any]:
        """
        지문을 바탕으로 오지선다 문제 생성
        
        Args:
            text: 지문 내용
            selected_types: 선택된 문제 유형 리스트
            num_problems: 생성할 문제 수 (기본 3개)
        
        Returns:
            생성된 문제 딕셔너리
        """
        try:
            # 입력 유효성 검사
            text_validation = self.validate_text_input(text)
            if not text_validation["valid"]:
                return {"success": False, "error": text_validation["error"]}
            
            types_validation = self.validate_problem_types(selected_types)
            if not types_validation["valid"]:
                return {"success": False, "error": types_validation["error"]}
            
            # 프롬프트 생성
            prompt = self._create_quiz_prompt(text, selected_types, num_problems)
            
            # Gemini API 호출
            message = HumanMessage(content=prompt)
            response = await self.llm.ainvoke([message])
            
            # 응답 파싱
            problems = self._parse_problems(response.content)
            
            if not problems:
                return {"success": False, "error": "문제를 올바르게 생성하지 못했습니다. 다시 시도해주세요."}
            
            return {
                "success": True,
                "problems": problems,
                "stats": self.get_quiz_statistics(text, problems)
            }
            
        except Exception as e:
            return {"success": False, "error": f"문제 생성 중 오류가 발생했습니다: {str(e)}"}
    
    def _create_quiz_prompt(self, text: str, selected_types: List[str], num_problems: int) -> str:
        """퀴즈 생성을 위한 프롬프트 생성"""
        type_descriptions = '\n'.join([
            f"- {self.PROBLEM_TYPES[type_key]['name']}: {self.PROBLEM_TYPES[type_key]['prompt']}"
            for type_key in selected_types
        ])
        
        prompt = f"""다음 지문을 바탕으로 초등학교 5-6학년 수준의 오지선다 문제 {num_problems}개를 만들어주세요.

지문:
{text}

선택된 문제 유형:
{type_descriptions}

요구사항:
1. 반드시 오지선다 문제로 만들어주세요 (5개의 선택지: ①, ②, ③, ④, ⑤)
2. 선택된 문제 유형 중에서 적절히 섞어서 {num_problems}개 문제를 만들어주세요
3. 각 문제는 지문의 내용을 정확히 이해했는지 확인하는 문제여야 합니다
4. 정답은 반드시 지문에서 명확히 확인할 수 있어야 합니다
5. 오답 선택지는 그럴듯하지만 틀린 내용이어야 합니다
6. 초등학교 5-6학년이 이해할 수 있는 수준으로 만들어주세요

출력 형식 (정확히 이 형식을 따라주세요):
[문제1]
유형: (문제 유형명)
질문: (문제 내용)
①: (선택지 1)
②: (선택지 2)  
③: (선택지 3)
④: (선택지 4)
⑤: (선택지 5)
정답: (정답 번호)

[문제2]
유형: (문제 유형명)
질문: (문제 내용)
①: (선택지 1)
②: (선택지 2)
③: (선택지 3)
④: (선택지 4)
⑤: (선택지 5)
정답: (정답 번호)

[문제3]
유형: (문제 유형명)
질문: (문제 내용)
①: (선택지 1)
②: (선택지 2)
③: (선택지 3)
④: (선택지 4)
⑤: (선택지 5)
정답: (정답 번호)"""

        return prompt
    
    def _parse_problems(self, generated_text: str) -> List[Dict]:
        """생성된 텍스트에서 문제 파싱"""
        problems = []
        problem_blocks = re.split(r'\[문제\d+\]', generated_text)
        problem_blocks = [block.strip() for block in problem_blocks if block.strip()]
        
        for index, block in enumerate(problem_blocks):
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            
            problem_data = {
                "number": index + 1,
                "type": "",
                "question": "",
                "choices": [],
                "answer": ""
            }
            
            for line in lines:
                if line.startswith('유형:'):
                    problem_data["type"] = line.replace('유형:', '').strip()
                elif line.startswith('질문:'):
                    problem_data["question"] = line.replace('질문:', '').strip()
                elif re.match(r'^[①②③④⑤]', line):
                    choice_text = re.sub(r'^[①②③④⑤]:?\s*', '', line)
                    problem_data["choices"].append(choice_text)
                elif line.startswith('정답:'):
                    problem_data["answer"] = line.replace('정답:', '').strip()
            
            # 문제가 완전히 파싱된 경우에만 추가
            if (problem_data["question"] and 
                len(problem_data["choices"]) == 5 and 
                problem_data["answer"]):
                problems.append(problem_data)
        
        return problems
    
    def format_problem_for_display(self, problem: Dict) -> Dict:
        """화면 표시용 문제 포맷팅"""
        choices_with_symbols = []
        symbols = ['①', '②', '③', '④', '⑤']
        
        for i, choice in enumerate(problem["choices"]):
            choices_with_symbols.append(f"{symbols[i]} {choice}")
        
        return {
            "number": problem["number"],
            "type": problem["type"],
            "question": problem["question"],
            "choices": choices_with_symbols,
            "answer": problem["answer"]
        }
    
    def get_quiz_statistics(self, text: str, problems: List[Dict]) -> Dict[str, any]:
        """
        퀴즈 통계 정보 생성
        
        Args:
            text: 원본 지문
            problems: 생성된 문제 목록
        
        Returns:
            통계 정보
        """
        type_counts = {}
        for problem in problems:
            problem_type = problem.get("type", "기타")
            type_counts[problem_type] = type_counts.get(problem_type, 0) + 1
        
        return {
            "char_count": len(text.strip()),
            "problem_count": len(problems),
            "type_distribution": type_counts,
            "average_choices_per_problem": 5  # 항상 5개 선택지
        }