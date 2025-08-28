# routers/exam_generator.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, AsyncGenerator
import json
import logging

from services.passage_generator_service import (
    process_and_store_document,
    generate_passage_sync,
    generate_passage_streaming
)
from services.quiz_generator_service import QuizGeneratorService

router = APIRouter()
logger = logging.getLogger(__name__)

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str

class PassageGenerationRequest(BaseModel):
    document_id: str
    user_prompt: str

# 요청/응답 모델 정의
class QuizGenerationRequest(BaseModel):
    text: str = Field(..., description="문제를 생성할 지문")
    selected_types: List[str] = Field(..., description="선택된 문제 유형 목록")
    num_problems: int = Field(3, description="생성할 문제 수", ge=1, le=10)

class QuizProblem(BaseModel):
    number: int
    type: str
    question: str
    choices: List[str]
    answer: str

class QuizGenerationResponse(BaseModel):
    success: bool
    problems: Optional[List[QuizProblem]] = None
    stats: Optional[Dict] = None
    error: Optional[str] = None

class ProblemTypesResponse(BaseModel):
    problem_types: Dict[str, Dict[str, str]]

# 서비스 인스턴스 생성
quiz_service = QuizGeneratorService()

@router.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    try:
        file_content = await file.read()
        info = process_and_store_document(file_content, file.filename)
        return {
            "document_id": info["document_id"],
            "filename": file.filename,
            "status": "completed",
            "message": f"문서 처리 완료. {info['chunk_count']}개 청크로 분할되어 저장됨"
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.exception("문서 업로드 실패")
        raise HTTPException(status_code=500, detail="문서 처리 중 오류가 발생했습니다.")

@router.post("/generate-passage")
async def generate_passage_streaming_route(request: PassageGenerationRequest):
    # 스트리밍: services가 주는 async generator의 dict들을 SSE 형식으로 래핑
    async def event_generator() -> AsyncGenerator[str, None]:
        async for chunk in generate_passage_streaming(request.document_id, request.user_prompt):
            yield f"data: {json.dumps(chunk)}\n\n"
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache", "Connection":"keep-alive"}
    )

@router.post("/generate-passage-sync")
async def generate_passage_sync_route(request: PassageGenerationRequest):
    try:
        resp = await generate_passage_sync(request.document_id, request.user_prompt)
        return resp
    except RuntimeError as re:
        raise HTTPException(status_code=404, detail=str(re))
    except Exception:
        logger.exception("지문 생성 실패")
        raise HTTPException(status_code=500, detail="지문 생성 중 오류 발생")

@router.get("/problem-types", response_model=ProblemTypesResponse)
async def get_problem_types():
    """
    사용 가능한 문제 유형 목록을 반환합니다.
    """
    try:
        problem_types = quiz_service.get_problem_types()
        return ProblemTypesResponse(problem_types=problem_types)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문제 유형을 가져오는 중 오류가 발생했습니다: {str(e)}")

@router.post("/generate-quiz", response_model=QuizGenerationResponse)
async def generate_quiz(request: QuizGenerationRequest):
    """
    지문을 바탕으로 오지선다 문제를 생성합니다.
    
    Args:
        request: 문제 생성 요청 (지문, 선택된 문제 유형, 문제 수)
    
    Returns:
        생성된 문제 목록과 통계 정보
    """
    try:
        result = await quiz_service.generate_quiz_problems(
            text=request.text,
            selected_types=request.selected_types,
            num_problems=request.num_problems
        )
        
        if not result["success"]:
            return QuizGenerationResponse(
                success=False,
                error=result["error"]
            )
        
        # 문제 데이터를 QuizProblem 모델로 변환
        problems = [
            QuizProblem(
                number=problem["number"],
                type=problem["type"],
                question=problem["question"],
                choices=problem["choices"],
                answer=problem["answer"]
            )
            for problem in result["problems"]
        ]
        
        return QuizGenerationResponse(
            success=True,
            problems=problems,
            stats=result["stats"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문제 생성 중 오류가 발생했습니다: {str(e)}")

@router.post("/validate-text")
async def validate_text(text: str):
    """
    지문의 유효성을 검사합니다.
    
    Args:
        text: 검사할 지문
    
    Returns:
        유효성 검사 결과
    """
    try:
        result = quiz_service.validate_text_input(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"지문 검사 중 오류가 발생했습니다: {str(e)}")

@router.post("/validate-problem-types")
async def validate_problem_types(selected_types: List[str]):
    """
    선택된 문제 유형의 유효성을 검사합니다.
    
    Args:
        selected_types: 선택된 문제 유형 목록
    
    Returns:
        유효성 검사 결과
    """
    try:
        result = quiz_service.validate_problem_types(selected_types)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문제 유형 검사 중 오류가 발생했습니다: {str(e)}")

@router.post("/format-problem")
async def format_problem_for_display(problem: dict):
    """
    문제를 화면 표시용으로 포맷팅합니다.
    
    Args:
        problem: 원본 문제 데이터
    
    Returns:
        포맷팅된 문제 데이터
    """
    try:
        formatted_problem = quiz_service.format_problem_for_display(problem)
        return formatted_problem
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문제 포맷팅 중 오류가 발생했습니다: {str(e)}")

# Milvus 관련 엔드포인트 제거

# 퀴즈 통계 엔드포인트
@router.post("/quiz-statistics")
async def get_quiz_statistics(text: str, problems: List[dict]):
    """
    퀴즈 통계 정보를 생성합니다.
    
    Args:
        text: 원본 지문
        problems: 생성된 문제 목록
    
    Returns:
        통계 정보
    """
    try:
        stats = quiz_service.get_quiz_statistics(text, problems)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 정보 생성 중 오류가 발생했습니다: {str(e)}")

# 헬스체크 엔드포인트
@router.get("/health")
async def health_check():
    """
    서비스 상태를 확인합니다.
    """
    return {
        "status": "healthy",
        "service": "quiz_generator",
        "version": "1.0.0"
    }