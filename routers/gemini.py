# routers/gemini.py

import os
from typing import List, Optional, Dict, Any, Annotated
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from functools import lru_cache
import json

from services.gemini_service import gemini_service
from routers.milvus import SearchRecordsRequest, get_milvus_collection

router = APIRouter()

# =========================
# Pydantic 모델 정의
# =========================

class ChatMessage(BaseModel):
    role: Annotated[str, Field(pattern="^(user|assistant)$")]
    content: Annotated[str, Field(min_length=1, max_length=10000)]
    timestamp: Optional[str] = None

class CounselingChatRequest(BaseModel):
    """상담 채팅 요청 모델"""
    query: Annotated[str, Field(min_length=1, max_length=2000, description="상담 질문")]
    use_rag: bool = Field(default=True, description="RAG 검색 사용 여부")
    search_top_k: Annotated[int, Field(default=3, ge=1, le=10)] = 3
    worry_tag_filter: Optional[str] = Field(default=None, max_length=100, description="검색시 고민 태그 필터")
    conversation_history: Optional[List[ChatMessage]] = Field(default=None, description="대화 히스토리 (최대 20개)")
    student_name: Optional[str] = Field(default=None, max_length=50, description="학생 이름 (선택)")
    context_info: Optional[Dict[str, str]] = Field(default=None, description="추가 상황 정보")

class QuickChatRequest(BaseModel):
    """간단 채팅 요청 (RAG 없이)"""
    query: Annotated[str, Field(min_length=1, max_length=2000)]
    conversation_history: Optional[List[ChatMessage]] = None
    urgency_level: Optional[str] = Field(default="normal", pattern="^(low|normal|high|urgent)$")

class SummarizeRequest(BaseModel):
    """대화 요약 요청"""
    conversation_history: List[ChatMessage] = Field(min_items=2, max_items=50)
    include_action_items: bool = Field(default=True, description="실행 계획 포함 여부")
    summary_type: str = Field(default="detailed", pattern="^(brief|detailed|formal)$")

class ExtractKeywordsRequest(BaseModel):
    """키워드 추출 요청"""  
    text: Annotated[str, Field(min_length=10, max_length=5000)]
    include_priority: bool = Field(default=True, description="우선순위 정보 포함")
    extract_emotions: bool = Field(default=False, description="감정 상태 분석 포함")

class CounselingPlanRequest(BaseModel):
    """상담 계획 수립 요청"""
    student_name: str = Field(max_length=50)
    grade: Optional[int] = Field(default=6, ge=1, le=6)
    main_concerns: List[str] = Field(min_items=1, max_items=5)
    current_situation: str = Field(min_length=10, max_length=1000)
    family_background: Optional[str] = Field(default=None, max_length=500)
    academic_level: Optional[str] = Field(default=None, pattern="^(high|medium|low)$")
    social_skills: Optional[str] = Field(default=None, pattern="^(excellent|good|fair|needs_improvement)$")

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    status: str
    response: Optional[str] = None
    error: Optional[str] = None
    timestamp: str
    used_rag: Optional[bool] = None
    search_results_count: Optional[int] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    context_quality: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = None

# =========================
# 헬퍼 함수
# =========================

async def perform_rag_search(query: str, top_k: int = 3, worry_tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """RAG 검색 수행"""
    try:
        collection = get_milvus_collection()
        
        # 임베딩 생성 (milvus.py의 함수 재사용)
        from routers.milvus import get_gemini_query_embedding
        emb = await get_gemini_query_embedding(query)
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        expr = None
        if worry_tag:
            expr = f'worry_tags like "%{worry_tag}%"'
        
        results = collection.search(
            data=[emb],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "title", "student_query", "counselor_answer", "date", "teacher_name", "student_name", "worry_tags"],
        )
        
        output = []
        for hit in results[0]:
            similarity = round(1 - hit.distance, 4)
            # 유사도가 너무 낮은 결과는 제외
            if similarity < 0.3:
                continue
                
            output.append({
                "id": hit.entity.get("id"),
                "title": hit.entity.get("title"),
                "student_query": hit.entity.get("student_query"),
                "counselor_answer": hit.entity.get("counselor_answer"),
                "date": hit.entity.get("date"),
                "teacher_name": hit.entity.get("teacher_name"),
                "student_name": hit.entity.get("student_name"),
                "worry_tags": hit.entity.get("worry_tags"),
                "similarity": similarity,
            })
        
        return output
    
    except Exception as e:
        print(f"RAG 검색 실패: {e}")
        return []

def log_conversation(query: str, response: str, used_rag: bool, search_count: int):
    """대화 로그 기록 (백그라운드 태스크)"""
    try:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "response_length": len(response),
            "used_rag": used_rag,
            "search_results_count": search_count,
        }
        # 실제 구현에서는 데이터베이스나 로그 파일에 저장
        print(f"대화 로그: {json.dumps(log_data, ensure_ascii=False)}")
    except Exception as e:
        print(f"로그 기록 실패: {e}")

# =========================
# templates
# =========================
@lru_cache(maxsize=1)
def _load_json_data(filename: str) -> Dict[str, Any]:
    """템플릿 폴더에서 JSON 파일을 읽어옵니다. 캐싱을 적용하여 효율성을 높입니다."""
    file_path = os.path.join(os.path.dirname(__file__), '..', 'templates', filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: '{filename}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        raise HTTPException(status_code=500, detail="서버 템플릿 파일을 찾을 수 없습니다.")
    
@router.get("/chat-templates/")
async def get_chat_templates():
    """자주 사용되는 상담 질문 템플릿 제공"""
    templates = _load_json_data("chat_templates.json")
    
    return {
        "status": "success",
        "templates": templates,
        "total_categories": len(templates),
        "total_templates": sum(len(v["templates"]) for v in templates.values()),
        "usage_tip": "상황에 맞는 템플릿을 선택하거나 참고하여 구체적인 질문을 작성해보세요."
    }

@router.get("/counseling-guidelines/")
async def get_counseling_guidelines():
    """초등학교 상담 가이드라인 제공"""
    data = _load_json_data("counseling_guidelines.json")
    
    # JSON 파일의 루트에 있는 'guidelines'와 기타 필드를 직접 반환
    return {
        "status": "success",
        **data
    }

# =========================
# API 엔드포인트
# =========================

@router.post("/counseling-chat/", response_model=ChatResponse)
async def counseling_chat(request: CounselingChatRequest, background_tasks: BackgroundTasks):
    """
    전문 상담 채팅 (RAG 지원)
    - 과거 상담 기록을 검색하여 컨텍스트로 활용
    - 대화 히스토리 지원
    - 학생별 맞춤 상담
    """
    start_time = datetime.now()
    
    try:
        search_results = []
        
        # RAG 검색 수행
        if request.use_rag:
            # 학생 이름이 있으면 검색 쿼리에 포함
            search_query = request.query
            if request.student_name:
                search_query = f"{request.student_name} 학생 {request.query}"
                
            search_results = await perform_rag_search(
                query=search_query,
                top_k=request.search_top_k,
                worry_tag=request.worry_tag_filter
            )
        
        # 대화 히스토리 변환
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history[-20:]  # 최근 20개만
            ]
        
        # 추가 컨텍스트 정보 처리
        enhanced_query = request.query
        if request.context_info:
            context_parts = []
            for key, value in request.context_info.items():
                context_parts.append(f"{key}: {value}")
            if context_parts:
                enhanced_query = f"{request.query}\n\n[추가 상황 정보]\n" + "\n".join(context_parts)
        
        # Gemini API 호출
        result = await gemini_service.generate_counseling_response(
            user_query=enhanced_query,
            search_results=search_results,
            conversation_history=conversation_history
        )
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        if result["status"] == "success":
            # 백그라운드에서 로그 기록
            background_tasks.add_task(
                log_conversation, 
                request.query, 
                result["response"], 
                request.use_rag, 
                len(search_results)
            )
            
            return ChatResponse(
                status="success",
                response=result["response"],
                timestamp=result["timestamp"],
                used_rag=request.use_rag,
                search_results_count=len(search_results),
                search_results=search_results if request.use_rag else None,
                context_quality=result.get("context_quality"),
                response_time=response_time
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상담 채팅 처리 실패: {str(e)}")


@router.post("/quick-chat/", response_model=ChatResponse)
async def quick_chat(request: QuickChatRequest):
    """
    간단 채팅 (RAG 없이)
    - 빠른 응답을 위한 일반적인 교육 상담
    - 과거 기록 검색 없이 기본 지식으로만 답변
    - 긴급도에 따른 응답 우선순위 적용
    """
    start_time = datetime.now()
    
    try:
        # 긴급도에 따른 쿼리 강화
        enhanced_query = request.query
        if request.urgency_level == "urgent":
            enhanced_query = f"[긴급 상담] {request.query}\n\n즉시 실행 가능한 구체적인 해결책을 우선 제시해주세요."
        elif request.urgency_level == "high":
            enhanced_query = f"[우선 처리] {request.query}\n\n빠른 해결이 필요한 상황입니다."
        
        # 대화 히스토리 변환
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history[-10:]  # 최근 10개만
            ]
        
        # Gemini API 호출 (RAG 없이)
        result = await gemini_service.generate_counseling_response(
            user_query=enhanced_query,
            search_results=None,  # RAG 사용 안함
            conversation_history=conversation_history
        )
        
        response_time = (datetime.now() - start_time).total_seconds()
        
        if result["status"] == "success":
            return ChatResponse(
                status="success",
                response=result["response"],
                timestamp=result["timestamp"],
                used_rag=False,
                search_results_count=0,
                response_time=response_time
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"간단 채팅 처리 실패: {str(e)}")


@router.post("/counseling-plan/")
async def create_counseling_plan(request: CounselingPlanRequest):
    """개별 학생을 위한 상담 계획 수립"""
    try:
        student_info = {
            "student_name": request.student_name,
            "grade": request.grade,
            "main_concerns": request.main_concerns,
            "current_situation": request.current_situation,
            "family_background": request.family_background,
            "academic_level": request.academic_level,
            "social_skills": request.social_skills
        }
        
        result = await gemini_service.generate_counseling_plan(student_info)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "counseling_plan": result["counseling_plan"],
                "student_name": request.student_name,
                "timestamp": result["timestamp"],
                "plan_duration": "1학기 (약 4개월)"
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상담 계획 수립 실패: {str(e)}")


@router.post("/summarize-conversation/")
async def summarize_conversation(request: SummarizeRequest):
    """대화 내용 요약 생성"""
    try:
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_history
        ]
        
        result = await gemini_service.generate_summary(conversation_history)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "summary": result["summary"],
                "timestamp": result["timestamp"],
                "conversation_length": len(request.conversation_history),
                "summary_type": request.summary_type,
                "includes_action_items": request.include_action_items
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대화 요약 실패: {str(e)}")


@router.post("/extract-keywords/")
async def extract_keywords(request: ExtractKeywordsRequest):
    """텍스트에서 고민 태그/키워드 추출"""
    try:
        result = await gemini_service.generate_keywords(request.text)
        
        if result["status"] == "success":
            return {
                "status": "success",
                "keywords": result["keywords"],
                "timestamp": result["timestamp"],
                "text_length": len(request.text),
                "include_priority": request.include_priority,
                "include_emotions": request.extract_emotions
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"키워드 추출 실패: {str(e)}")

@router.get("/service-status/")
async def get_service_status():
    """서비스 상태 확인"""
    try:
        # Gemini API 상태 확인
        test_result = await gemini_service.generate_counseling_response(
            user_query="안녕하세요. 시스템 상태 확인 테스트입니다.",
            search_results=None
        )
        
        gemini_status = "healthy" if test_result["status"] == "success" else "error"
        
        # Milvus 상태 확인
        milvus_status = "healthy"
        milvus_info = {}
        try:
            collection = get_milvus_collection()
            milvus_info = {
                "total_records": collection.num_entities,
                "collection_name": collection.name,
                "is_loaded": True
            }
        except Exception as e:
            milvus_status = "error"
            milvus_info = {"error": str(e)}
        
        overall_status = "healthy" if gemini_status == "healthy" and milvus_status == "healthy" else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "gemini_api": {
                    "status": gemini_status,
                    "model": "gemini-2.5-flash",
                    "features": ["chat", "summarization", "keyword_extraction", "planning"]
                },
                "milvus_db": {
                    "status": milvus_status,
                    **milvus_info
                },
                "rag_system": {
                    "status": "healthy" if overall_status == "healthy" else "degraded",
                    "search_enabled": milvus_status == "healthy"
                }
            },
            "performance": {
                "average_response_time": "< 3초",
                "rag_search_time": "< 1초",
                "concurrent_users": "최대 10명"
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/usage-statistics/")
async def get_usage_statistics():
    """서비스 사용 통계 (데모용)"""
    # 실제 구현에서는 데이터베이스에서 실제 통계를 가져와야 합니다
    return {
        "status": "success",
        "period": "최근 30일",
        "statistics": {
            "total_conversations": 156,
            "total_queries": 423,
            "rag_usage_rate": 0.73,
            "most_common_topics": [
                {"topic": "학습지도", "count": 98},
                {"topic": "교우관계", "count": 87},
                {"topic": "행동문제", "count": 65},
                {"topic": "정서지원", "count": 54},
                {"topic": "학부모상담", "count": 41}
            ],
            "average_response_time": 2.3,
            "user_satisfaction": 4.2,
            "peak_usage_hours": ["09:00-10:00", "14:00-15:00", "16:00-17:00"]
        },
        "trends": {
            "weekly_growth": "+12%",
            "monthly_active_teachers": 28,
            "repeat_usage_rate": 0.68
        },
        "generated_at": datetime.now().isoformat()
    }