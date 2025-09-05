# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from contextlib import asynccontextmanager

from routers import milvus, gemini, exam_generator
from services.gemini_service import gemini_service

# 애플리케이션 생명주기 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    print("🚀 교육 상담 RAG 챗봇 시스템 시작")
    try:
        # Milvus 연결 초기화
        milvus.get_milvus_collection()
        print("✅ Milvus 연결 완료")
        
        # Gemini 서비스 상태 확인
        test_result = await gemini_service.generate_counseling_response(
            user_query="시스템 초기화 테스트",
            search_results=None
        )
        if test_result["status"] == "success":
            print("✅ Gemini API 연결 완료")
        else:
            print("⚠️ Gemini API 연결 실패, 하지만 시스템은 계속 실행됩니다.")
            
    except Exception as e:
        print(f"❌ 시스템 초기화 중 오류: {e}")
        print("⚠️ 일부 서비스에 문제가 있을 수 있지만 시스템을 계속 실행합니다.")
    
    yield
    
    # 종료 시 실행
    print("🛑 시스템 종료 중...")
    try:
        from pymilvus import connections
        if connections.has_connection("default"):
            connections.disconnect("default")
        print("✅ Milvus 연결 정리 완료")
    except Exception as e:
        print(f"❌ 연결 정리 중 오류: {e}")

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="초등학교 교육 상담 RAG 챗봇",
    description="""
    초등학교 6학년 담임선생님을 위한 AI 상담 어시스턴트입니다.
    
    주요 기능:
    - 과거 상담 기록 기반 RAG (Retrieval-Augmented Generation)
    - 실시간 상담 채팅
    - 상담 기록 관리 (CRUD)
    - 대화 요약 및 키워드 추출
    
    기술 스택:
    - Milvus: 벡터 데이터베이스
    - Google Gemini: 임베딩 및 채팅 모델
    - FastAPI: 백엔드 프레임워크
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적인 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(milvus.router, prefix="/api/milvus", tags=["Milvus 벡터 DB"])
app.include_router(gemini.router, prefix="/api/gemini", tags=["Gemini AI 채팅"])
app.include_router(exam_generator.router, prefix="/api/exam", tags=["시험 AI 초안 생성기"])

# 루트 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "초등학교 교육 상담 RAG 챗봇 API",
        "version": "1.0.0",
        "docs": "/docs",
        "health_check": "/health"
    }

# 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    try:
        # Milvus 상태 확인
        milvus_status = "healthy"
        try:
            collection = milvus.get_milvus_collection()
            milvus_entities = collection.num_entities
        except Exception as e:
            milvus_status = f"error: {str(e)}"
            milvus_entities = 0
        
        # Gemini 상태 확인
        gemini_status = "healthy"
        try:
            test_result = await gemini_service.generate_counseling_response(
                user_query="헬스체크",
                search_results=None
            )
            if test_result["status"] != "success":
                gemini_status = f"error: {test_result.get('error', 'Unknown error')}"
        except Exception as e:
            gemini_status = f"error: {str(e)}"
        
        overall_status = "healthy" if milvus_status == "healthy" and gemini_status == "healthy" else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": "2024-08-12T00:00:00Z",  # 실제로는 datetime.now().isoformat()
            "services": {
                "milvus": {
                    "status": milvus_status,
                    "collection_name": milvus.MILVUS_COLLECTION_NAME,
                    "total_records": milvus_entities
                },
                "gemini": {
                    "status": gemini_status,
                    "model_embed": milvus.GEMINI_MODEL_EMBED,
                    "model_chat": "gemini-2.5-flash-lite"
                }
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": "2024-08-12T00:00:00Z"
            }
        )

# 전역 예외 처리
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "내부 서버 오류가 발생했습니다.",
            "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None
        }
    )

if __name__ == "__main__":
    # 개발 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )