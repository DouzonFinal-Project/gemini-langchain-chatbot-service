# routers/milvus.py

import os
import asyncio
from typing import List, Optional, Annotated

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import retry, wait_exponential, stop_after_attempt, Retrying

# =========================
# 환경 설정 로드
# =========================
load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "lang_counseling_v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_EMBED = os.getenv("GEMINI_MODEL_EMBED", "models/text-embedding-004")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768")) 

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

# GoogleGenerativeAIEmbeddings 인스턴스 초기화
embeddings = GoogleGenerativeAIEmbeddings(
    model=GEMINI_MODEL_EMBED,
    google_api_key=GEMINI_API_KEY,
    task_type="retrieval_document" # embedding_content의 task_type을 여기서 설정
)

router = APIRouter()

# =========================
# Pydantic 모델 (간단화 + U, D 기능 추가)
# =========================
TitleStr = Annotated[str, Field(min_length=1, max_length=256)]
LongText = Annotated[str, Field(min_length=1, max_length=10000)]
DateStr = Annotated[str, Field(pattern=r'^\d{4}-\d{2}-\d{2}$')]
Name = Annotated[str, Field(min_length=1, max_length=50)]
WorryTagsStr = Annotated[str, Field(min_length=0, max_length=500)]
ShortID = Annotated[str, Field(min_length=1, max_length=100)]

class AddRecordRequest(BaseModel):
    title: Optional[TitleStr] = None
    student_query: LongText
    counselor_answer: LongText
    teacher_name: Optional[Name] = None
    student_name: Optional[Name] = None
    date: DateStr
    worry_tags: Optional[WorryTagsStr] = ""

class SearchRecordsRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=1000)]
    worry_tag: Optional[Annotated[str, Field(max_length=100)]] = None
    top_k: Annotated[int, Field(default=5, ge=1, le=20)]

# <<< 신규 모델 시작 >>>
class UpdateRecordRequest(BaseModel):
    record_id: int = Field(description="수정할 레코드의 고유 ID")
    title: Optional[TitleStr] = None
    student_query: Optional[LongText] = None
    counselor_answer: Optional[LongText] = None
    teacher_name: Optional[Name] = None
    student_name: Optional[Name] = None
    date: Optional[DateStr] = None
    worry_tags: Optional[WorryTagsStr] = None

class DeleteRecordRequest(BaseModel):
    record_id: int = Field(description="삭제할 레코드의 고유 ID")
# <<< 신규 모델 종료 >>>


# =========================
# 전역 및 헬퍼 함수 (기존 코드와 동일)
# =========================
_collection: Optional[Collection] = None
_embedding_semaphore = asyncio.Semaphore(5)

async def _run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def get_gemini_document_embedding(text: str) -> List[float]:
    """
    LangChain을 사용하여 문서를 임베딩하는 함수입니다.
    
    Args:
        text (str): 임베딩할 문서 텍스트.
    
    Returns:
        List[float]: 임베딩 벡터.
    
    Raises:
        ValueError: 입력 텍스트가 비어 있는 경우.
        Exception: 임베딩 실패 시.
    """
    if not text or not text.strip():
        raise ValueError("빈 문자열은 임베딩할 수 없습니다.")
    
    # LangChain의 비동기 문서 임베딩 메서드 사용
    # 단일 문서를 임베딩하지만, 메서드는 리스트를 받으므로 [text]로 전달합니다.
    embedding_vectors = await embeddings.aembed_documents([text])
    
    # aembed_documents는 리스트를 반환하므로 첫 번째 요소를 반환
    return embedding_vectors[0]

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def get_gemini_query_embedding(text: str) -> List[float]:
    """
    LangChain을 사용하여 검색 쿼리를 임베딩하는 함수입니다.
    
    Args:
        text (str): 임베딩할 검색 쿼리 텍스트.
    
    Returns:
        List[float]: 임베딩 벡터.
    
    Raises:
        ValueError: 입력 텍스트가 비어 있는 경우.
        Exception: 임베딩 실패 시.
    """
    if not text or not text.strip():
        raise ValueError("빈 검색 쿼리입니다.")
    
    # LangChain의 비동기 검색 쿼리 임베딩 메서드 사용
    embedding_vector = await embeddings.aembed_query(text)
    
    return embedding_vector

# =========================
# Milvus 초기화 (기존 코드와 동일)
# =========================
def get_milvus_collection() -> Collection:
    global _collection
    if _collection is None:
        _collection = init_milvus_collection()
    return _collection

def init_milvus_collection() -> Collection:
    try:
        if not connections.has_connection("default"):
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=30)

        if not utility.has_collection(MILVUS_COLLECTION_NAME):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="student_query", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="counselor_answer", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="teacher_name", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="student_name", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="worry_tags", dtype=DataType.VARCHAR, max_length=500),
            ]
            schema = CollectionSchema(fields=fields, description="상담 기록 - 이미지 기반 스키마")
            col = Collection(name=MILVUS_COLLECTION_NAME, schema=schema)

            index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}}
            col.create_index(field_name="embedding", index_params=index_params)
            col.load()
            print(f"Collection '{MILVUS_COLLECTION_NAME}' created and loaded (dim={EMBEDDING_DIM})")
            return col
        else:
            col = Collection(name=MILVUS_COLLECTION_NAME)
            col.load()
            return col
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus 초기화 실패: {str(e)}")

# =========================
# API 라우터 (C, R, U, D)
# =========================
@router.post("/add-record/")
async def add_record(req: AddRecordRequest):
    try:
        collection = get_milvus_collection()
        emb = await embeddings.aembed_query(req.student_query)
        insert_data = [
            [emb],
            [req.title or ""],
            [req.student_query],
            [req.counselor_answer],
            [req.date],
            [req.teacher_name or ""],
            [req.student_name or ""],
            [req.worry_tags or ""],
        ]
        insert_result = collection.insert(insert_data)
        generated_ids = list(insert_result.primary_keys) if insert_result else []
        return {
            "status": "success",
            "generated_ids": generated_ids,
            "embedding_dim": len(emb)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"레코드 추가 실패: {str(e)}")


@router.post("/bulk-add-records/")
async def bulk_add_records(records: List[AddRecordRequest]):
    try:
        collection = get_milvus_collection()
        batch_data = [[], [], [], [], [], [], [], []] 
        errors = []
        for i, req in enumerate(records):
            try:
                emb = await embeddings.aembed_query(req.student_query)
                batch_data[0].append(emb)
                batch_data[1].append(req.title or "")
                batch_data[2].append(req.student_query)
                batch_data[3].append(req.counselor_answer)
                batch_data[4].append(req.date)
                batch_data[5].append(req.teacher_name or "")
                batch_data[6].append(req.student_name or "")
                batch_data[7].append(req.worry_tags or "")
            except Exception as e:
                errors.append({"index": i, "error": str(e)})

        if batch_data[0]:
            insert_result = collection.insert(batch_data)
            collection.flush()

        return {"status": "success", "total": len(records), "successful": len(batch_data[0]), "errors": errors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"일괄 추가 실패: {str(e)}")


# <<< 🚀 UPDATE 기능 (신규) >>>
@router.post("/update-record/")
async def update_record(req: UpdateRecordRequest):
    try:
        collection = get_milvus_collection()

        # 1. 수정할 기존 레코드 조회
        query_expr = f"id == {req.record_id}"
        existing_records = collection.query(
            expr=query_expr,
            output_fields=["title", "student_query", "counselor_answer", "date", "teacher_name", "student_name", "worry_tags"]
        )

        if not existing_records:
            raise HTTPException(status_code=404, detail=f"ID {req.record_id}에 해당하는 레코드를 찾을 수 없습니다.")
        
        old_record = existing_records[0]

        # 2. 요청받은 데이터로 새 레코드 정보 구성
        new_student_query = req.student_query if req.student_query is not None else old_record['student_query']
        new_emb = await embeddings.aembed_query(new_student_query)

        new_data = [
            [new_emb],
            [req.title if req.title is not None else old_record['title']],
            [new_student_query],
            [req.counselor_answer if req.counselor_answer is not None else old_record['counselor_answer']],
            [req.date if req.date is not None else old_record['date']],
            [req.teacher_name if req.teacher_name is not None else old_record['teacher_name']],
            [req.student_name if req.student_name is not None else old_record['student_name']],
            [req.worry_tags if req.worry_tags is not None else old_record['worry_tags']],
        ]

        # 3. 기존 레코드 삭제
        collection.delete(expr=query_expr)

        # 4. 새로운 정보로 레코드 추가
        insert_result = collection.insert(new_data)
        new_id = insert_result.primary_keys[0]
        
        collection.flush()

        return {
            "status": "success",
            "message": "레코드가 성공적으로 업데이트되었습니다.",
            "old_record_id": req.record_id,
            "new_record_id": new_id
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"레코드 업데이트 실패: {str(e)}")

# <<< 🗑️ DELETE 기능 (신규) >>>
@router.post("/delete-record/")
async def delete_record(req: DeleteRecordRequest):
    try:
        collection = get_milvus_collection()
        expr = f"id == {req.record_id}"

        check_result = collection.query(expr=expr, output_fields=["id"], limit=1)
        if not check_result:
             raise HTTPException(status_code=404, detail=f"ID {req.record_id}에 해당하는 레코드를 찾을 수 없습니다.")

        delete_result = collection.delete(expr)
        collection.flush()

        return {
            "status": "success",
            "deleted_id": req.record_id,
            "deleted_count": delete_result.delete_count
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"레코드 삭제 실패: {str(e)}")


@router.post("/search-records/")
async def search_records(req: SearchRecordsRequest):
    try:
        collection = get_milvus_collection()
        emb = await get_gemini_query_embedding(req.query)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        expr = None
        if req.worry_tag:
            expr = f'worry_tags like "%{req.worry_tag}%"'

        results = collection.search(
            data=[emb],
            anns_field="embedding",
            param=search_params,
            limit=req.top_k,
            expr=expr,
            output_fields=["id", "title", "student_query", "counselor_answer", "date", "teacher_name", "student_name", "worry_tags"],
        )

        output = []
        # COSINE 거리는 0(유사) ~ 2(비유사) 범위를 가지므로 1에서 빼서 유사도로 변환
        for hit in results[0]:
            output.append({
                "id": hit.entity.get("id"),
                "title": hit.entity.get("title"),
                "student_query": hit.entity.get("student_query"),
                "counselor_answer": hit.entity.get("counselor_answer"),
                "date": hit.entity.get("date"),
                "teacher_name": hit.entity.get("teacher_name"),
                "student_name": hit.entity.get("student_name"),
                "worry_tags": hit.entity.get("worry_tags"),
                "similarity": round(1 - hit.distance, 4), # Cosine 유사도로 변환
            })

        return {"status": "success", "total_found": len(output), "results": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")


@router.get("/collection-stats/")
def get_collection_stats():
    try:
        collection = get_milvus_collection()
        return {
            "status": "success",
            "collection_name": MILVUS_COLLECTION_NAME,
            "total_entities": collection.num_entities,
            "has_index": collection.has_index(),
            "is_loaded": True,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# =========================
# 앱 생명주기 (기존 코드와 동일)
# =========================
@router.on_event("startup")
async def startup_event():
    try:
        get_milvus_collection()
        print("✅ Milvus 연결 및 컬렉션 초기화 완료")
    except Exception as e:
        print(f"❌ Milvus 초기화 실패: {e}")

@router.on_event("shutdown")
async def shutdown_event():
    try:
        if connections.has_connection("default"):
            connections.disconnect("default")
        print("✅ Milvus 연결 정리 완료")
    except Exception as e:
        print(f"❌ Milvus 연결 정리 실패: {e}")