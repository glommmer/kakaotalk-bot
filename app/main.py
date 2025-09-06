import os
import re
import sys
import json
import asyncio
import logging
import requests
import redis.asyncio as redis
from typing import List, cast
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from openai import OpenAI, APIError
from openai.types.chat import ChatCompletionMessageParam
from tavily import TavilyClient
from duckduckgo_search import DDGS

# --- 1. 로깅 설정 ---
# 로그 포맷 및 레벨 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- 2. 환경 변수 로드 및 검증 ---
load_dotenv()

REQUIRED_VARS = ["AX_API_URL", "AX_API_KEY", "REDIS_HOST", "REDIS_PORT"]
if all(var in os.environ for var in REQUIRED_VARS):
    AX_API_URL = os.getenv("AX_API_URL")
    AX_API_KEY = os.getenv("AX_API_KEY")
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT"))  # 포트 정수형 변환
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    logger.info("환경 변수가 성공적으로 로드되었습니다.")
else:
    missing_vars = [var for var in REQUIRED_VARS if var not in os.environ]
    logger.critical(f"필수 환경 변수가 설정되지 않았습니다: {missing_vars}")
    sys.exit("## API 환경 변수 확인 필요. 프로그램을 종료합니다.")

# --- 3. 상수 및 클라이언트 초기화 ---
# 카카오톡 스킬 타임아웃(5초)보다 짧은 시간 설정
SKILL_TIMEOUT = float(os.getenv("SKILL_TIMEOUT", 4.5))
N_HISTORY = int(os.getenv("N_HISTORY", 10))

client: OpenAI
redis_client: redis.Redis
tavily_client: TavilyClient


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 애플리케이션 시작할 때 실행
    global client, redis_client, tavily_client
    logger.info("애플리케이션 시작...")
    client = OpenAI(base_url=AX_API_URL, api_key=AX_API_KEY)
    redis_client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
    )
    await redis_client.ping()
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
    logger.info("모든 클라이언트가 성공적으로 초기화되었습니다.")

    yield  # 애플리케이션 실행

    # 애플리케이션 종료할 때 실행
    logger.info("애플리케이션 종료... 리소스를 정리합니다.")
    await redis_client.close()


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


# --- 4. API 엔드포인트 ---
@app.get("/healthcheck", status_code=200)
def health_check():
    """로드밸런서 상태 검사를 위한 엔드포인트"""
    return JSONResponse(content={"status": "ok"})


@app.post("/basic")
async def basic(request: Request):
    """사용자 요청을 받아 LLM 응답을 반환하거나, 지연 시 콜백을 사용"""
    req_data = await request.json()

    user_id = req_data["userRequest"]["user"]["id"]
    query = req_data["userRequest"]["utterance"]
    callback_url = req_data["userRequest"].get("callbackUrl")

    # Redis에서 대화 기록 가져오기
    redis_key = f"history:{user_id}"
    # Redis는 문자열 리스트로 저장하므로, 각 문자열을 JSON으로 파싱
    history_str_list = await redis_client.lrange(redis_key, -N_HISTORY, -1)
    user_history: List[ChatCompletionMessageParam] = []
    for item in history_str_list:
        try:
            loaded_item = json.loads(item)
            user_history.append(cast(ChatCompletionMessageParam, loaded_item))
        except json.JSONDecodeError:
            logger.warning(
                f"Redis에서 손상된 대화 기록 발견 (key: {redis_key}): {item}"
            )

    logger.info(f"수신된 쿼리: {query} (사용자 ID: {user_id})")

    if not callback_url:
        logger.info("Callback URL이 없으므로 즉시 답변을 시도합니다.")
        # content = simple_answer(query)
        content = simple_answer(query, user_history)
        await update_history_in_redis(
            redis_key, query, content
        )  # Redis에 대화 기록 업데이트
        return build_kakao_response(content)

    # asyncio 는 이벤트 루프 중심으로 동작
    # 이벤트 루프: 실행해야 할 작업들을 등록, 순서에 맞게 실행, 전체 흐름 관리하는 매니저 역할
    # 현재 실행 중인 이벤트 루프를 가져와 앞으로의 작업을 맡기기 위해 준비
    loop = asyncio.get_event_loop()

    # simple_answer 같은 동기 함수를 별도의 스레드에서 실행시켜 이벤트 루프 중단 방지
    # simple_answer 함수 즉시 실행, 그 작업 자체를 llm_task 라는 Future 객체로 만듦
    # None: asyncio 기본 ThreadPoolExecutor 사용
    # llm_task = loop.run_in_executor(None, simple_answer, query)
    llm_task = loop.run_in_executor(None, simple_answer, query, user_history)

    # 지정된 시간(SKILL_TIMEOUT) 동안 기다리는 비동기 함수
    # 이벤트 루프에 등록만 해놓고, 그 동안 다른 작업 실행될 수 있게 제어권 넘김
    timeout_task = asyncio.create_task(asyncio.sleep(SKILL_TIMEOUT))

    # asyncio.wait(): 여러 개의 비동기 작업을 묶어서 실행 후 특정 조건 만족될 때까지 기다리는 함수
    # llm_task 와 timeout_task 두 개의 task 동시 실행
    # 둘 중 어느 것 하나라도 끝나면 결과 반환
    done, pending = await asyncio.wait(
        {llm_task, timeout_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if llm_task in done:
        # 타임아웃 내 LLM 응답 도착
        for task in pending:
            task.cancel()

        content = llm_task.result()
        await update_history_in_redis(
            redis_key, query, content
        )  # Redis에 대화 기록 업데이트
        logger.info(f"즉시 답변 성공 (소요 시간 < {SKILL_TIMEOUT}초)")
        return build_kakao_response(content)
    else:
        # 타임아웃 발생 (LLM 응답 지연)
        logger.info(
            f"응답 지연 감지 (소요 시간 > {SKILL_TIMEOUT}초). 콜백을 사용합니다."
        )
        loop.create_task(process_and_callback(llm_task, callback_url, redis_key, query))
        return {"version": "2.0", "useCallback": True, "template": {}}


# --- 5. Helper 함수 ---
async def process_and_callback(llm_task, callback_url, redis_key, query):
    """백그라운드에서 LLM 작업 완료 후 결과를 콜백으로 전송"""
    try:
        content = await llm_task
        logger.info("백그라운드 LLM 작업 완료. 콜백을 전송합니다.")
        call_kakao_callback(callback_url, content)
        await update_history_in_redis(
            redis_key, query, content
        )  # Redis에 대화 기록 업데이트
    except Exception as exc:
        logger.error(f"백그라운드 작업 실패: {exc}")
        error_message = "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다."
        call_kakao_callback(callback_url, error_message)


async def update_history_in_redis(key: str, user_query: str, assistant_response: str):
    """사용자 질문과 AI 답변을 Redis List에 추가하고 만료 시간을 설정합니다. (파이프라인 사용)"""
    # asyncio.gather 사용, 여러 Redis 명령 동시 실행
    await asyncio.gather(
        redis_client.rpush(key, json.dumps({"role": "user", "content": user_query})),
        redis_client.rpush(
            key, json.dumps({"role": "assistant", "content": assistant_response})
        ),
        redis_client.expire(key, 3600),
    )


def web_search(query: str) -> str:
    """Tavily 또는 DuckDuckGo를 사용하여 웹 검색을 수행 후 결과 반환"""
    logger.info(f"웹 검색 실행: {query}")
    results = []
    try:
        if tavily_client:
            logger.info("Tavily API를 사용하여 검색합니다.")
            response = tavily_client.search(query=query, search_depth="basic")
            results = [obj["content"] for obj in response.get("results", [])]
        else:
            raise Exception("Tavily API 키가 설정되지 않았습니다.")
    except Exception as exc:
        logger.warning(f"Tavily 검색 실패: {exc}. DuckDuckGo로 대체합니다.")
        with DDGS() as ddgs:
            # max_results=5 -> 검색 결과 5개
            for r in ddgs.text(query, max_results=5):
                results.append(r.get("body", ""))
    return "\n".join(results)


def _parse_p_tag(text: str) -> str:
    """<p> 태그 안의 내용 추출, 태그가 없으면 원본 텍스트 반환"""
    match = re.search(r"<p>(.*?)</p>", text, re.DOTALL)
    if match:
        logger.info("Parsing <p> tag.")
        return match.group(1).strip()
    # logger.warning("응답에서 <p> 태그를 찾을 수 없어 원본 텍스트를 반환합니다.")
    return text


def simple_answer(query: str, history: List[ChatCompletionMessageParam]) -> str:
    """LLM API를 호출하여 답변을 생성"""

    # 1> 웹 검색이 필요한지, 필요하다면 검색어가 무엇인지 LLM에게 판단 요청
    planner_prompt = f"""
    You are a JSON generator that analyzes user questions to determine whether a web search is necessary. 
    Your sole task is to return one JSON object according to the rules below. 
    Do not add any other explanations or greetings.

    [Rules]
    1. If up-to-date information (news, weather) or fact-checking about a specific person, place, or product is needed, `search_required` should be `true`.
    2. If `search_required` is `true`, the **`query` field must contain the core search keywords in English.** This field cannot be left empty.
    3. If the question is a simple greeting, expression of feelings, etc., and does not require a search, `search_required` should be `false` and `query` should be `null`.
    
    [User Question] 
    "{query}" 
    
    [Your JSON Response]
    """

    try:
        planner_messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": planner_prompt}]  # type: ignore

        response = client.chat.completions.create(
            model="ax4",
            messages=planner_messages,
            temperature=0.0,
            response_format={"type": "json_object"},  # type: ignore
        )
        decision_str = response.choices[0].message.content

        if "```" in decision_str:
            decision_str = decision_str.split("json\n")[1].strip().rstrip("`")

        decision = json.loads(decision_str.strip())
        logger.info(f"검색 계획: {decision}")
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error(f"검색 계획 수립 실패: {exc}. 검색 없이 답변을 시도합니다.")
        decision = {"search_required": False, "query": None}

    # 2> LLM의 결정에 따라 행동
    if decision.get("search_required"):
        search_query = decision.get("query")

        # LLM이 검색어를 제대로 반환했는지 검증
        if search_query:
            logger.info(f"검색이 필요하다고 판단. 검색어: '{search_query}'")
            search_results = web_search(search_query)  # type: ignore
            truncated_results = search_results[:4000]

            # 3> 검색 결과를 바탕으로 최종 답변 생성 요청
            synthesizer_prompt = f"""
            You are an AI assistant that answers user questions based on the given web search results.
            Do not mention anything not found in the search results, and synthesize your answer using only the provided information.
            **Crucially, you MUST wrap your final, complete answer in a single <p> tag.** For example: <p>이것이 최종 답변입니다.</p>
        
            [Web Search Results]
            {truncated_results}
        
            [User's Original Question]
            {query}
            
            [Your Final Answer in <p> tag]
            """
            messages = history + [{"role": "user", "content": synthesizer_prompt}]

        else:
            # 검색이 필요하다고 했지만 검색어가 없는 경우, 검색 없이 답변
            logger.warning("LLM이 검색어를 제공하지 않아 검색 없이 답변을 시도합니다.")
            messages = history + [{"role": "user", "content": query}]
    else:
        # 검색이 필요 없을 경우, 바로 답변 생성
        logger.info("검색이 불필요하다고 판단. 직접 답변을 생성합니다.")
        messages = history + [{"role": "user", "content": query}]

    # 최종 답변 생성
    system_prompt = """
    You are a helpful AI assistant. 
    Your response must be in Korean and formatted for a text messenger.
    Never use markdown formats such as ##, **.
    """

    final_messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}] + messages  # type: ignore

    try:
        final_response = client.chat.completions.create(
            model="ax4", messages=final_messages, timeout=20
        )

        content_with_tag = final_response.choices[0].message.content
        final_content = _parse_p_tag(content_with_tag)

        # return final_response.choices[0].message.content
        return final_content

    except APIError as exc:
        logger.error(f"LLM API 오류 발생: {exc}")
        return "API 오류가 발생하여 답변을 가져올 수 없습니다."
    except Exception as exc:
        logger.error(f"simple_answer 내 알 수 없는 오류: {exc}")
        return "알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


def call_kakao_callback(callback_url: str, content: str):
    """생성된 답변을 카카오 콜백 URL로 POST 요청"""
    payload = build_kakao_response(content)
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(callback_url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()  # 2xx 상태 코드가 아니면 예외 발생
        logger.info(f"콜백 성공: {resp.status_code} {resp.text}")
    except requests.exceptions.RequestException as exc:
        logger.error(f"콜백 실패: {exc}")


def build_kakao_response(content: str) -> dict:
    """카카오톡 simpleText 응답 포맷을 생성"""
    return {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": content}}]},
    }
