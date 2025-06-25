import os
import platform
import sys

# protobuf 호환성을 위한 환경변수 설정 (다른 모든 import보다 먼저!)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# sqlite3 패치 (배포 환경 고려)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("<<<<< sqlite3 patched with pysqlite3 >>>>>")
except ImportError:
    # pysqlite3가 없으면 기본 sqlite3 사용 (로컬 환경에서는 정상)
    print("<<<<< using default sqlite3 >>>>>")

print(f"<<<<< app.py IS BEING LOADED on {platform.system()} >>>>>")

import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Dict, List, Any
import time
import uuid
import subprocess
import shutil
import glob
import gc
from pathlib import Path

# from streamlit_extras.buy_me_a_coffee import button

# 수동으로 Buy Me a Coffee 버튼 구현
def add_buy_me_coffee_button():
    st.markdown(
        """
        <style>
        .coffee-button {
            text-align: left;
            margin-bottom: 20px;
        }
        .coffee-button img {
            height: 40px !important;
            width: 144px !important;
            max-height: 40px !important;
            max-width: 144px !important;
            object-fit: contain !important;
            border: none !important;
        }
        </style>
        <div class="coffee-button">
            <a href="https://www.buymeacoffee.com/ehtjd33e" target="_blank">
                <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" 
                     alt="Buy Me A Coffee">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# .env 파일 로드
load_dotenv()

st.set_page_config(page_title="법률가 챗봇", page_icon=":books:", layout="wide")

# 버튼 추가 (제목 바로 위)
add_buy_me_coffee_button()

st.title("📚 법률가 챗봇")
st.caption("PDF 문서를 업로드하여 법률 관련 질문에 답변받으세요")

# OpenAI API 키 로드
openai_api_key = os.getenv("OPENAI_API_KEY")

# OS 감지 및 플랫폼별 설정
SYSTEM_OS = platform.system().lower()
IS_WINDOWS = SYSTEM_OS == 'windows'
IS_LINUX = SYSTEM_OS == 'linux'
IS_MACOS = SYSTEM_OS == 'darwin'

# 배포 환경 감지 (Streamlit Cloud, Heroku 등)
IS_DEPLOYMENT = any([
    os.getenv('STREAMLIT_CLOUD'),
    os.getenv('HEROKU'),
    os.getenv('RAILWAY'),
    os.getenv('RENDER'),
    '/app' in os.getcwd(),  # 일반적인 컨테이너 경로
    '/home/appuser' in os.getcwd()  # Streamlit Cloud 경로
])

print(f"시스템 정보: OS={SYSTEM_OS}, 배포환경={IS_DEPLOYMENT}")

# 스트리밍을 위한 콜백 핸들러
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▋")  # 커서 효과
        time.sleep(0.01)  # 자연스러운 타이핑 효과

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.container.markdown(self.text)  # 최종 텍스트 (커서 제거)

# 사이드바
with st.sidebar:
    st.header("📁 파일 업로드")
    
    # PDF 파일 업로드
    uploaded_file = st.file_uploader(
        "PDF 파일을 업로드하세요",
        type=["pdf"],
        help="법률 문서나 관련 자료를 PDF 형태로 업로드하세요"
    )
    
    st.divider()
    
    st.header("⚙️ 설정")
    if not openai_api_key:
        st.error("⚠️ .env 파일에 OPENAI_API_KEY를 설정해주세요!")
        st.info("📝 .env 파일 예시:\nOPENAI_API_KEY=your_api_key_here")
    else:
        st.success("✅ OpenAI API 키가 설정되었습니다.")
    
    if uploaded_file:
        st.success(f"✅ 파일 업로드됨: {uploaded_file.name}")
        st.info(f"📄 파일 크기: {uploaded_file.size / 1024:.1f} KB")

# 크로스 플랫폼 호환 ChromaDB 삭제 함수
def safe_delete_chromadb_cross_platform(target_dir, max_retries=3, delay=1):
    """모든 OS에서 안전하게 ChromaDB 폴더를 삭제합니다."""
    
    # 먼저 vectorstore 객체 해제 및 가비지 컬렉션 강제 실행
    if 'vectorstore' in st.session_state:
        st.session_state.vectorstore = None
    gc.collect()
    
    if not os.path.exists(target_dir):
        return True
    
    # Path 객체 사용으로 크로스 플랫폼 호환성 개선
    target_path = Path(target_dir)
    temp_name = f"{target_dir}_deleted_{uuid.uuid4().hex[:8]}"
    temp_path = Path(temp_name)
    
    try:
        # 1단계: 폴더 이름 변경 (모든 OS에서 안전)
        target_path.rename(temp_path)
        print(f"ChromaDB 디렉토리 이름 변경: {target_dir} -> {temp_name}")
        
        # 2단계: 플랫폼별 삭제 전략
        success = False
        
        for attempt in range(max_retries):
            try:
                time.sleep(delay)
                
                if IS_WINDOWS and not IS_DEPLOYMENT:
                    # Windows 로컬 환경: 강화된 삭제 방법
                    success = _windows_force_delete(temp_path)
                else:
                    # Linux/macOS 또는 배포 환경: 표준 방법
                    success = _standard_delete(temp_path)
                
                if success:
                    print(f"ChromaDB 디렉토리 삭제 성공: {temp_name}")
                    return True
                    
            except Exception as e:
                print(f"삭제 시도 {attempt + 1}/{max_retries} 실패: {str(e)}")
                if attempt < max_retries - 1:
                    delay *= 2
        
        # 삭제 실패 시에도 이름 변경은 성공했으므로 True 반환
        print(f"ChromaDB 디렉토리 삭제 실패, 나중에 정리됨: {temp_name}")
        return True
        
    except Exception as e:
        print(f"ChromaDB 디렉토리 처리 실패: {str(e)}")
        return False

def _windows_force_delete(path):
    """Windows에서 강화된 삭제 방법"""
    try:
        # 방법 1: 표준 shutil 삭제
        shutil.rmtree(path)
        return True
    except Exception:
        pass
    
    try:
        # 방법 2: Windows 전용 robocopy 사용 (robocopy가 있는 경우에만)
        if shutil.which("robocopy"):
            empty_dir = Path("./temp_empty_dir")
            empty_dir.mkdir(exist_ok=True)
            
            # robocopy로 빈 폴더를 미러링하여 삭제 효과
            result = subprocess.run([
                "robocopy", str(empty_dir), str(path), "/MIR", "/NFL", "/NDL", "/NJH", "/NJS"
            ], capture_output=True, text=True)
            
            # 빈 폴더들 정리
            empty_dir.rmdir()
            path.rmdir()
            return True
    except Exception:
        pass
    
    try:
        # 방법 3: 파일별 개별 삭제
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                os.chmod(os.path.join(root, file), 0o777)
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(path)
        return True
    except Exception:
        pass
    
    return False

def _standard_delete(path):
    """표준 삭제 방법 (Linux/macOS/배포환경)"""
    try:
        shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"표준 삭제 실패: {str(e)}")
        return False

# 전체 ChromaDB 정리 함수 (크로스 플랫폼)
def cleanup_all_chromadb():
    """모든 ChromaDB 관련 폴더를 정리합니다."""
    try:
        # 현재 디렉토리에서 ChromaDB 관련 폴더 찾기
        current_dir = Path.cwd()
        chroma_patterns = ["chroma_db*", ".chromadb*"]
        
        cleaned_count = 0
        for pattern in chroma_patterns:
            for chroma_path in current_dir.glob(pattern):
                if chroma_path.is_dir():
                    try:
                        success = safe_delete_chromadb_cross_platform(str(chroma_path))
                        if success:
                            cleaned_count += 1
                            print(f"정리됨: {chroma_path}")
                    except Exception as e:
                        print(f"정리 중 오류 ({chroma_path}): {str(e)}")
        
        if cleaned_count > 0:
            print(f"총 {cleaned_count}개의 ChromaDB 폴더가 정리되었습니다.")
        
        return True
    except Exception as e:
        print(f"ChromaDB 전체 정리 중 오류: {str(e)}")
        return False

# 안전한 ChromaDB 삭제 함수 (기본 호환)
def safe_delete_chromadb(max_retries=3, delay=1):
    """ChromaDB 폴더를 안전하게 삭제합니다."""
    return safe_delete_chromadb_cross_platform("./chroma_db", max_retries, delay)

# 앱 시작 시 ChromaDB 정리 (개선된 버전)
if 'app_initialized' not in st.session_state:
    print(f"앱 초기화 시작 - OS: {SYSTEM_OS}, 배포환경: {IS_DEPLOYMENT}")
    
    # 전체 ChromaDB 폴더 정리
    cleanup_all_chromadb()
    
    st.session_state.app_initialized = True
    print("앱 초기화 완료")

# RAG 시스템 초기화 (크로스 플랫폼 개선 버전)
@st.cache_resource
def initialize_rag_system(file_path):
    """RAG 시스템을 초기화합니다."""
    try:
        print(f"RAG 시스템 초기화 시작: {file_path}")
        
        # PDF 문서 로딩
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"문서 로딩 완료: {len(documents)}개 페이지")
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        print(f"텍스트 분할 완료: {len(splits)}개 청크")
        
        # 임베딩 모델 설정 (한국어 지원)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # 고유한 ChromaDB 디렉토리 사용 (충돌 방지)
        timestamp = int(time.time())
        random_id = uuid.uuid4().hex[:8]
        chroma_dir = f"./chroma_db_{timestamp}_{random_id}"
        
        print(f"새 ChromaDB 디렉토리: {chroma_dir}")
        
        # 기존 chroma_db 폴더가 있다면 정리
        old_chroma_dir = "./chroma_db"
        if os.path.exists(old_chroma_dir):
            try:
                safe_delete_chromadb_cross_platform(old_chroma_dir)
                print(f"기존 ChromaDB 디렉토리 정리됨: {old_chroma_dir}")
            except Exception as e:
                print(f"기존 ChromaDB 디렉토리 정리 실패: {str(e)}")

        # 새로운 ChromaDB 벡터 저장소 생성
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=chroma_dir
        )
        
        print("RAG 시스템 초기화 완료")
        return vectorstore
        
    except Exception as e:
        error_msg = f"RAG 시스템 초기화 중 오류 발생: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return None

# 업로드된 파일을 임시 파일로 저장 (크로스 플랫폼)
def save_uploaded_file(uploaded_file):
    """업로드된 파일을 임시 파일로 저장하고 경로를 반환합니다."""
    try:
        # 크로스 플랫폼 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="uploaded_") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        print(f"임시 파일 생성: {temp_path}")
        return temp_path
        
    except Exception as e:
        error_msg = f"파일 저장 중 오류 발생: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return None

# RAG 질의응답 함수 (스트리밍 버전)
def get_rag_response_streaming(question, vectorstore, api_key, container):
    """RAG를 사용하여 질문에 답변합니다 (스트리밍 방식)."""
    try:
        # 스트리밍 콜백 핸들러 생성
        callback_handler = StreamlitCallbackHandler(container)
        
        # LLM 설정 (스트리밍 활성화)
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=0,
            streaming=True,
            callbacks=[callback_handler]
        )
        
        # 프롬프트 템플릿 설정
        prompt_template = """
        당신은 문서 분석 전문가입니다. 제공된 문서를 바탕으로 정확하고 유용한 답변을 제공해주세요.
        문서에 없는 내용은 추측하지 말고, 모를 경우 솔직히 말씀해주세요.
        
        문서 내용:
        {context}
        
        질문: {question}
        
        답변:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RetrievalQA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # 질의응답 실행
        response = qa_chain.invoke({"query": question})
        return callback_handler.text  # 스트리밍된 전체 텍스트 반환
        
    except Exception as e:
        error_msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        print(error_msg)
        container.error(error_msg)
        return error_msg

# 문서 요약 함수 추가
def generate_document_summary(vectorstore, api_key):
    """업로드된 문서의 종합적인 요약을 생성합니다."""
    try:
        # LLM 설정 (요약용)
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=0.1
        )
        
        # 문서 요약용 프롬프트 템플릿
        summary_prompt = """
        제공된 문서를 다음과 같이 간단히 요약해주세요:

        **📄 문서 정보**
        - 문서 유형과 주제

        **📝 핵심 내용**
        - 중요한 내용 3-4개 요약

        **⚠️ 주요 포인트**
        - 날짜, 금액, 조건 등 중요 정보

        문서 내용:
        {context}
        """
        
        SUMMARY_PROMPT = PromptTemplate(
            template=summary_prompt,
            input_variables=["context"]
        )
        
        # 문서 검색 및 요약
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # 더 많은 컨텍스트 수집
        
        # 관련 문서들 검색
        docs = retriever.get_relevant_documents("문서 전체 내용 요약")
        
        # 문서 내용 결합
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 요약 생성
        formatted_prompt = SUMMARY_PROMPT.format(context=context)
        response = llm.invoke(formatted_prompt)
        
        return response.content
        
    except Exception as e:
        error_msg = f"문서 요약 생성 중 오류 발생: {str(e)}"
        print(error_msg)
        return f"❌ 문서 요약 생성에 실패했습니다: {str(e)}"

# 추천 질문 생성 함수 추가
def generate_recommended_questions(vectorstore, api_key):
    """문서 이해를 위한 핵심 질문들을 생성합니다."""
    try:
        # LLM 설정
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=0.1
        )
        
        # 추천 질문 생성용 프롬프트
        questions_prompt = """
        제공된 문서를 바탕으로 이 문서를 이해하는 데 도움이 될 수 있는 핵심 질문들을 5-7개 생성해주세요.
        질문은 구체적이고 실용적이어야 하며, 문서의 중요한 내용을 다뤄야 합니다.
        
        각 질문은 한 줄씩 다음 형식으로 작성해주세요:
        - 질문 내용
        
        문서 내용:
        {context}
        """
        
        QUESTIONS_PROMPT = PromptTemplate(
            template=questions_prompt,
            input_variables=["context"]
        )
        
        # 문서 검색
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        docs = retriever.get_relevant_documents("문서의 핵심 내용과 중요한 정보")
        
        # 문서 내용 결합
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 질문 생성
        formatted_prompt = QUESTIONS_PROMPT.format(context=context)
        response = llm.invoke(formatted_prompt)
        
        # 질문들을 리스트로 파싱
        questions_text = response.content
        questions = []
        
        for line in questions_text.split('\n'):
            line = line.strip()
            if line.startswith('-') and len(line) > 3:
                question = line[1:].strip()
                if question and '?' in question:
                    questions.append(question)
        
        return questions[:7]  # 최대 7개 질문만 반환
        
    except Exception as e:
        error_msg = f"추천 질문 생성 중 오류 발생: {str(e)}"
        print(error_msg)
        return []

# 추가 질문 생성 함수
def generate_additional_questions(vectorstore, api_key, existing_questions):
    """기존 질문들과 중복되지 않는 추가 핵심 질문들을 생성합니다."""
    try:
        # LLM 설정
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=0.2  # 다양성을 위해 temperature 약간 증가
        )
        
        # 기존 질문들을 텍스트로 변환
        existing_questions_text = "\n".join([f"- {q}" for q in existing_questions])
        
        # 추가 질문 생성용 프롬프트
        additional_questions_prompt = """
        제공된 문서를 바탕으로 추가적인 핵심 질문들을 5-6개 생성해주세요.
        아래 기존 질문들과 중복되지 않도록 다른 관점이나 세부사항에 대한 질문을 만들어주세요.
        
        **기존 질문들 (중복 금지):**
        {existing_questions}
        
        **새로운 질문 생성 조건:**
        - 기존 질문들과 완전히 다른 내용이어야 함
        - 문서의 다른 측면이나 세부사항에 초점
        - 구체적이고 실용적인 질문
        - 문서 내용을 깊이 이해할 수 있는 질문
        
        각 질문은 한 줄씩 다음 형식으로 작성해주세요:
        - 질문 내용
        
        문서 내용:
        {context}
        """
        
        ADDITIONAL_QUESTIONS_PROMPT = PromptTemplate(
            template=additional_questions_prompt,
            input_variables=["existing_questions", "context"]
        )
        
        # 문서 검색 (다른 관점으로)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        docs = retriever.get_relevant_documents("문서의 세부사항과 추가 정보")
        
        # 문서 내용 결합
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 추가 질문 생성
        formatted_prompt = ADDITIONAL_QUESTIONS_PROMPT.format(
            existing_questions=existing_questions_text,
            context=context
        )
        response = llm.invoke(formatted_prompt)
        
        # 질문들을 리스트로 파싱
        questions_text = response.content
        new_questions = []
        
        for line in questions_text.split('\n'):
            line = line.strip()
            if line.startswith('-') and len(line) > 3:
                question = line[1:].strip()
                if question and '?' in question:
                    # 기존 질문과 중복 체크 (간단한 키워드 비교)
                    is_duplicate = False
                    for existing_q in existing_questions:
                        # 질문의 핵심 키워드가 70% 이상 겹치면 중복으로 판단
                        existing_words = set(existing_q.lower().split())
                        new_words = set(question.lower().split())
                        if len(existing_words & new_words) / max(len(existing_words), len(new_words)) > 0.7:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        new_questions.append(question)
        
        return new_questions[:6]  # 최대 6개 추가 질문 반환
        
    except Exception as e:
        error_msg = f"추가 질문 생성 중 오류 발생: {str(e)}"
        print(error_msg)
        return []

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'current_file' not in st.session_state:
    st.session_state.current_file = None

if 'document_summary' not in st.session_state:
    st.session_state.document_summary = None

if 'recommended_questions' not in st.session_state:
    st.session_state.recommended_questions = []

# 파일 업로드 처리
if uploaded_file:
    # 새 파일이 업로드되었는지 확인
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.vectorstore = None  # 기존 벡터 스토어 초기화
        st.session_state.messages = []  # 대화 히스토리 초기화
        st.session_state.document_summary = None  # 문서 요약 초기화
        st.session_state.recommended_questions = []  # 추천 질문 초기화
        
        # 파일 저장 및 RAG 시스템 초기화
        with st.spinner("📄 PDF 문서를 분석하고 임베딩하고 있습니다... 잠시만 기다려주세요!"):
            temp_file_path = save_uploaded_file(uploaded_file)
            if temp_file_path:
                st.session_state.vectorstore = initialize_rag_system(temp_file_path)
                
                # 임시 파일 정리 (크로스 플랫폼)
                try:
                    os.unlink(temp_file_path)
                    print(f"임시 파일 정리됨: {temp_file_path}")
                except Exception as e:
                    print(f"임시 파일 정리 실패: {str(e)}")
                
                if st.session_state.vectorstore:
                    st.success("✅ 문서가 성공적으로 분석되었습니다!")
                    
                    # 문서 요약 및 추천 질문 자동 생성
                    if openai_api_key:
                        with st.spinner("📝 문서 요약을 생성하고 있습니다..."):
                            summary = generate_document_summary(st.session_state.vectorstore, openai_api_key)
                            st.session_state.document_summary = summary
                            
                        with st.spinner("🤔 핵심 질문들을 생성하고 있습니다..."):
                            questions = generate_recommended_questions(st.session_state.vectorstore, openai_api_key)
                            st.session_state.recommended_questions = questions
                            
                        st.success("✅ 문서 요약과 핵심 질문들이 완성되었습니다!")
                    else:
                        st.warning("⚠️ API 키가 없어 문서 요약을 생성할 수 없습니다.")
                else:
                    st.error("❌ 문서 분석에 실패했습니다.")
else:
    # 파일이 삭제된 경우 처리
    if st.session_state.current_file is not None:
        # 안전한 ChromaDB 폴더 삭제
        with st.spinner("🗑️ 기존 문서 데이터를 정리하고 있습니다..."):
            success = cleanup_all_chromadb()
            if success:
                st.info("🗑️ 기존 문서 데이터가 정리되었습니다.")
            else:
                st.warning("⚠️ 데이터 정리가 지연되었습니다. 다음 앱 재시작 시 자동으로 정리됩니다.")
        
        # 세션 상태 초기화
        st.session_state.current_file = None
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.session_state.document_summary = None
        st.session_state.recommended_questions = []

# 파일이 업로드되지 않았을 때 안내
if not uploaded_file:
    st.info("📁 먼저 사이드바에서 PDF 파일을 업로드해주세요!")
    st.markdown("""
    ### 📋 사용 방법
    1. **파일 업로드**: 사이드바에서 PDF 파일 업로드
    2. **문서 분석**: 업로드된 문서가 자동으로 분석됩니다
    3. **질문하기**: 문서 내용에 대해 질문해보세요
    
    ### 📄 지원 파일 형식
    - PDF 파일만 지원됩니다
    - 텍스트가 포함된 PDF 파일이어야 합니다
    """)
else:
    # 문서 요약 표시 (파일이 업로드되고 요약이 있을 때)
    if st.session_state.document_summary:
        st.markdown("---")
        st.markdown("## 📄 문서 요약")
        
        # 요약을 예쁘게 표시
        with st.container():
            st.markdown(st.session_state.document_summary)
        
        # 추천 질문 버튼들 표시
        if st.session_state.recommended_questions:
            st.markdown("---")
            st.markdown("## 🤔 핵심 질문들")
            st.info("💡 아래 질문들을 클릭하면 바로 답변을 받을 수 있습니다!")
            
            # 질문 버튼들을 2열로 배치
            cols = st.columns(2)
            for idx, question in enumerate(st.session_state.recommended_questions):
                col_idx = idx % 2
                with cols[col_idx]:
                    # 질문을 버튼으로 표시 (키 값에 인덱스 추가로 중복 방지)
                    if st.button(
                        question, 
                        key=f"question_btn_{idx}",
                        help="클릭하면 이 질문에 대한 답변이 생성됩니다",
                        use_container_width=True
                    ):
                        # 버튼이 클릭되면 해당 질문을 자동으로 처리
                        if st.session_state.vectorstore and openai_api_key:
                            # 사용자 질문으로 추가
                            st.session_state.messages.append({"role": "user", "content": question})
                            
                            # 답변 생성
                            with st.spinner("🤖 답변을 생성하고 있습니다..."):
                                # 임시 컨테이너 생성 (실제로 표시하지 않기 위해)
                                temp_container = st.empty()
                                ai_response = get_rag_response_streaming(
                                    question,
                                    st.session_state.vectorstore,
                                    openai_api_key,
                                    temp_container
                                )
                                temp_container.empty()  # 임시 컨테이너 정리
                            
                            # AI 응답을 메시지에 추가
                            st.session_state.messages.append({"role": "assistant", "content": ai_response})
                            
                            # 페이지 새로고침으로 채팅 인터페이스 업데이트
                            st.rerun()
            
            # 더 많은 질문 생성 버튼 추가
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "🔄 더 많은 질문 생성하기",
                    key="generate_more_questions",
                    help="추가적인 핵심 질문들을 생성합니다",
                    use_container_width=True,
                    type="secondary"
                ):
                    if st.session_state.vectorstore and openai_api_key:
                        with st.spinner("🤔 추가 핵심 질문들을 생성하고 있습니다..."):
                            # 기존 질문들과 중복되지 않는 새로운 질문들 생성
                            new_questions = generate_additional_questions(
                                st.session_state.vectorstore, 
                                openai_api_key, 
                                st.session_state.recommended_questions
                            )
                            
                            if new_questions:
                                # 새로운 질문들을 기존 질문들에 추가
                                st.session_state.recommended_questions.extend(new_questions)
                                st.success(f"✅ {len(new_questions)}개의 새로운 질문이 추가되었습니다!")
                                st.rerun()  # 페이지 새로고침으로 새 질문들 표시
                            else:
                                st.warning("⚠️ 추가 질문 생성에 실패했습니다. 다시 시도해주세요.")
                    else:
                        st.error("❌ 문서가 분석되지 않았거나 API 키가 설정되지 않았습니다.")
        
        st.markdown("---")
        st.markdown("### 💬 직접 질문하기")
        st.info("📝 위 질문들 외에도 문서에 대해 자유롭게 질문해보세요!")
    
    # 기존 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if user_question := st.chat_input(placeholder="업로드된 문서에 대해 질문해주세요... 💬"):
        # API 키 확인
        if not openai_api_key:
            st.error("OpenAI API 키를 먼저 설정해주세요! .env 파일을 확인해주세요.")
        elif st.session_state.vectorstore is None:
            st.error("문서 분석이 완료되지 않았습니다. 잠시 후 다시 시도해주세요.")
        else:
            # 사용자 질문 표시
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

            # AI 답변 생성 및 스트리밍 표시
            with st.chat_message("assistant"):
                # 스트리밍을 위한 빈 컨테이너 생성
                message_container = st.empty()
                
                # 스트리밍으로 답변 생성
                ai_response = get_rag_response_streaming(
                    user_question, 
                    st.session_state.vectorstore, 
                    openai_api_key,
                    message_container
                )
                
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

# 하단에 사용 팁 추가
if uploaded_file:
    st.divider()
    with st.expander("💡 사용 팁"):
        st.markdown(f"""
        **현재 분석 중인 파일: {uploaded_file.name}**
        **시스템 환경: {SYSTEM_OS.title()}** {'(배포환경)' if IS_DEPLOYMENT else '(로컬환경)'}
        
        **이 챗봇은 어떻게 작동하나요?**
        - 📄 업로드된 PDF 문서의 내용을 기반으로 답변합니다
        - 🔍 관련 정보를 검색하여 정확한 답변을 제공합니다
        - ⚡ 실시간 스트리밍으로 답변이 생성됩니다
        
        **더 나은 답변을 위한 팁:**
        - 구체적이고 명확한 질문을 해주세요
        - 문서의 특정 부분에 대해 질문해보세요
        - 문서에 없는 내용은 솔직히 "모른다"고 답변할 수 있습니다
        """)

    # 디버깅용 (개발 환경에서만)
    if st.checkbox("🔧 디버그 모드"):
        st.json({
            "메시지 개수": len(st.session_state.messages),
            "OS": SYSTEM_OS,
            "배포환경": IS_DEPLOYMENT,
            "Python 버전": sys.version
        })
        with st.expander("전체 대화 내역"):
            st.json(st.session_state.messages)