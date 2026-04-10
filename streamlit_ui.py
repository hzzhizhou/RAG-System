import streamlit as st
import requests
import uuid
from datetime import datetime
from config.settings import DASHSCOPE_API_KEY

# 页面配置
st.set_page_config(
    page_title="企业级 RAG + Agent 智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .context-box {
        background-color: #ffffff;
        border-left: 4px solid #1f77b4;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .metric {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 初始化会话状态 ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_001"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = DASHSCOPE_API_KEY
if "mode" not in st.session_state:
    st.session_state.mode = "rag"  # 默认 RAG 模式
if "rag_backend" not in st.session_state:
    st.session_state.rag_backend = "http://127.0.0.1:8000/rag/query"
if "agent_backend" not in st.session_state:
    st.session_state.agent_backend = "http://127.0.0.1:8000/agent/query"

# ---------- 侧边栏配置 ----------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/chat.png", width=80)
    st.markdown("## ⚙️ 系统配置")

    # 模式选择
    mode = st.radio(
        "选择模式",
        options=["rag", "agent"],
        format_func=lambda x: "RAG 模式（固定检索+生成）" if x == "rag" else "Agent 模式（自主决策+工具调用）",
        index=0 if st.session_state.mode == "rag" else 1,
        key="mode_radio"
    )
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        # 切换模式时清空历史（可选）
        # st.session_state.messages = []

    # API 密钥输入
    api_key_input = st.text_input(
        "API 密钥",
        type="password",
        value=st.session_state.api_key,
        help="请使用你在 .env 中配置的 DASHSCOPE_API_KEY"
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    # 后端地址（显示当前模式对应的地址，但允许用户手动覆盖）
    if st.session_state.mode == "rag":
        backend_url = st.text_input(
            "后端服务地址 (RAG)",
            value=st.session_state.rag_backend,
            help="RAG 接口地址，流式模式会自动将 /rag/query 替换为 /rag/stream"
        )
        if backend_url != st.session_state.rag_backend:
            st.session_state.rag_backend = backend_url
    else:
        backend_url = st.text_input(
            "后端服务地址 (Agent)",
            value=st.session_state.agent_backend,
            help="Agent 接口地址"
        )
        if backend_url != st.session_state.agent_backend:
            st.session_state.agent_backend = backend_url

    st.divider()

    # RAG 专用配置（仅在 RAG 模式下显示）
    if st.session_state.mode == "rag":
        st.markdown("### 🧪 检索增强选项")
        route_mode = st.selectbox(
            "路由模式",
            ["rule", "llm", "hybrid"],
            index=0,
            help="rule: 规则路由; llm: 大模型路由; hybrid: 始终使用混合检索"
        )
        use_context = st.checkbox("上下文改写（多轮对话）", value=True, help="启用对话历史改写当前问题")
        use_hyde = st.checkbox("HyDE 假设文档", value=False, help="生成假设文档增强检索")
        use_multi = st.checkbox("多查询扩展", value=False, help="生成多个查询并行检索")
    else:
        # Agent 模式下隐藏 RAG 配置，但保留占位（避免变量未定义）
        route_mode = "rule"
        use_context = True
        use_hyde = False
        use_multi = False

    st.divider()
    st.markdown("### 🗑️ 会话管理")
    if st.button("清空当前会话"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.caption(f"当前会话ID: `{st.session_state.session_id[:8]}...`")

# ---------- 主界面 ----------
st.markdown('<div class="main-header">🤖 企业级 RAG + Agent 智能问答系统</div>', unsafe_allow_html=True)
st.markdown(f"当前模式：**{'RAG 模式' if st.session_state.mode == 'rag' else 'Agent 模式'}**")

# 显示历史消息
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入你的问题..."):
    if not st.session_state.api_key:
        st.error("❌ 请先在侧边栏配置 API 密钥")
        st.stop()

    # 添加用户消息到界面
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 根据模式构建请求
    if st.session_state.mode == "rag":
        # RAG 模式：使用流式端点
        payload = {
            "question": prompt,
            "route_mode": route_mode,
            "api_key": st.session_state.api_key,
            "session_id": st.session_state.session_id,
            "use_context": use_context,
            "use_hyde": use_hyde,
            "use_multi": use_multi
        }
        # 自动将 /rag/query 替换为 /rag/stream，如果用户已手动填写流式端点则保持不变
        base_url = st.session_state.rag_backend
        if "/rag/query" in base_url:
            stream_url = base_url.replace("/rag/query", "/rag/stream")
        else:
            stream_url = base_url  # 假设用户已填写正确流式端点
        
        try:
            with st.chat_message("assistant"):
                response = requests.post(stream_url, json=payload, stream=True, timeout=30)
                if response.status_code == 200:
                    # 使用生成器逐块输出
                    def text_generator():
                        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                yield chunk
                    # st.write_stream 会自动逐字显示并返回完整内容
                    answer = st.write_stream(text_generator())
                    
                    # 可选：显示额外的元数据（流式响应通常不包含，可单独请求）
                    with st.expander("📊 详细信息"):
                        st.caption(f"流式响应完成 | 会话: {st.session_state.session_id[:8]}...")
                else:
                    # 尝试获取错误信息（非流式）
                    try:
                        error_detail = response.json().get("detail", "未知错误")
                    except:
                        error_detail = response.text
                    st.error(f"请求失败 (HTTP {response.status_code}): {error_detail}")
                    answer = f"❌ 服务出错：{error_detail}"
            
            if answer:
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
        except requests.exceptions.Timeout:
            st.error("请求超时，请稍后重试。")
        except requests.exceptions.ConnectionError:
            st.error("无法连接到后端服务，请确认服务已启动。")
        except Exception as e:
            st.error(f"发生错误：{str(e)}")
    
    else:
        # Agent 模式：保持原有非流式请求（Agent 暂不支持流式）
        payload = {
            "question": prompt,
            "api_key": st.session_state.api_key,
            "session_id": st.session_state.session_id
        }
        url = st.session_state.agent_backend
        try:
            with st.spinner("Agent 正在思考中..."):
                response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "无返回内容")
                response_time = data.get("response_time", 0)
                session_id = data.get("session_id", st.session_state.session_id)
                
                if session_id != st.session_state.session_id:
                    st.session_state.session_id = session_id
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("📊 详细信息"):
                        st.caption(f"Agent 模式 | 耗时: {response_time}秒 | 会话: {session_id[:8]}...")
                        if "intermediate_steps" in data:
                            st.markdown("**工具调用过程:**")
                            for step in data["intermediate_steps"]:
                                st.text(f"工具: {step[0].tool} | 输入: {step[0].tool_input} | 输出: {step[1]}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_detail = response.json().get("detail", "未知错误")
                st.error(f"请求失败 (HTTP {response.status_code}): {error_detail}")
                error_msg = f"❌ 服务出错：{error_detail}"
                with st.chat_message("assistant"):
                    st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except requests.exceptions.Timeout:
            st.error("请求超时，请稍后重试。")
        except requests.exceptions.ConnectionError:
            st.error("无法连接到后端服务，请确认服务已启动。")
        except Exception as e:
            st.error(f"发生错误：{str(e)}")