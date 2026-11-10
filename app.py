import streamlit as st
import ollama
from datetime import datetime 
from pathlib import path
import time

CHAT_DIR = path ('chats')
SUMMARY_DIR = ('summaries')
CHAT_DIR.mkdir(exist_ok=True)
SUMMARY_DIR.mkdir(exist_ok=True)

MODEL_NAME = 'GEMMA3:1b'

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False


def format_chat_for_file(messages: list) -> str:
    """تبدیل لیست پیام‌ها به متن قابل ذخیره"""
    lines = []
    for msg in messages:
        role = "کاربر" if msg["role"] == "user" else "دستیار"
        lines.append(f"[{role}]:\n{msg['content']}\n")
    return "\n".join(lines)


def save_chat_to_file(messages: list) -> Path | None:
    """ذخیره کل چت در فایل متنی"""
    if not messages:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = CHAT_DIR / f"chat_{timestamp}.txt"
    content = format_chat_for_file(messages)
    filename.write_text(content, encoding="utf-8")
    return filename


def summarize_chat(messages: list) -> str:
    """خلاصه کردن کل چت با استفاده از مدل و پرامپت بهینه"""
    if not messages:
        return "مکالمه کافی برای خلاصه‌سازی وجود ندارد."

    chat_text = format_chat_for_file(messages)
    prompt = SUMMARY_PROMPT.format(chat_text=chat_text)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return response["message"]["content"].strip()


def save_summary(summary: str, chat_file: Path | None = None) -> Path:
    """ذخیره خلاصه در فایل جدید"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if chat_file:
        base = chat_file.stem
        filename = SUMMARY_DIR / f"summary_of_{base}.txt"
    else:
        filename = SUMMARY_DIR / f"summary_{timestamp}.txt"

    header = f"خلاصه مکالمه\nتاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    if chat_file:
        header += f"فایل اصلی: {chat_file.name}\n"
    header += "-" * 50 + "\n\n"

    filename.write_text(header + summary, encoding="utf-8")
    return filename


def stream_response(messages: list):
    """تولید پاسخ به صورت استریم با قابلیت توقف"""
    st.session_state.stop_generation = False
    st.session_state.is_generating = True

    full_response = ""
    placeholder = st.empty()

    try:
        stream = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if st.session_state.stop_generation:
                full_response += "\n\n[تولید پاسخ متوقف شد]"
                break

            content = chunk.get("message", {}).get("content", "")
            if content:
                full_response += content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    except Exception as e:
        full_response = f"خطا در ارتباط با Ollama: {e}"
        placeholder.error(full_response)

    finally:
        st.session_state.is_generating = False

    return full_response


def main():
    st.set_page_config(
        page_title="چت با Gemma3-1B",
        page_icon="💬",
        layout="centered",
    )

    init_session_state()

    st.title("💬 چت با Gemma3-1B (Ollama)")
    st.caption("مدل: gemma3:1b | قابلیت توقف و خلاصه‌سازی")

    # ---------- سایدبار ----------
    with st.sidebar:
        st.header("⚙️ کنترل‌ها")

        if st.session_state.is_generating:
            if st.button("⏹️ توقف تولید پاسخ", use_container_width=True, type="primary"):
                st.session_state.stop_generation = True
                st.toast("درخواست توقف ارسال شد...", icon="⏹️")
        else:
            st.button("⏹️ توقف تولید پاسخ", use_container_width=True, disabled=True)

        st.divider()

        # دکمه پاک کردن تاریخچه
        if st.button("🗑️ پاک کردن تاریخچه", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.stop_generation = False
            st.session_state.is_generating = False
            st.rerun()

        st.divider()

        # دکمه ذخیره چت
        if st.button("💾 ذخیره چت در فایل", use_container_width=True):
            if st.session_state.messages:
                path = save_chat_to_file(st.session_state.messages)
                st.success(f"ذخیره شد:\n`{path}`")
            else:
                st.warning("چتی برای ذخیره وجود ندارد.")

        # دکمه خلاصه‌سازی
        if st.button("📝 خلاصه کردن کل چت", use_container_width=True, type="primary"):
            if not st.session_state.messages:
                st.warning("چتی برای خلاصه وجود ندارد.")
            else:
                with st.spinner("در حال خلاصه‌سازی با مدل..."):
                    # اول چت را ذخیره کن
                    chat_path = save_chat_to_file(st.session_state.messages)
                    # سپس خلاصه بگیر
                    summary = summarize_chat(st.session_state.messages)
                    # خلاصه را ذخیره کن
                    summary_path = save_summary(summary, chat_path)

                st.success("✅ خلاصه آماده شد!")
                st.markdown("**خلاصه:**")
                st.info(summary)
                st.caption(f"فایل چت: `{chat_path}`")
                st.caption(f"فایل خلاصه: `{summary_path}`")

        st.divider()
        st.markdown("### راهنما")
        st.markdown(
            """
            1. پیام خود را بنویسید.
            2. در حین تولید پاسخ، از دکمه **⏹️ توقف تولید پاسخ** در سایدبار استفاده کنید.
            3. از سایدبار می‌توانید تاریخچه را پاک کنید، چت را ذخیره کنید یا خلاصه بگیرید.
            """
        )
        st.caption("مطمئن شوید Ollama در حال اجراست و مدل `gemma3:1b` دانلود شده.")

    # ---------- نمایش تاریخچه چت ----------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------- ورودی کاربر ----------
    if prompt := st.chat_input("پیام خود را بنویسید...", disabled=st.session_state.is_generating):
        # افزودن پیام کاربر
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # تولید پاسخ
        with st.chat_message("assistant"):
            response = stream_response(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()