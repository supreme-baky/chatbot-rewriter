# app.py
import subprocess
import streamlit as st
from agent import chatbot

st.set_page_config(page_title="Offline Rewriting Coach", page_icon="✍️", layout="centered")

st.title("✍️ Offline Rewriting Coach")
st.caption("Paste text (≥50 words) to rewrite using Strunk & White + Williams & Bizup style principles.")

with st.sidebar:
    st.header("Tools")
    if st.button("Rebuild vectorstore from ./data"):
        st.info("Rebuilding vectorstore — this may take a few minutes. See console for progress.")
        try:
            # call rebuild_chroma.py; user must have Python env active
            subprocess.Popen(["python", "rebuild_chroma.py"])
            st.success("Rebuild process started (runs in background).")
        except Exception as e:
            st.error(f"Failed to start rebuild: {e}")

# Input area
user_input = st.text_area("Paste your text here:", height=300, placeholder="Paste paragraphs to rewrite...")

col1, col2 = st.columns([1, 4])
with col1:
    submit = st.button("Rewrite / Ask")
with col2:
    st.write("Tip: For rewrite, paste ≥50 words. For short questions, the app will answer referencing your uploaded texts.")

if submit:
    if not user_input or not user_input.strip():
        st.warning("Please paste some text or a short question.")
    else:
        with st.spinner("Processing — generating revision..."):
            try:
                out = chatbot(user_input)
                if out.startswith("Revised Version:"):
                    st.markdown("**Revised Version:**")
                    st.write(out.replace("Revised Version:", "").strip())
                else:
                    st.markdown("**Answer:**")
                    st.write(out.strip())
            except Exception as e:
                st.error(f"Error during generation: {e}")

# Footer
st.markdown("---")
st.write("Model & vectorstore should be cached locally for offline operation. Put your PDFs in `./data` and run `python rebuild_chroma.py` to update the reference content.")
