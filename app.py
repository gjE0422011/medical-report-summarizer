import streamlit as st
import pymupdf
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="Medical Report Summarizer",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Report Summarizer")
st.markdown("Upload a medical report (PDF) and get a plain-English summary instantly.")
st.divider()

hf_token = st.text_input(
    "🔑 Enter your Hugging Face Token",
    type="password",
    placeholder="hf_..."
)

uploaded_file = st.file_uploader("📄 Upload Medical Report (PDF)", type=["pdf"])

def extract_text(file_bytes):
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def summarize_report(token, report_text):
    client = InferenceClient(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        token=token
    )

    response = client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": f"""You are a medical assistant helping patients understand their reports.

Here is a medical report:
{report_text[:3000]}

Give a structured summary with these sections:

1. **What This Report Is About** - 1-2 simple sentences
2. **Key Findings** - plain English, no jargon
3. **Values Outside Normal Range** - highlight abnormal values
4. **What This Could Mean** - simple explanation
5. **Suggested Next Steps** - what the patient should do
6. **Urgency Level** - one of: 🟢 Routine / 🟡 Follow Up Soon / 🔴 See Doctor Promptly

End with: "This summary is for informational purposes only and does not replace professional medical advice."
"""
            }
        ],
        max_tokens=800,
        temperature=0.4,
    )

    return response.choices[0].message.content

if uploaded_file and hf_token:
    if st.button("🔍 Summarize Report"):
        with st.spinner("Analyzing your report... this may take 20-30 seconds"):
            try:
                file_bytes = uploaded_file.read()
                report_text = extract_text(file_bytes)

                if not report_text:
                    st.error("Could not extract text. The PDF may be a scanned image.")
                elif len(report_text) < 50:
                    st.warning("Very little text found. Results may be limited.")
                else:
                    summary = summarize_report(hf_token, report_text)
                    st.divider()
                    st.subheader("📋 Your Report Summary")
                    st.markdown(summary)
                    st.divider()
                    st.download_button(
                        label="⬇️ Download Summary",
                        data=summary,
                        file_name="medical_summary.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

elif uploaded_file and not hf_token:
    st.warning("Please enter your Hugging Face token above.")
elif hf_token and not uploaded_file:
    st.info("Please upload a PDF to continue.")

st.divider()
st.caption("Built with Python · Mistral 7B · Hugging Face · Streamlit  |  For informational use only.")
