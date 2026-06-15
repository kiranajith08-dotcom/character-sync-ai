import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import io
import zipfile

# ============ CONFIGURATION ============
APP_TITLE = "🚀 CHARACTER SYNC: Ai SAMPLE MODEL"
APP_DESCRIPTION = "Generate consistent character variations in a futuristic 3D workspace."

# 1. PAGE SETUP (Must be the first Streamlit command)
st.set_page_config(
    page_title="CharacterSync AI Sample",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. FUTURISTIC 3D UI CSS
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at top right, #0a1128, #000000);
        color: #e0e0e0;
    }

    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(0, 242, 255, 0.2);
    }

    /* 3D Floating Containers */
    div[data-testid="stImage"], .stDownloadButton > button, div.stButton > button, .stFileUploader {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        padding: 10px;
    }

    /* Hover 3D Pop & Glow */
    div[data-testid="stImage"]:hover, div.stButton > button:hover {
        transform: translateY(-8px) rotateX(2deg) rotateY(2deg) !important;
        border: 1px solid #00f2ff !important;
        box-shadow: 0 20px 50px rgba(0, 242, 255, 0.3) !important;
    }

    /* Neon Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #00f2ff 0%, #0072ff 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
        border: none !important;
    }

    /* Futuristic Inputs */
    .stTextInput input {
        background: rgba(0, 0, 0, 0.4) !important;
        color: #00f2ff !important;
        border: 1px solid rgba(0, 242, 255, 0.2) !important;
        border-radius: 10px !important;
    }

    /* Title Styling */
    h1 {
        text-shadow: 0 0 20px #00f2ff;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 3px;
    }

    /* Chatbot Bubbles */
    [data-testid="stChatMessage"] {
        background: rgba(0, 242, 255, 0.05) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(0, 242, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. CHATBOT LOGIC
def get_bot_response(user_input):
    user_input = user_input.lower()
    if "prompt" in user_input:
        return "TRY THIS: 'Futuristic warrior, high-tech armor, neon glowing highlights, 8k resolution'."
    elif "strength" in user_input:
        return "STRENGTH TIP: 0.75 is the sweet spot. Lower (0.5) keeps the original shape; Higher (0.9) reinvents the character."
    elif "slow" in user_input or "quality" in user_input:
        return "QUALITY: Increase 'Steps' to 50 for a polished look. Decrease to 20 for speed."
    return "I am your system interface. How can I assist with your generation today?"

# --- SIDEBAR ---
st.sidebar.title("🛠️ CONTROL CENTER")
num_images = st.sidebar.slider("Variations", 1, 10, 3)
strength = st.sidebar.slider("Consistency Strength", 0.5, 0.95, 0.75, 0.05)
steps = st.sidebar.slider("Processing Steps", 20, 50, 30, 5)
guidance = st.sidebar.slider("Prompt Strictness", 5.0, 15.0, 7.5, 0.5)

# --- SIDEBAR CHATBOT ---
st.sidebar.markdown("---")
st.sidebar.subheader("💬 SYSTEM ASSISTANT")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Interface Initialized. Awaiting input..."}]

for msg in st.session_state.messages:
    with st.sidebar.chat_message(msg["role"]):
        st.markdown(msg["content"])

if chat_query := st.sidebar.chat_input("Ask for tips..."):
    st.session_state.messages.append({"role": "user", "content": chat_query})
    with st.sidebar.chat_message("user"):
        st.markdown(chat_query)
    
    bot_res = get_bot_response(chat_query)
    st.session_state.messages.append({"role": "assistant", "content": bot_res})
    with st.sidebar.chat_message("assistant"):
        st.markdown(bot_res)

# --- MAIN INTERFACE ---
st.title("this is a sample model of character sync ai done by me")
st.write(app_description)

st.subheader("1️⃣ TARGET DATA UPLOAD")
uploaded_file = st.file_uploader("Upload Reference Character", type=["jpg", "jpeg", "png"])

if uploaded_file:
    ref_image = Image.open(uploaded_file).convert("RGB")
    ref_image_resized = ref_image.resize((512, 512))
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(ref_image, caption="Base Reference")
    with col2:
        st.success("🛰️ Signal Locked: Image Loaded")
        prompt = st.text_input("MODIFICATION PROMPT", value="a character, anime style, full body")
        negative_prompt = st.text_input("EXCLUDE FROM DATA", value="blurry, low quality, deformed")

    st.subheader("2️⃣ EXECUTE GENERATION")
    if st.button("🚀 INITIATE SYNC", use_container_width=True):
        st.info("🧬 Processing neural layers... please wait.")
        
        @st.cache_resource
        def load_model():
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None
            )
            return pipe.to("cuda")
        
        pipe = load_model()
        generated_images = []
        
        # Generation Loop
        for i in range(num_images):
            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=ref_image_resized,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    generator=torch.Generator(device="cuda").manual_seed(i)
                )
                generated_images.append(output.images[0])
        
        # Results Display
        st.subheader("3️⃣ OUTPUTS GENERATED")
        cols = st.columns(3)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            for idx, img in enumerate(generated_images):
                with cols[idx % 3]:
                    st.image(img, caption=f"Variation {idx+1}")
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    zip_file.writestr(f"char_{idx+1}.png", img_byte_arr.getvalue())
                    st.download_button(f"⬇️ SAVE {idx+1}", data=img_byte_arr.getvalue(), file_name=f"char_{idx+1}.png", mime="image/png")
        
        st.download_button("📥 DOWNLOAD BATCH (ZIP)", data=zip_buffer.getvalue(), file_name="output_sync.zip", use_container_width=True)

st.markdown("---")
st.caption("NEURAL SYNC INTERFACE v3.0 | POWERED BY STABLE DIFFUSION - Created by KIRAN AJITH")
