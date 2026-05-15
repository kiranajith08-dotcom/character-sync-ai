import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import io
import zipfile

# ============ CUSTOMIZE THESE ============
APP_TITLE = "🎨 CharacterSync AI with unlimited generation"
APP_DESCRIPTION = "Generate consistent character variations in your favorite style"
DEFAULT_PROMPT = "a character, anime style, full body"
DEFAULT_NEGATIVE = "blurry, low quality, deformed, bad anatomy"
# ========================================

# Page setup
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BOT LOGIC ---
def get_bot_response(user_input):
    user_input = user_input.lower()
    if "prompt" in user_input:
        return "Be specific! Add details like 'cyberpunk', 'watercolor', or 'cinematic lighting' for better results."
    elif "strength" in user_input:
        return "Strength (0.5-0.95) determines how much the AI changes the original. 0.95 is a big change; 0.5 is a small tweak."
    elif "quality" in user_input or "steps" in user_input:
        return "Higher steps (50) look better but take longer. 20-30 is usually the sweet spot for speed."
    elif "hello" in user_input or "hi" in user_input:
        return "Hello! I'm your helper. Ask me about prompts or settings!"
    else:
        return "I can help with settings or prompt ideas! Try asking: 'How does strength work?' or 'Give me a prompt tip'."

# Title
st.title(APP_TITLE)
st.write(APP_DESCRIPTION)

# Sidebar settings
st.sidebar.header("⚙️ Generation Settings")
num_images = st.sidebar.slider("How many variations?", 1, 10, 3)
strength = st.sidebar.slider("Keep original style (higher = more consistent)", 0.5, 0.95, 0.75, step=0.05)
steps = st.sidebar.slider("Quality (higher = better but slower)", 20, 50, 30, step=5)
guidance = st.sidebar.slider("How strictly follow prompt", 5.0, 15.0, 7.5, step=0.5)

# --- CHATBOT UI IN SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.subheader("💬 AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you generate characters today?"}]

# Display chat history in sidebar
for message in st.session_state.messages:
    with st.sidebar.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if chat_query := st.sidebar.chat_input("Ask a question..."):
    with st.sidebar.chat_message("user"):
        st.markdown(chat_query)
    st.session_state.messages.append({"role": "user", "content": chat_query})

    bot_res = get_bot_response(chat_query)
    with st.sidebar.chat_message("assistant"):
        st.markdown(bot_res)
    st.session_state.messages.append({"role": "assistant", "content": bot_res})


# Main content
st.subheader("1️⃣ Upload Reference Image")
st.write("Upload a character or art style you like. The AI will generate variations in that same style.")

uploaded_file = st.file_uploader("Choose an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    ref_image = Image.open(uploaded_file).convert("RGB")
    ref_image_resized = ref_image.resize((512, 512))
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(ref_image, caption="Your uploaded image")
    
    with col2:
        st.success("✅ Image loaded successfully!")
        st.info("👇 Customize the settings on the left, then click 'Generate'")
    
    # Prompts
    st.subheader("2️⃣ Customize Your Prompt")
    
    col1, col2 = st.columns(2)
    with col1:
        prompt = st.text_input(
            "What kind of character?",
            value=DEFAULT_PROMPT,
            help="Be specific: 'a warrior', 'a cute cat girl', 'a cyberpunk assassin', etc."
        )
    
    with col2:
        negative_prompt = st.text_input(
            "What to AVOID? (optional)",
            value=DEFAULT_NEGATIVE,
            help="Things the AI should NOT generate"
        )
    
    # Generate button
    st.subheader("3️⃣ Generate")
    
    if st.button("🚀 Generate Images", key="generate", use_container_width=True):
        st.info(f"⏳ Generating {num_images} image(s)... Please wait...")
        
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
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(num_images):
            status_text.text(f"⏳ Generating image {i+1}/{num_images}...")
            try:
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
                progress_bar.progress((i + 1) / num_images)
            except Exception as e:
                st.error(f"Error generating image {i+1}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if generated_images:
            st.success(f"✅ Successfully generated {len(generated_images)} image(s)!")
            st.subheader("4️⃣ Your Generated Images")
            
            cols = st.columns(3)
            for idx, img in enumerate(generated_images):
                with cols[idx % 3]:
                    st.image(img, caption=f"Variation {idx+1}")
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    st.download_button(
                        label=f"⬇️ Download {idx+1}",
                        data=img_byte_arr,
                        file_name=f"character_{idx+1}.png",
                        mime="image/png",
                        key=f"download_{idx}",
                        use_container_width=True
                    )
            
            # Batch download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                for idx, img in enumerate(generated_images):
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    zip_file.writestr(f"character_{idx+1}.png", img_byte_arr.getvalue())
            
            st.download_button(
                label="📥 Download All as ZIP",
                data=zip_buffer.getvalue(),
                file_name="characters.zip",
                mime="application/zip",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.caption("Made with ❤️ | Powered by Stable Diffusion | Built by Kiran Ajith")
