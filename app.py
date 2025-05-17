import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

# Initialize Gemini API client
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
model_id = "gemini-2.0-flash"

# Streamlit UI
st.set_page_config(page_title="LinkedIn Post Generator", layout="centered")
st.title("LinkedIn Post Generator")

# User input
user_input = st.text_input("📝 What would you like to make a post about?", placeholder="e.g. OpenAI's new Codex model...")

if user_input:
    with st.spinner("🔍 Collecting information using Gemini + Google Search..."):
        google_search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        response = client.models.generate_content(
            model=model_id,
            contents=user_input,
            config=types.GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            )
        )

        search_result_text = "".join(part.text for part in response.candidates[0].content.parts)
        st.subheader("🔎 Search Info Collected")
        st.write(search_result_text)

    with st.spinner("🪄 Crafting your funky LinkedIn post..."):
        linkedin_prompt = f"""
        Rewrite the following information into a professional, engaging LinkedIn post suitable for sharing tech updates in a funky way. Add emojis and hashtags:
        {search_result_text}
        Make it concise, friendly, and informative.
        """
        linkedin_response = client.models.generate_content(
            model=model_id,
            contents=linkedin_prompt
        )

        st.subheader("💼 LinkedIn-style Post")
        st.write(linkedin_response.text)

    with st.spinner("🎨 Creating an image prompt..."):
        image_prompt = f"""
        Based on the following topic and insights {search_result_text}, generate only one image prompt that can be used for a LinkedIn post. 
        The image should visually reflect the theme and innovations mentioned.
        """
        image_prompt_response = client.models.generate_content(
            model=model_id,
            contents=image_prompt
        )
        image_prompt_text = image_prompt_response.text
        st.subheader("🖼️ Image Prompt")
        st.write(image_prompt_text)

    with st.spinner("🧠 Generating image from prompt..."):
        final_image_prompt = f"""
        Generate an image based on the following prompt: {image_prompt_text}. 
        Make it vivid and professional to match the LinkedIn post.
        """
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=final_image_prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )

        st.subheader("📸 AI-Generated Visual")
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                st.write(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                st.image(image, caption="Generated Image", use_column_width=True)
