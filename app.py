import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import io

# --- Configuration and Setup ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the generative AI library
try:
    genai.configure(api_key=API_KEY)
    # Check if API key is actually set
    if not API_KEY:
        st.error("Google API Key not found. Please set it in the .env file.", icon="🚨")
        st.stop() # Halt execution if key is missing

    # Model Configuration (Using Gemini 1.5 Flash - a fast multimodal model)
    # This model replaces the deprecated gemini-pro-vision
    vision_model = genai.GenerativeModel('gemini-1.5-flash') # <--- MODEL UPDATED HERE

except Exception as e:
    st.error(f"Error configuring Google AI SDK: {e}", icon="🚨")
    st.stop()


# --- Helper Functions ---

def get_gemini_response(prompt_text, image_data=None):
    """
    Gets response from Gemini model (handles text and image).

    Args:
        prompt_text (str): The text part of the prompt.
        image_data (bytes, optional): The image data bytes. Defaults to None.

    Returns:
        str: The generated response from the model or an error message.
    """
    model_to_use = vision_model # Using 1.5 flash for both text & image
    response = None # Initialize response
    try:
        if image_data:
            # Prepare image for Vision model
            img = Image.open(io.BytesIO(image_data))
            # Combine text and image in the prompt list
            prompt_parts = [prompt_text, img]
            response = model_to_use.generate_content(prompt_parts, stream=False)
        else:
            # Send text-only prompt
            response = model_to_use.generate_content(prompt_text, stream=False)

        # Extract text, handling potential blocks or lack of content
        if response and hasattr(response, 'text'):
             return response.text
        elif response and hasattr(response, 'parts'):
             # Sometimes response might be in parts, attempt to join text parts
             text_parts = [part.text for part in response.parts if hasattr(part, 'text')]
             return "\n".join(text_parts) if text_parts else "Assistant could not generate a text response for this."
        else:
             # Handle cases where response might be blocked or empty
             # Accessing prompt_feedback requires checking its existence first
             feedback_reason = "Unknown reason"
             if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                  feedback_reason = response.prompt_feedback.block_reason
                  return f"Response blocked due to: {feedback_reason}"
             # Log unexpected structure if needed without causing error
             # print(f"Unexpected response structure: {response}") # Use print for backend logging
             return "Assistant did not provide a response or it was empty."

    except genai.types.BlockedPromptException as e:
         st.error(f"Your prompt was blocked: {e}", icon="🚫")
         return f"Error: Prompt blocked. Reason: {e}"
    # Note: StopCandidateException might be less common or handled differently in newer APIs/models
    # except genai.types.StopCandidateException as e:
    #      st.error(f"Response generation stopped unexpectedly: {e}", icon="⚠️")
    #      # Sometimes the partial response might be available
    #      if response and hasattr(response, 'text'):
    #           return response.text + "\n\n(Response may be incomplete)"
    #      return f"Error: Response generation stopped. Reason: {e}"
    except Exception as e:
        # Catch other potential errors (API connection, invalid image format, etc.)
        st.error(f"An error occurred: {e}", icon="🔥")
        # Provide more specific feedback if possible
        error_str = str(e)
        if "API key not valid" in error_str:
             return "Error: Invalid Google API Key. Please check your .env file."
        # Check for permission errors explicitly
        elif "User location is not supported" in error_str:
             return "Error: Your location is not supported for this Google AI service."
        elif "permission" in error_str.lower() or "denied" in error_str.lower():
            return f"Error: Permission denied. Please check your API key permissions and ensure the Gemini API is enabled in your Google Cloud project. Details: {error_str[:200]}"
        return f"Error: Could not get response from AI. Details: {error_str[:500]}" # Limit error length

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="AI Teacher")
st.title("🧠 AI Teacher Assistant")
st.caption("Using Gemini 1.5 Flash. Ask questions, get explanations, or upload an image!") # Updated caption

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Stores {role: "user/model", parts: ["text", img_ref(optional)]}

# Initialize image state for current turn
if "uploaded_image_data" not in st.session_state:
    st.session_state.uploaded_image_data = None
if "uploaded_image_name" not in st.session_state:
    st.session_state.uploaded_image_name = None


# --- Sidebar for Image Upload ---
with st.sidebar:
    st.header("Upload Problem Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image data when uploaded
        image_bytes = uploaded_file.getvalue()
        # Store image data and name in session state FOR THIS TURN
        st.session_state.uploaded_image_data = image_bytes
        st.session_state.uploaded_image_name = uploaded_file.name
        st.image(image_bytes, caption=f"Ready: {uploaded_file.name}", use_column_width=True)
        st.info("Image ready. Now type your question or instruction in the chat.")
    # Add button to clear the uploaded image if user changes mind before sending message
    if st.session_state.uploaded_image_data is not None:
        if st.button("Clear Uploaded Image"):
            st.session_state.uploaded_image_data = None
            st.session_state.uploaded_image_name = None
            st.rerun() # Rerun to update sidebar display

# --- Main Chat Interface ---

# Display chat history
for message in st.session_state.chat_history:
    role = message["role"]
    # Adjust role for display if needed (Gemini uses 'model' for assistant)
    display_role = "assistant" if role == "model" else role
    with st.chat_message(display_role):
        # Display text part
        st.markdown(message["parts"][0])
        # Display image reference if it exists for this user message
        if len(message["parts"]) > 1 and message["parts"][1] is not None:
            st.markdown(f"*Image Uploaded: {message['parts'][1]}*")


# User input using st.chat_input
user_prompt = st.chat_input("Ask your question or describe the image...")

if user_prompt:
    # Get the image data from session state (if uploaded in this turn)
    current_image_data = st.session_state.uploaded_image_data
    current_image_name = st.session_state.uploaded_image_name

    # --- Display User Message ---
    with st.chat_message("user"):
        st.markdown(user_prompt)
        if current_image_name:
            st.markdown(f"*Image Uploaded: {current_image_name}*")

    # Add user message to history (store image name as reference)
    user_history_parts = [user_prompt]
    if current_image_name:
        user_history_parts.append(current_image_name) # Store name for history display
    st.session_state.chat_history.append({"role": "user", "parts": user_history_parts})

    # --- Get AI Response ---
    with st.chat_message("assistant"): # Gemini uses 'model' role internally
        message_placeholder = st.empty() # For potential streaming effect later
        message_placeholder.markdown("Thinking... 🧑‍🏫")

        # Construct the prompt for the AI - BE EXPLICIT about the persona
        ai_prompt = f"""You are an AI Teacher Assistant using the Gemini 1.5 Flash model. Your goal is to help users understand concepts and solve problems across various subjects, including math and reasoning.
        - If asked a question, provide a clear and accurate answer.
        - If given a math or reasoning problem, explain the steps clearly to help the user learn. Show your work whenever possible. Use markdown for formatting math (like $ax^2+bx+c=0$) and code blocks for steps if appropriate.
        - If an image is provided, analyze the image content (text, diagrams, equations) along with the text prompt. Describe what you see in the image if relevant to the explanation.
        - Be patient, encouraging, and break down complex topics.

        User's request: {user_prompt}
        """
        # Include image data if available
        response_text = get_gemini_response(ai_prompt, current_image_data)

        # Display the AI response
        message_placeholder.markdown(response_text)

    # Add AI response to history
    st.session_state.chat_history.append({"role": "model", "parts": [response_text]})

    # --- IMPORTANT: Clear the uploaded image state after processing this turn ---
    # Ensures the image isn't accidentally reused for the next text-only message
    st.session_state.uploaded_image_data = None
    st.session_state.uploaded_image_name = None

    # Optional: Rerun slightly to ensure the sidebar clears visually if needed
    # st.rerun()