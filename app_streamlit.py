# app_streamlit.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM
import torch

MODEL_DIR = "sentiment_model"

@st.cache_resource
def load_sentiment_model():
    try:
        # Try to load local model first
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)  # type: ignore
        return pipe
    except Exception as e:
        st.warning(f"Local model not found ({e}). Using pre-trained model instead.")
        # Fallback to pre-trained model
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device)  # type: ignore
        return pipe

@st.cache_resource
def load_conversation_model():
    try:
        device = 0 if torch.cuda.is_available() else -1
        
        # Choose model based on your needs:
        # Option 1: DialoGPT-small (Fast, less accurate) ← CURRENT (Fast!)
        # Option 2: DialoGPT-medium (Balanced)
        # Option 3: DialoGPT-large (Slower, more accurate)
        # Option 4: GPT-2 (Better at general conversation)
        
        model_name = "microsoft/DialoGPT-small"  # Changed to small for faster loading!
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Failed to load conversation model: {e}")
        return None, None, -1

st.set_page_config(page_title="AI Chatbot + Sentiment", layout="wide")
st.title("🤖 AI Chatbot + Sentiment Checker")
st.markdown("Type a message — the AI bot generates contextual responses using **DialoGPT-Small** and shows sentiment analysis from the fine-tuned **DistilBERT** model.")

# Model info
with st.expander("ℹ️ About the AI Models"):
    st.write("""
    **Conversation Model**: DialoGPT-Small (117M parameters)
    - Trained on 147 million Reddit conversations
    - Fast loading and response times
    - Good balance of quality and speed
    
    **Sentiment Model**: Your Fine-tuned DistilBERT (66M parameters)
    - Analyzes positive/negative sentiment
    - Custom-trained on your data
    """)

# Load both models
sentiment_nlp = load_sentiment_model()
conv_tokenizer, conv_model, conv_device = load_conversation_model()

# Add some styling
st.markdown("""
<style>
.sentiment-positive {
    color: green;
    font-weight: bold;
}
.sentiment-negative {
    color: red;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

user_input = st.text_input("💬 Your message:", "", placeholder="Type your message here...")

if user_input:
    # Sentiment Analysis
    out = sentiment_nlp(user_input[:1000])[0]
    label = out["label"]
    score = out["score"]

    # Convert label to human-readable format
    if isinstance(label, str):
        if "POS" in label.upper() or "LABEL_1" in label.upper():
            sentiment = "Positive"
            sentiment_class = "sentiment-positive"
        elif "NEG" in label.upper() or "LABEL_0" in label.upper():
            sentiment = "Negative"
            sentiment_class = "sentiment-negative"
        else:
            sentiment = label
            sentiment_class = ""
    else:
        sentiment = "Positive" if label == 1 else "Negative"
        sentiment_class = "sentiment-positive" if label == 1 else "sentiment-negative"

    # AI-Generated Response System using DialoGPT
    if conv_tokenizer and conv_model:
        try:
            # Initialize conversation history if not exists
            if 'chat_history_ids' not in st.session_state:
                st.session_state.chat_history_ids = None
            
            # Add context to improve response quality
            # For first message, add a system context
            if st.session_state.chat_history_ids is None:
                context_prompt = "You are a friendly and helpful chatbot assistant. "
                enhanced_input = context_prompt + user_input
            else:
                enhanced_input = user_input
            
            # Encode the user input
            new_input_ids = conv_tokenizer.encode(enhanced_input + conv_tokenizer.eos_token, return_tensors="pt")
            
            # Set up device properly
            device = torch.device('cuda:0' if conv_device >= 0 else 'cpu')
            new_input_ids = new_input_ids.to(device)
            conv_model.to(device)  # type: ignore
            
            # Append the new user input tokens to the chat history
            if st.session_state.chat_history_ids is not None:
                bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
            else:
                bot_input_ids = new_input_ids
            
            # Generate response
            with torch.no_grad():
                st.session_state.chat_history_ids = conv_model.generate(
                    bot_input_ids,
                    max_length=1000,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    top_k=100,
                    top_p=0.92,
                    temperature=0.8,
                    pad_token_id=conv_tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )

            # Decode and extract just the bot's response
            full_response = conv_tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            reply = full_response.strip()
            
            # Clean up response - remove any leftover context
            if "You are a friendly" in reply:
                reply = reply.split("You are a friendly")[-1].strip()
            
            # If response is too short, empty, or seems off-topic, provide a contextual fallback
            if not reply or len(reply) < 3:
                reply = f"I understand you're feeling {sentiment.lower()}. Tell me more!"
            elif len(reply) < 10:
                # Very short responses - enhance them
                contextual_responses = {
                    "Positive": [
                        "That's great! I'd love to hear more about that.",
                        "That sounds interesting! Tell me more.",
                        "Wonderful! What else would you like to share?",
                        "That's nice! Can you elaborate?"
                    ],
                    "Negative": [
                        "I'm sorry to hear that. Would you like to talk about it?",
                        "That sounds difficult. I'm here to listen.",
                        "I understand. What's been going on?",
                        "That must be tough. Want to share more?"
                    ]
                }
                import random
                reply = random.choice(contextual_responses.get(sentiment, contextual_responses["Positive"]))

        except Exception as e:
            st.warning(f"AI response generation failed: {e}")
            reply = f"I sense you're feeling {sentiment.lower()}. How can I help you today? 🤖"
    else:
        # Fallback if AI model fails to load
        reply = f"I sense you're feeling {sentiment.lower()}. How can I help you today? 🤖"
                                    
    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Bot Reply (AI-Generated)")
        st.write(reply)
        st.caption("💡 Response generated by DialoGPT AI model")

    with col2:
        st.subheader("📊 Sentiment Analysis")
        st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{sentiment}</span>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {score:.3f}")

        # Visual confidence meter
        if score > 0.8:
            st.success("Very confident prediction!")
        elif score > 0.6:
            st.info("Moderately confident prediction")
        else:
            st.warning("Low confidence prediction")

    # Raw output (collapsible)
    with st.expander("🔍 Raw Model Output"):
        st.write(out)

    # Add a button to reset conversation
    if st.button("🔄 Reset Conversation"):
        st.session_state.chat_history_ids = None
        st.rerun()
