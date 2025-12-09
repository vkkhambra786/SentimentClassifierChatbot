# app_streamlit_gpt2.py - Alternative version with GPT-2
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
import torch

MODEL_DIR = "sentiment_model"

@st.cache_resource
def load_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)  # type: ignore
        return pipe
    except Exception as e:
        st.warning(f"Local model not found ({e}). Using pre-trained model instead.")
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device)  # type: ignore
        return pipe

@st.cache_resource
def load_conversation_model():
    try:
        device = 0 if torch.cuda.is_available() else -1
        
        # Using GPT-2 Medium - Better for general conversation
        model_name = "gpt2-medium"  # 355M parameters, better than DialoGPT for many tasks
        
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Failed to load conversation model: {e}")
        return None, None, -1

st.set_page_config(page_title="AI Chatbot + Sentiment (GPT-2)", layout="wide")
st.title("🤖 AI Chatbot + Sentiment Checker (GPT-2)")
st.markdown("Type a message — the AI bot generates contextual responses using **GPT-2** and shows sentiment analysis from the fine-tuned **DistilBERT** model.")

# Model info
with st.expander("ℹ️ About the AI Models"):
    st.write("""
    **Conversation Model**: GPT-2 Medium (355M parameters)
    - Trained on diverse internet text
    - Better at general knowledge and varied topics
    - More coherent long-form responses
    
    **Sentiment Model**: Your Fine-tuned DistilBERT (66M parameters)
    - Analyzes positive/negative sentiment
    - Custom-trained on your data
    
    **Note**: GPT-2 is better for general conversation but not specifically trained on dialogue like DialoGPT.
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

    # AI-Generated Response System using GPT-2
    if conv_tokenizer and conv_model:
        try:
            # Create a conversational prompt
            prompt = f"Human: {user_input}\nAssistant:"
            
            # Encode the prompt
            input_ids = conv_tokenizer.encode(prompt, return_tensors="pt")
            
            # Set up device properly
            device = torch.device('cuda:0' if conv_device >= 0 else 'cpu')
            input_ids = input_ids.to(device)
            conv_model.to(device)  # type: ignore
            
            # Generate response
            with torch.no_grad():
                output = conv_model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 50,  # Generate up to 50 new tokens
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    top_k=100,
                    top_p=0.92,
                    temperature=0.8,
                    pad_token_id=conv_tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    early_stopping=True
                )

            # Decode and extract just the assistant's response
            full_response = conv_tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the assistant's reply
            if "Assistant:" in full_response:
                reply = full_response.split("Assistant:")[-1].strip()
            else:
                reply = full_response.replace(prompt, "").strip()
            
            # Clean up - take first sentence or line
            if "\n" in reply:
                reply = reply.split("\n")[0].strip()
            
            # If response is too short or empty, provide a fallback
            if not reply or len(reply) < 5:
                contextual_responses = {
                    "Positive": [
                        "That's wonderful! I'd love to hear more about that.",
                        "That sounds interesting! Tell me more.",
                        "Great! What else would you like to share?"
                    ],
                    "Negative": [
                        "I'm sorry to hear that. Would you like to talk about it?",
                        "That sounds difficult. I'm here to listen.",
                        "I understand. What's been going on?"
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
        st.caption("💡 Response generated by GPT-2 AI model")

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

st.markdown("---")
st.caption("💡 Tip: GPT-2 is better for general topics. DialoGPT is better for chat. Try both!")
