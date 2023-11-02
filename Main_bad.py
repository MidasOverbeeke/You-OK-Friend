# THIS IS THE BAD APP, ONLY DO DEMONSTRATE HOW NOT TO DO IT
# THE EXPLAINABILITY, BIAS AND PRIVACY IS INTENTIONALLY BAD
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer

st.set_page_config(page_title="You OK Friend? (Bad version)")

@st.cache_resource
def load_model():
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # load model
    model = AutoModelForSequenceClassification.from_pretrained("./modelv0.1")
    # initialize pipeline
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
classifier = load_model()

def predict(text):
    pred = classifier(text)[0]
    if pred["label"] == 'LABEL_0':
        pred["label"] = "non-suicidal"
    if pred["label"] == 'LABEL_1':
        pred["label"] = "suicidal"
    return pred

# button functions
def homeButton():
    del st.session_state["page"]
    button03() # also reset input
def button01():
    st.session_state.page = "try_out"
def button02():
    st.session_state["prediction"] = predict(st.session_state["message"])
def button03():
    if "prediction" in st.session_state:
        del st.session_state["prediction"]
    if "message" in st.session_state:
        del st.session_state["message"]

# main page
if "page" not in st.session_state:
    st.title("You OK Friend?")
    st.divider()
    st.header("What does this app do?")
    st.write("You’re worried about a friend? You feel like they sent you a message that may indicate they are suicidal? Are you doubting to seek professional help? Don’t sit and wait, use the app to find out what to do!")
    st.header("Why AI?")
    st.write("AI is the future. AI is basically a human being that can work very fast and (basically) for free. We chose to setup an AI to analyze your text messages because a human being - like a therapist - often cost a lot of money. Is it really worth it to spend money on a single message you got from a friend? Just because you are a little worried? AI can do this so much better for no cost! AI is the future and can handle complex tasks, like this one.")
    st.divider()
    st.header("Begin the test")
    st.write("If you like what we do and want to try it yourself, go to the Try Out page!")
    st.button("Take me!",on_click=button01)
    # st.success("This app is completely anonymous, and we do not collect any personal information from our users.")

# try out page
if "page" in st.session_state:
    st.button("< Home",on_click=homeButton)
    if st.session_state["page"] == "try_out":
        if 'prediction' not in st.session_state:
            st.header("Input text to analyze.")
            st.divider()
            st.write("Enter your text message...")
            st.session_state["message"] = st.text_area("text_area", height=400, label_visibility="collapsed")
            if st.session_state["message"] != "":
                st.button("Analyze...",on_click=button02)
            else:
                st.write("Please press Ctrl+Enter to apply the text.")
                
    # answer page
        if 'prediction' in st.session_state:
            st.header("Prediction result")
            st.divider()
            sc = round(st.session_state.prediction['score']*100,2)
            if st.session_state.prediction['label'] == "suicidal":
                st.markdown(f"Our AI predicted: suicidal ({sc}\% confident)")
                st.progress(sc/100, text="Suicide indicator")
                st.write("This is seriously bad news. You should seek professional help immediately before your friend hurts himself. Call 911 if you feel like this is urgent to be sure he is safe.")
            if st.session_state.prediction['label'] == "non-suicidal":
                st.markdown(f"Our AI predicted: non-suicidal ({sc}\% confident)")
                st.progress(1-sc/100, text="Suicide indicator")
                st.write("Congratulations! Your friend is not suicidal. This is a good time to relax yourself and stop worrying so much about others. Give yourself some space and start enjoying your life more!")
                st.balloons()
            
            # st.divider()
            st.button("Retry...",on_click=button03)
            