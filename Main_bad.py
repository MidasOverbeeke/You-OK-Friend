# THIS IS THE BAD APP, ONLY DO DEMONSTRATE HOW NOT TO DO IT
# THE EXPLAINABILITY, BIAS AND PRIVACY IS INTENTIONALLY BAD
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer

st.set_page_config(page_title="You OK Friend?")

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
def button04():
    st.session_state.page = "info_page"

# main page
if "page" not in st.session_state:
    st.title("You OK Friend?")
    st.divider()
    st.markdown("**The bad app**")
    # st.header("Why does this app exist?")
    # st.write("Sometimes it can be hard to seek help when dealing with depression. Talking to friends may be a first step a person with depression may take, but it may be hard to judge the seriousness of the situation for the person they talk to. This app tries to help decide if your friend needs professional help.")
    # st.header("What does this app do?")
    # st.write("This app lets you analyze a text you received from a friend that may be suffering from depression and tries to decide if the text is similar to texts that suicidal people wrote. If so, it might be an indication to seek professional help for your friend.")
    # st.header("How does this app work?")
    # st.write("This app uses an AI to analyze the text you want. When it contains series of words that a suicidal person has said in our training data, the certainty of the person being suicidal will go up. If you would like to know more about our AI, click the button below.")
    # st.button("More info...",on_click=button04)
    # st.divider()
    st.header("Begin the test")
    st.write("If you like what we do and want to try it yourself, go to the Try Out page!")
    st.button("Take me!",on_click=button01)
    # st.success("This app is completely anonymous, and we do not collect any personal information from our users.")

# try out page
if "page" in st.session_state:
    st.button("< Home",on_click=homeButton)
    if st.session_state["page"] == "try_out":
        if 'prediction' not in st.session_state:
            st.header("Text input")
            st.divider()
            # st.write("Input the text you got from your friend to analyze. Please only use English.")
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
            if st.session_state.prediction['label'] == "non-suicidal":
                st.markdown(f"Our AI predicted: non-suicidal ({sc}\% confident)")
            # st.divider()
            st.button("Retry...",on_click=button03)
            