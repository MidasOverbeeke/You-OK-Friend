
import streamlit as st
import pandas as pd
from sklearn.utils import shuffle
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer

st.set_page_config(page_title="You OK Friend?")

# load sample data from csv to show on info page
@st.cache_data
def load_data():
    return pd.read_csv("dataset_sample/Suicide_Detection_0-999.csv")
dataframe = load_data()

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
    st.header("Why does this app exist?")
    st.write("Sometimes it can be hard to seek help when dealing with depression. Talking to friends may be a first step a person with depression may take, but it may be hard to judge the seriousness of the situation for the person they talk to. This app tries to help decide if your friend needs professional help.")
    st.header("What does this app do?")
    st.write("This app lets you analyze a text you received from a friend that may be suffering from depression and tries to decide if the text is similar to texts that suicidal people wrote. If so, it might be an indication to seek professional help for your friend.")
    st.header("How does this app work?")
    st.write("This app uses an AI to analyze the text you want. When it contains series of words that a suicidal person has said in our training data, the certainty of the person being suicidal will go up. If you would like to know more about our AI, click the button below.")
    st.button("More info...",on_click=button04)
    st.divider()
    st.header("Begin the test")
    st.write("If you like what we do and want to try it yourself, go to the Try Out page!")
    st.button("Take me!",on_click=button01)
    st.success("This app is completely anonymous, and we do not collect any personal information from our users.")

# try out page
if "page" in st.session_state:
    st.button("< Home",on_click=homeButton)
    if st.session_state["page"] == "try_out":
        if 'prediction' not in st.session_state:
            st.header("Text input")
            st.divider()
            st.write("Input the text you got from your friend to analyze. Please only use English.")
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
            if st.session_state.prediction['score'] < 0.65:
                st.write(f"Our AI is not sure. It predicted a probability of {sc}%, which is basically like flipping a coin. As the developers of this AI, we decided not to flip a coin on lives. We would recommend letting others human beings - like close friends or professionals - look at the text you are worried about. They can often do better than an AI because an AI cannot process human feelings or pick up on subtle signals.")
                st.header("What now?")
                st.markdown("- Let your friend know you are worried about them. Often it helps a lot if you just talk about the situation. You can start the conversation by telling them you are worried about what they said in the text, after that it’s important to listen to them. If you’re not sure how to confront them, a professional can inform you how to do it well.\n- Speak to close friends who you and your friend trust. Being open about your feelings may help everyone involved and strengthen the bond you have with each other.")
            elif st.session_state.prediction['label'] == "suicidal":
                if st.session_state.prediction['score'] > 0.90:
                    st.write("Our AI is very sure your friend is suicidal. This means that the text you have given has a lot of elements referring to being suicidal. Terms like “suicide”, “kill myself” and “wanting to die” will convince the AI that the text is written by a depressed person. Please note that the AI can’t understand sarcasm, humor or human feelings. Always listen to your gut feeling and ask a professional if you’re worried about the safety of your friend.")
                    st.header("What can you do?")
                    st.markdown("- Don’t panic, but be serious about this situation. \n- Contact a professional. You can reach out to https://988lifeline.org/, they can help you further.\n- Let your friend know you are there for them and are always happy to help. You can begin the conversation by telling them you are worried about what they said in the text, after that it’s important to listen to them. Tell them how brave they are for reaching out. If you’re not sure how to confront them, a professional can inform you how to do it well.")
                elif st.session_state.prediction['score'] > 0.75:
                    st.write("Our AI is worried your friend is suicidal. This means that the text you have given has quite some elements referring to being suicidal. Terms like “suicide”, “kill myself” and “wanting to die” will convince the AI that the text is written by a depressed person. Please note that the AI can’t understand sarcasm, humor or human feelings. Always listen to your gut feeling and ask a professional if you’re worried about the safety of your friend.")
                    st.header("What can you do?")
                    st.markdown("- Let your friend know you are there for them and are always happy to help. You can begin the conversation by telling them you are worried about what they said in the text, after that it’s important to listen to them. If you’re not sure how to confront them, a professional can inform you how to do it well.\n- Contact a professional if more messages like the one you submitted. You can reach out to https://988lifeline.org/, they can help you further.")
                else:
                    st.write("Our AI is a little worried your friend is suicidal, but is not sure. This means that the text you have given has a few elements referring to being suicidal. Terms like “suicide”, “kill myself” and “wanting to die” will convince the AI that the text is written by a depressed person. Please note that the AI can’t understand sarcasm, humor or human feelings. Always listen to your gut feeling and ask a professional if you’re worried about the safety of your friend.")
                    st.header("What can you do?")
                    st.markdown("- If you feel like your friend might not be doing well, let them know you are there for them and are always happy to help. You can begin the conversation by telling them you are worried about what they said in the text, after that it’s important to listen to them. If you’re not sure how to confront them, a professional can inform you how to do it well.\n- Feel free to contact a professional if you still doubt the advice of the AI. You can reach out to https://988lifeline.org/, they can help you further.")
            elif st.session_state.prediction['label'] == "non-suicidal":
                if st.session_state.prediction['score'] > 0.90:
                    st.write("Our AI is very sure the text does not contain any references to suicide. This only means the text does not contain terms like “suicide”, “kill myself” and “wanting to die”, which will convince the AI that the text is written by a depressed person. Please keep in mind that this does not imply that your friend is OK.")
                    st.header("What can you do with this information?")
                    st.markdown("- You can assure yourself that this text does not have much in common with texts mentioning suicidal thoughts.\n- Send your friend a message if you doubt their well-being. Often by talking with them you can understand them better.\n- Get a second opinion. Our AI is certainly not perfect, so if you doubt the advice listed above, contact a professional. We recommend going to https://988lifeline.org/, a website dedicated to helping people with depression or suicidal thoughts.")
                elif st.session_state.prediction['score'] > 0.75:
                    st.write("Our AI is quite sure the text does not contain many references to suicide. This only means the text does not contain terms like “suicide”, “kill myself” and “wanting to die”, which will convince the AI that the text is written by a depressed person. Please keep in mind that this does not imply that your friend is OK.")
                    st.header("What can you do with this information?")
                    st.markdown("- You can assure yourself that this text is not like the typical texts mentioning suicidal thoughts that can be found in our database.\n- Send your friend a message if you doubt their well-being. Often by talking with them you can understand them better.\n- Get a second opinion. Our AI is certainly not perfect, so if you doubt the advice listed above, contact a professional. We recommend going to https://988lifeline.org/, a website dedicated to helping people with depression or suicidal thoughts.")
                else:
                    st.write("Our AI is convinced the text does not contain any references to suicide, although is it not very sure about that. This only means the text does not contain terms like “suicide”, “kill myself” and “wanting to die”, which will convince the AI that the text is written by a depressed person. Please keep in mind that this does not imply that your friend is OK.")
                    st.header("What can you do with this information?")
                    st.markdown("- Get a second opinion. Our AI is certainly not perfect, so if you doubt the advice listed above, contact a professional. We recommend going to https://988lifeline.org/, a website dedicated to helping people with depression or suicidal thoughts.\n- You can assure yourself that this text does not have much in common with texts mentioning suicidal thoughts.\n- Send your friend a message if you doubt their well-being. Often by talking with them you can understand them better.")
            st.write("For more info about what you can do, go to https://988lifeline.org/")
            st.divider()
            st.button("Retry...",on_click=button03)
            st.info("Please keep in mind this answer is generated by an AI. This AI was trained on multiple stories of suicidal and non-suicidal individuals. The AI may generate false answers. Please go to a mental health professional for a second opinion.")

    # more info page
    elif st.session_state["page"] == "info_page":
        df = shuffle(dataframe) # show different messages each time the page reloads
        st.header("Who are we?")
        st.markdown("We are [Matthias Hulscher](https://www.linkedin.com/in/matthias-hulscher-a04369164/) and [Midas Overbeeke](https://www.linkedin.com/in/midas-overbeeke-8b4122226/). We study Mechatronics on Avans Breda, The Netherlands and are currenly following a minor AI. We made this app for the course 'Human Centered AI Design'. We chose this subject because a lot of young people suffer from depression these days. We believe that spotting suicide in a friend can be hard sometimes, especially if you don't talk about it often. We made this app to make identifying depression accessible and easy, hopefully saving lives.")
        st.divider()

        st.header("How we made the AI")
        st.markdown("We utilized the Hugging Face Transformer model for our AI. The Transformer model is a type of artificial intelligence model that is designed to handle sequential data, making it particularly effective for natural language processing tasks. Unlike traditional models that process data in order, the Transformer model uses a mechanism called attention to weigh the importance of different words or features in the input data. This allows it to capture long-range dependencies in the data and generate more accurate and coherent outputs. More info can be found on my [GitHub](https://github.com/MidasOverbeeke/You-OK-Friend).")
        st.divider()

        st.header("The data we used")
        st.markdown("The table below shows 5 examples of the raw data from the Kaggle dataset. To view the full dataset, go over to [Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)\nIt contains 232,074 messages of suicial and non-suicidal messages, all labeled by catagory.")
        st.dataframe(df[["text", "class"]][:5])
        st.markdown("We took this dataset and cleaned it as well as we could. We removed all the emoji's, put spaces between words where needed and formatted every message to (somewhat) the same layout. \nTo prevent a [bias](https://en.wikipedia.org/wiki/Algorithmic_bias) in our AI, we made sure we made sure to use half of the data are suicidal message, half of it non-suicidal. The graph below shows the distribution of the catagories in the datasamples.")
        st.bar_chart({"suicide":116037,"non-suicide":116037})
        st.markdown("After that, we split shuffled all the data and split it up into train- and test datasets with the train_test_split method from scikit-learn. We used 80\% for train data and 20\% for test data. We do this to be able to test how accurate the AI is, without overfitting the AI on the train data.")
        st.divider()

        st.header("How we tested the AI")
        st.markdown("We used the classification_report from scikit-learn to generate a report of our AI based on the train and test datasets. We found that the AI could get an accuracy of 1 on the trainings data and an accuracy of 0.94 on the test data, which is quite good. For more information on how we tested the AI and the results, go to our [Jupyter Notebook](https://github.com/MidasOverbeeke/You-OK-Friend/blob/master/jupyter/notebook.ipynb)")
        st.markdown("After that we generated three different, new texts. One which was very suicidal, one that was made to give the AI mixed signals and one that was very happy. Those texts and the preducted outcome were:\n- Hey my name is Midas and lately i've been feeling down. I don't seem to be interested in my hobbies like I used to. I'm worried I will end up alone and collapse under the loneliness. What if I'll harm myself in the long run? Should I seek help or am I just overreacting? (AI: suicidal, 98.96\%)\n- If you're happy and you know it clap your hands If you're happy and you know it clap your hands If you're happy and you know it and you really want to show it If you're happy and you know it clap your hands. (AI: non-suicidal, 99.62\%)\n- Hey how are you doing, i'm not gonna do anything to myself but i need your help. I'm doing great btw. My kid just beat cancer, probably. (AI: suicidal, 58.88\%)")
        st.markdown("Based on these results, we concluded that the AI was working well.")
        st.divider()

        st.header("Contact us")
        st.markdown("If you have any questions for the developers, you can send us an email:\n- [Matthias Hulscher](mailto:m.hulscher@student.avans.nl)\n- [Midas Overbeeke](mailto:m.overbeeke@student.avans.nl)")
        
        # automatically scroll up with javascript
        js = '''
        <script>
            var body = window.parent.document.querySelector(".main");
            console.log(body);
            body.scrollTop = 0;
        </script>
        '''
        st.components.v1.html(js)