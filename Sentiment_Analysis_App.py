import pickle
import streamlit as st
import nltk
import re
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# Load the model and vectorizer from the pickle files
# model_path = r'C:\Users\ASUS\Desktop\FYP\pickle files\model.pkl'
# vectorizer_path = r'C:\Users\ASUS\Desktop\FYP\pickle files\vectorizer.pkl'
# stopwords_path = r'C:\Users\ASUS\Desktop\FYP\pickle files\stopwords.pkl'

# Load stop words from pickle file
my_sw = pickle.load(open('stopwords.pkl', 'rb'))
LSVC_vader_smote_classifier = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))


#--------------------------------------------------------------------------------------
# Preprocess text 
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Replace contractions
    text = contractions.fix(text)

    # Remove URLs
    text = re.sub(r'https?://\S+', ' ', text)
    
    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]|[\d]+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    wordnet_pos = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    pos_tagged_tokens = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, wordnet_pos.get(pos[0], wordnet.NOUN)) for token, pos in pos_tagged_tokens  if len(pos) >= 1]

    # Remove stop words
    clean_tokens = [word for word in lemmatized_tokens if word not in my_sw]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(clean_tokens)
    
    return preprocessed_text

# Sentiment analysis function
def predict_sentiment(preprocessed_text):
    text = preprocess_text(preprocessed_text)
    text = tfidf.transform([text])
    pred = LSVC_vader_smote_classifier.predict(text)
    if pred == 'Positive':
        return 'Positive'
    elif pred == 'Negative':
        return 'Negative'
    else:
        return 'Neutral'


#------------------------------------------------------------------------------------------------------
#Streamlit App Customization

#Set web page name, layout
st.set_page_config(page_title='E-Book Sentiment Analysis', page_icon=':books:', layout="wide")

#Set font size
st.markdown("""
<style>
.medium-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

#Create "About" page
def create_page1():
    st.title('_Welcome to the :green[E-Book Sentiment Analysis Web App!]_ :wink:')
    st.subheader(':orange[_What is Sentiment Analysis(SA)?_]')
    st.markdown('<p class="medium-font">Sentiment analysis is a technique used to determine the sentiment or tone of a piece of text. It involves using natural language processing (NLP) techniques to identify positive,negative, or neutral sentiments in written content. Sentiment analysis is important because it allows businesses and organizations to better understand the opinions and attitudes of their customers and stakeholders. By analyzing sentiment in social media posts, product reviews, customer feedback, and other sources, businesses can gain insights into customer preferences and satisfaction levels, identify areas for improvement, and make informed decisions about product development, marketing campaigns, and customer service.</p>',unsafe_allow_html=True)
    
    st.subheader(':orange[_What about the e-book domain?_]')
    st.markdown('<p class="medium-font">E-books have become increasingly popular in recent years, and sentiment analysis can be a powerful tool for businesses in this industry. By analyzing sentiment in e-book reviews, publishers can gain valuable insights into reader preferences and opinions. For example, they can use sentiment analysis to identify which genres or authors are most popular, which plot elements or writing styles resonate with readers, and what aspects of the reading experience are most important to customers.Many e-book businesses are already using sentiment analysis to gain insights into customer preferences and improve their products and services. For example, Amazon, one of the largest e-book retailers in the world, uses sentiment analysis to analyze customer reviews and identify the most helpful and informative reviews. With its increasing popularity, Amazon collects a significant amount of data daily.</p>',unsafe_allow_html=True)
    
    st.subheader(':orange[_Importance of SA_]')
    st.markdown('<p class="medium-font">Customers express their opinions by reviewing and rating the products they purchase. The sentiment analysis of such data can aid in the establishment of a win-win partnership for both businesses and users. Buyers place a greater emphasis on product reviews before making a purchasing decision. This leads to merchants learning about the quality of their product and enhance the quality of their businesses by reviewing the feedback left on them. Furthermore, humans are fairly adept at determining sentiments, whether a review is positive or negative just by looking at it. However, it is difficult to manually read through the thousands of reviews on a regular basis and comprehend an overview on which books customers like or dislike. Besides, it is time-consuming and does not provide a concise idea of product comparison in general. This is where sentiment analysis comes in! The significance of sentiment analysis has driven companies to determine the customer"s opinion of the product after examining the reviews, hence advances to develop recommendation systems or enhance their marketing efforts. </p>',unsafe_allow_html=True)

#Create "Sentiment Prediction system" page
def create_page2():
    st.title('E-Book Sentiment Analysis :books:')
    st.subheader(':violet[_Enter your review below and Click on the button to the sentiment predicted_ :]')
    input_text = st.text_area('')
    if st.button('Predict :rocket:'):
        if not input_text:
            st.warning("Please enter a review before predicting sentiment!")
        else:
            sentiment = predict_sentiment(input_text)
            if sentiment == 'Positive':
                st.write('Sentiment:', sentiment,' :smile:')
            elif sentiment == 'Negative':
                st.write('Sentiment:', sentiment,' :slightly_frowning_face:')
            else:
                st.write('Sentiment:', sentiment,' :neutral_face:')

#Create "Dashboard" page
def create_page3():
    st.title('Get insights on the Kindle E-book Reviews!')
    st.subheader('_Dashboard for_ :blue[Vader Data]')
    st.markdown('<iframe title="Vader Power BI" width="1024" height="804" src="https://app.powerbi.com/view?r=eyJrIjoiMjY2MTBkMWItMThmMy00M2YwLTg0ZjktY2I4YTc1ZWQyMWZmIiwidCI6IjBmZWQwM2EzLTQwMmQtNDYzMy1hOGNkLThiMzA4ODIyMjUzZSIsImMiOjEwfQ%3D%3D" frameborder="0" allowFullScreen="true"></iframe>',unsafe_allow_html=True)
    st.text("")
    st.text("")
    st.text("")
    st.subheader('_Dashboard for_ :orange[Rating Data]')
    st.markdown('<iframe title="Rating Power BI" width="1024" height="804" src="https://app.powerbi.com/view?r=eyJrIjoiYjRlNjA0NjQtNDZlMy00MDNhLTk3OGUtMGQyNmI2NjE1Y2M5IiwidCI6IjBmZWQwM2EzLTQwMmQtNDYzMy1hOGNkLThiMzA4ODIyMjUzZSIsImMiOjEwfQ%3D%3D" frameborder="0" allowFullScreen="true"></iframe>',unsafe_allow_html=True)


# Define the app
def app():
    
    # Create the navigation menu
    st.sidebar.title('Navigation Pane')
    choice = st.sidebar.radio('Click on a page', ('About', 'Sentiment Analysis', 'Visualization & Insights'))
    
    # Show the appropriate page based on the user's choice
    if choice == 'About':
        create_page1()
    elif choice == 'Sentiment Analysis':
        create_page2()
    elif choice == 'Visualization & Insights':
        create_page3()

# Run the app
if __name__ == '__main__':
    app()
