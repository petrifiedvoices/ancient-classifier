import os
import pickle

import pandas as pd
import streamlit as st
import altair as alt

from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

from plots import plot_bar_confidence
from util import extract_decision_function

# ===
# Externals
# ===

with open('mdl/conservative_oversampled/210603_ridge.pcl', 'rb') as fin:
    mdl_clf_conservative = pickle.load(fin)

with open('mdl/conservative_oversampled/210603_preprocessing.pcl', 'rb') as fin:
    mdl_prep_conservative = pickle.load(fin)

with open('mdl/interpretive_oversampled/210602_ridge.pcl', 'rb') as fin:
    mdl_clf_interpretive = pickle.load(fin)

with open('mdl/interpretive_oversampled/210602_preprocessing.pcl', 'rb') as fin:
    mdl_prep_interpretive = pickle.load(fin)

# ===
# Streamlit config
# ===
st.set_page_config(
    page_icon='🏛',
    layout='centered',
    page_title='Epigraphic Classifier for Latin inscriptions'
)

# ===
# Sidebar
# ===
with st.sidebar:
    # track which model to use 
    model_choice = st.selectbox(
        'Which model do you want to use?',
        ('Text as-is on inscriptions (conservative text)', 'Reconstructed text (interpretive text)')
    )

    if model_choice == 'Text as-is on inscriptions (conservative text)':
        default_input = 'Accae l Myrine Accae l Sympherusae M Ant M l Ero poma'

    elif model_choice == 'Reconstructed text (interpretive text)':
        default_input = 'Accae mulieris libertae Myrine Accae mulieris libertae Sympherusae Marco Antonio Marci liberto Ero pomario'


# ===
# Classify input
# ===
st.title('Epigraphic Classifier for Latin inscriptions')

st.header('What does it do?')
st.write('*Scenario 1:* Imagine you are an archaeologist, excavating an ancient settlement and you have found an inscription with Latin text. You are not an expert on inscriptions, but knowing what kind of text you are dealing with, while still in the field, would help your immediate understanding of the archaeological situation and would help you guide the excavation in the right direction.')
st.write('*Scenario 2:* Imagine you are a museum archivist and you have found an unlabelled inscription in the depository. You would like to be able to record in a museum catalogue its type, so the future experts can find it more easily.')

st.info('Enter the text of the inscription to the Classifier and it will tell you what kind of inscription you are dealing with!')  
st.write('The Classifier is trained on the text of 50,000 inscriptions to come up with the most probable epigraphic classification and its alternatives. However, you still should consult a human-epigrapher to be 100% sure!')

st.header('How do I classify my text?')
st.write('1. Choose the format of your text in the dropdown menu in the left-side panel:')
st.write('      a) does your text has the original format as-is on inscriptions, including unexpanded abbreviations and no modern additions or reconstructions >>> choose `Text as-is on inscriptions`;')
st.write('      b) does your text contain expanded abbreviations and modern reconstructions that are not present on the inscribed object >>> choose `Reconstructed text` option.')
st.write('2. Insert the text into the input window below and click the `Classify me!` button.')
st.write('3. The classifical results will appear below. The closer any typological variant is to 1, the higher is the probability of your text belonging to that category. If ambiguous, too short, or too fragmentary, the text can easily belong to multiple categories, without one clear favourite.')
st.write('Don\' forget - the `Epigraphic Classifier` is still experimentary and so may be the results!')
    
user_input = st.text_area("Insert text", default_input)

if not isinstance(user_input, str):
    raise TypeError('Input is not a string')

if len(user_input) > 1000:
    raise MemoryError('Input text too long. Max 1000 characters allowed')

if st.button('Classify me!'):

    if model_choice == 'Text as-is on inscriptions (conservative text)':
        transformer = mdl_prep_conservative
        model = mdl_clf_conservative
    elif model_choice == 'Reconstructed text (interpretive text)':
        transformer = mdl_prep_interpretive
        model = mdl_clf_interpretive

    y_pred, confidence_df = extract_decision_function(
        transformer,
        model,
        user_input
    )

    st.write('\n\n')
    st.markdown("""---""")

    st.subheader('Text was classified as')
    st.code(f'{y_pred}')
    st.write('\n\n')

    st.subheader('Confidence values')
    st.write('\n')
    st.write(
        plot_bar_confidence(confidence_df)
    )

st.header('About Classifier')
st.write('Authors: Petra Hermankova, postdoc at Aarhus University, [ORCID:0000-0002-6349-0540](https://orcid.org/0000-0002-6349-0540), [Github](https://github.com/petrifiedvoices) & Jan Kostkan, CHCAA, Aarhus University, [GitHub](https://github.com/supplyandcommand)')
st.write('The model was trained on Latin inscriptions from the [Epigraphic Database Heidelberg](https://edh-www.adw.uni-heidelberg.de/).')
st.write('[Source code](https://github.com/petrifiedvoices/ancient-classifier), forked from [CHCAA Ancient-classifier](https://github.com/centre-for-humanities-computing/ancient-classifier/). Outcome of the [Epigraphic Roads](https://github.com/sdam-au/epigraphic_roads/) project.')
         
# display image using streamlit
# width is used to set the width of an image

# st.image(img, width=200)    
st.write('We welcome any feedback or criticism at petra.hermankova@cas.au.dk!')
# ===
# More info about the model
# ===
