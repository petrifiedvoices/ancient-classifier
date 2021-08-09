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
st.write('This model assigns standard typologies of inscriptions to a given Latin text and shows the confidence of assigned classifications. The model was trained on inscriptions from the [Epigraphic Database Heidelberg](https://edh-www.adw.uni-heidelberg.de/).[Source code](https://github.com/centre-for-humanities-computing/ancient-classifier/), forked from the [Epigraphic Roads](https://github.com/sdam-au/epigraphic_roads/) project.')

st.write('Authors: Petra Hermankova & Jan Kostkan, Aarhus University')

user_input = st.text_area("Insert text", default_input)

if not isinstance(user_input, str):
    raise TypeError('Input is not a string')

if len(user_input) > 1000:
    raise MemoryError('Input text too long. Max 1000 characters allowed')

if st.button('Classify!'):

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


# ===
# More info about the model
# ===
