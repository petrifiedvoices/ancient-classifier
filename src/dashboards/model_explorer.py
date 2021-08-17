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

    if model_choice == 'Reconstructed text (interpretive text)':
        default_input = 'Accae mulieris libertae Myrine Accae mulieris libertae Sympherusae Marco Antonio Marci liberto Ero pomario'

    elif model_choice == 'Text as-is on inscriptions (conservative text)':
        default_input = 'Accae l Myrine Accae l Sympherusae M Ant M l Ero poma'


# ===
# Classify input
# ===

st.title('Epigrapher-ex-machina: machine-learning classifier for Latin inscriptions')

st.header('What does it do?')
st.write('*Scenario 1:* Imagine you are an archaeologist, excavating an ancient settlement and you have found an inscription with Latin text. You are not an expert on inscriptions, but knowing what kind of text you are dealing with, while still in the field, would help your immediate understanding of the archaeological situation and would help you guide the excavation in the right direction.')
st.write('*Scenario 2:* Imagine you are a museum archivist and you have found an unlabelled inscription in the depository. You would like to be able to record in a museum catalogue its type, so the future experts can find it more easily.')

st.info('Enter the text of the inscription to the Classifier and it will tell you what kind of inscription you are dealing with!')  
st.write('The Classifier is trained on the text of 50,000 inscriptions to come up with the most probable epigraphic classification and its alternatives. However, you still should consult a human-epigrapher to be 100% sure!')

from PIL import Image

image = Image.open('src/dashboards/F000100_HD033469.jpeg')

st.image(image, caption="Inscription HD033469 from Lambaesis in Numidia, http://edh-www.adw.uni-heidelberg.de/edh/foto/F000100", use_column_width=None, clamp=False, channels='RGB', output_format='auto') 


st.header('How do I classify my text?')
st.markdown('1. Choose the best-fitting model in the dropdown menu in the left-side panel:')
st.markdown('- does your text have the format *as-is* on the inscribed medium, including unexpanded abbreviations, but does not contain any modern additions or editorial reconstructions, choose `Text as-is on inscriptions` option')
st.markdown('- does your text contain editorial additions, such as expanded abbreviations and modern reconstructions that are not present on the inscribed object, choose `Reconstructed text` option')
st.markdown('2. Insert the text into the input window below and click the `Classify me!` button.')
st.markdown('3. The results will appear below as confidence levels. The higher the value is for a given typological variant, the more confident the model is that your text belongs to that category. If the values are negative values, the model is less confident. If ambiguous, too short, or too fragmentary, the text can easily belong to multiple categories, without one clear favourite.')
st.markdown('4. You can interact with the results: hover over the typological variant to see a confidence value. Click on the three horizontal dots menu in the top right corner of the plot with results to can save the plot as SVG or PNG.')
st.markdown('5. If you don\'t have any inscriptions at hand but still want to play, use our sample inscription [`HD000470`](https://edh-www.adw.uni-heidelberg.de/edh/inschrift/HD000470) already present in the input window.')
st.markdown('*Don\'t forget - the `Epigraphic Classifier` is still experimentary and so may be the results!*')
    
user_input = st.text_area("Insert input text", default_input)

if not isinstance(user_input, str):
    raise TypeError('Input is not a string')

if len(user_input) > 1000:
    raise MemoryError('Input text is too long. Max 1000 characters is allowed')

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

st.header('About the Epigraphic Classifier')
st.write('Authors: Petra Hermankova, postdoc at Aarhus University, [ORCID:0000-0002-6349-0540](https://orcid.org/0000-0002-6349-0540), [@petrifiedvoices on Github](https://github.com/petrifiedvoices) & Jan Kostkan, CHCAA, Aarhus University, [@supplyandcommand on GitHub](https://github.com/supplyandcommand)')
st.write('The model was trained on Latin inscriptions from the [Epigraphic Database Heidelberg](https://edh-www.adw.uni-heidelberg.de/).')
st.write('[Source code](https://github.com/petrifiedvoices/ancient-classifier), forked from [CHCAA Ancient-classifier](https://github.com/centre-for-humanities-computing/ancient-classifier/).')
st.markdown('The Epigraphic Classifier was developed as part of the Digital Literacy 2.0 Program, 2020-2021, as on of the outcomes of the [Epigraphic Roads](https://github.com/sdam-au/epigraphic_roads/) project by Petra Hermankova, Aarhus University; created in collaboration with the Centre for Humanities Computing Aarhus (CHCAA).')

# display image using streamlit
# width is used to set the width of an image


st.write('We welcome any feedback and suggestions at petra.hermankova@cas.au.dk!')

# ===
# More info about the model
# ===

