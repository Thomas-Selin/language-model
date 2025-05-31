import streamlit as st
from serving import generate_text

st.title('Sbab GPT Testground')

user_input = st.text_area('Enter your prompt:', height=100)
max_tokens = st.slider('Max new tokens', 10, 1000, 200)

if st.button('Generate'):
    if user_input.strip():
        with st.spinner('Generating...'):
            result = generate_text(user_input, max_new_tokens=max_tokens)
        st.subheader('Generated Output:')
        st.write(result)
    else:
        st.warning('Please enter a prompt.')
