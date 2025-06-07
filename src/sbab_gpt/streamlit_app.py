import streamlit as st
from serving import generate_text

st.title('Sbab GPT Testground')

user_input = st.text_area('Enter your prompt:', height=100)
max_tokens = st.slider('Max new tokens', 1, 1000, 10)
temperature = st.slider('Temperature', 0.01, 2.0, 1.0, 0.05)

if st.button('Generate'):
    if user_input.strip():
        with st.spinner('Generating...'):
            result = generate_text(user_input, max_new_tokens=max_tokens, temperature=temperature)
        st.subheader('Generated Output:')
        st.write(result)
    else:
        st.warning('Please enter a prompt.')
