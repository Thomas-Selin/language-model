import streamlit as st
from serving import generate_text
import matplotlib.pyplot as plt

st.title('Sbab GPT Testground')

user_input = st.text_area('Enter your prompt:', height=100)

# Tokenizer selection
# st.subheader('Tokenizer Settings')
tokenizer_type = st.selectbox(
    'Choose tokenizer',
    options=['char', 'word', 'subword'],
    format_func=lambda x: {'char': 'Character', 'word': 'Word', 'subword': 'Subword'}[x]
)

max_tokens = st.slider('Max new tokens', 1, 1000, 5)
temperature = st.slider('Temperature', 0.01, 2.0, 1.0, 0.05)

if st.button('Generate'):
    if user_input.strip():
        with st.spinner('Generating...'):
            result, mean_fig, max_fig, layer_figs = generate_text(
                user_input, max_new_tokens=max_tokens, temperature=temperature, tokenizer_type=tokenizer_type)
        
        st.subheader('Generated Output:')
        st.write(result)
        
        # Create tabs for different visualizations
        st.subheader('Attention Visualizations')
        tabs = st.tabs(["Combined (Mean)", "Combined (Max)", "Layer-specific", " ❓"])
        
        with tabs[0]:
            st.markdown("""
            ### Combined Mean Attention
            
            This visualization shows the average attention patterns across all layers and heads.
            Brighter colors indicate stronger average attention between tokens.
            
            This is useful for understanding what the model is generally focusing on when generating text.
            """)
            st.pyplot(mean_fig)
        
        with tabs[1]:
            st.markdown("""
            ### Combined Max Attention
            
            This visualization shows the maximum attention weight between any tokens across all layers and heads.
            It reveals the strongest connections, even if they only appear in a single head.
            
            Look for bright spots that might indicate important token relationships.
            """)
            st.pyplot(max_fig)
            
        with tabs[2]:
            st.markdown("""
            ### Layer-specific Attention
            
            These visualizations show attention patterns in individual layers.
            Different layers may focus on different linguistic aspects:
            - Earlier layers often capture more local patterns 
            - Later layers may focus on higher-level relationships
            """)
            for i, fig in enumerate(layer_figs):
                st.markdown(f"#### Layer {i}, Head 0")
                st.pyplot(fig)
        
        with tabs[3]:
            st.markdown("""
            ### Interpreting These Attention Plots:
            
            - **Y-axis ("From token")**: The token that is paying attention
            - **X-axis ("Attended to")**: The token receiving attention
            - **Color**: Brighter colors = stronger attention
            - **Numbers**: Actual attention weight values
            
            ### Key Patterns:
            - **Diagonal line**: Self-attention (tokens attending to themselves)
            - **Lower triangle**: Each token attending to previous tokens
            - **Dark upper triangle**: No attention to future tokens (causal mask)
            
            ### When to Use Each View:
            
            **Mean Attention**: 
            - For understanding typical behavior
            - For consistent patterns across heads
            - For general explanation of model behavior
            
            **Max Attention**:
            - For finding specialized attention heads
            - For investigating strongest influences
            - For identifying unique token relationships
            """    )
    else:
        st.warning('Please enter a prompt.')
