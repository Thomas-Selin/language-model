import streamlit as st
from serving import generate_text, find_latest_model, load_tokenizer, GPTLanguageModel, safe_open
import torch


@st.cache_resource
def load_model_and_tokenizer(tokenizer_type='subword'):
    latest_model = find_latest_model()
    tokenizer = load_tokenizer(tokenizer_type, latest_model)
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    model = GPTLanguageModel(vocab_size=tokenizer.get_vocab_size())
    with safe_open(f'{latest_model}/model.safetensors', framework='pt') as f:
        for k in f.keys():
            model.state_dict()[k].copy_(f.get_tensor(k))
    model = model.to(device)
    
    # Apply energy-efficient optimizations
    from serving import quantize_model
    model = quantize_model(model, device)
    
    # Enable torch optimizations
    if hasattr(torch, 'compile') and device.type != 'mps':
        model = torch.compile(model, mode='reduce-overhead')
    
    model.eval()
    return model, tokenizer, device

st.title('Language Model Testground')

user_input = st.text_area('Enter your prompt:', height=100)

# Tokenizer selection
# st.subheader('Tokenizer Settings')
# tokenizer_type = st.selectbox(
#     'Tokenizer',
#     options=['subword'],
#     format_func=lambda x: {'subword': 'Subword'}[x]
# )


# max_tokens = st.slider('Max new tokens', 1, 1000, 5)
temperature = st.slider('Temperature', 0.01, 1.0, 0.8, 0.05)

# Option to show attention visualizations
show_attention = st.checkbox('Show attention visualizations', value=True)



if st.button('Generate'):
    if user_input.strip():
        with st.spinner('Generating...'):
            model, tokenizer, device = load_model_and_tokenizer()
            result, all_attentions, tokenizer_obj = generate_text(
                prompt=user_input, max_new_tokens=200,
                temperature=temperature, tokenizer_type='subword',
                model=model, tokenizer=tokenizer, device=device)
            
            # Clean up memory after generation
            from serving import cleanup_memory
            cleanup_memory()
        
        st.subheader('Generated Output:')
        st.write(result)
        if show_attention:
            from serving import visualize_combined_attention, visualize_attention
            st.subheader('Attention Visualizations')
            mean_fig = visualize_combined_attention(result, all_attentions, tokenizer_obj, aggregation='mean')
            max_fig = visualize_combined_attention(result, all_attentions, tokenizer_obj, aggregation='max')
            layer_figs = []
            for layer_idx in range(2):  # Assuming 2 layers
                fig = visualize_attention(result, all_attentions, tokenizer_obj, layer_idx=layer_idx)
                layer_figs.append(fig)
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
        # else: do not show attention visualizations
    else:
        st.warning('Please enter a prompt.')
