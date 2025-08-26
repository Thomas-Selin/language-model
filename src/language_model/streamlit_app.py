import streamlit as st
from serving import generate_text, find_latest_model, load_tokenizer, GPTLanguageModel
import torch
import os


@st.cache_resource
def load_model_and_tokenizer(tokenizer_type='subword'):
    """Load and cache the model and tokenizer. This will only run once per session."""
    
    # Check available memory before loading
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_free = gpu_memory - gpu_allocated
        print(f"💾 GPU Memory: {gpu_free:.1f}GB free / {gpu_memory:.1f}GB total")
        
        if gpu_free < 2.0:  # Less than 2GB free
            print("⚠️  Warning: Low GPU memory, applying aggressive optimizations")
    
    latest_model = find_latest_model()
    print(f"🔄 Loading model from: {latest_model}")
    
    tokenizer = load_tokenizer(tokenizer_type, os.path.dirname(latest_model))
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    model = GPTLanguageModel(vocab_size=tokenizer.get_vocab_size())
    
    # Load the PyTorch model directly
    checkpoint = torch.load(latest_model, map_location=device)
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Apply energy-efficient optimizations
    from serving import quantize_model
    model = quantize_model(model, device)
    
    # Enable torch optimizations
    if hasattr(torch, 'compile') and device.type != 'mps':
        model = torch.compile(model, mode='reduce-overhead')
    
    model.eval()
    
    # Clean up checkpoint from memory
    del checkpoint
    from serving import cleanup_memory
    cleanup_memory()
    
    print("✅ Model loaded and optimized successfully")
    return model, tokenizer, device, latest_model

st.title('Language Model Testground')

# Load model and tokenizer once at startup
with st.spinner('🔄 Loading model (this happens only once per session)...'):
    model, tokenizer, device, latest_model_path = load_model_and_tokenizer()
    st.success('✅ Model loaded successfully!')

# Show which model is being used (using cached path)
try:
    st.info(f"🤖 Using model: `{os.path.basename(latest_model_path)}` from `{os.path.dirname(latest_model_path)}`")
except Exception as e:
    st.error(f"❌ Error finding model: {e}")

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
            # Use the pre-loaded model, tokenizer, and device
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
