import streamlit as st
import torch
import os
import gc
import psutil
import logging
from language_model.serving import generate_text, find_latest_model, load_tokenizer, GPTLanguageModel
from language_model.helpers import get_device


@st.cache_resource
def load_model_and_tokenizer(tokenizer_type='subword', model_type='chat'):
    """Load and cache the model and tokenizer. This will only run once per session."""
    
    # Check system memory
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    total_gb = memory_info.total / (1024**3)
    logging.info(f"💾 System Memory: {available_gb:.1f}GB free / {total_gb:.1f}GB total")
    
    if available_gb < 4.0:  # Less than 4GB free
        st.warning(f"⚠️ Low system memory ({available_gb:.1f}GB free). Consider closing other applications.")
    
    # Check GPU memory if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_free = gpu_memory - gpu_allocated
        logging.info(f"💾 GPU Memory: {gpu_free:.1f}GB free / {gpu_memory:.1f}GB total")
        
        if gpu_free < 2.0:  # Less than 2GB free
            logging.warning("⚠️  Warning: Low GPU memory, applying aggressive optimizations")
    
    latest_model = find_latest_model(model_type)
    logging.info(f"🔄 Loading model from: {latest_model}")
    
    tokenizer = load_tokenizer(tokenizer_type, os.path.dirname(latest_model))
    device = get_device()
    model = GPTLanguageModel(vocab_size=tokenizer.get_vocab_size())
    
    # Load the PyTorch model directly
    checkpoint = torch.load(latest_model, map_location=device)
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Apply energy-efficient optimizations
    from language_model.serving import quantize_model
    model = quantize_model(model, device)
    
    # Enable torch optimizations (disabled for MPS to prevent issues)
    if hasattr(torch, 'compile') and device.type == 'cuda':  # Only on CUDA
        model = torch.compile(model, mode='reduce-overhead')
    
    model.eval()
    
    # Clean up checkpoint from memory
    del checkpoint
    from language_model.serving import cleanup_memory
    cleanup_memory()
    
    logging.info("✅ Model loaded and optimized successfully")
    return model, tokenizer, device, latest_model

def aggressive_cleanup():
    """Aggressive memory cleanup to prevent OOM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass  # synchronize might not be available in all MPS versions

def check_memory_usage():
    """Check current memory usage and warn if high"""
    memory_info = psutil.virtual_memory()
    used_percent = memory_info.percent
    available_gb = memory_info.available / (1024**3)
    
    if used_percent > 85:
        st.warning(f"⚠️ High memory usage ({used_percent:.1f}% used, {available_gb:.1f}GB free)")
        return False
    return True

st.title('Language Model Testground')

# Model selection
st.subheader('Model Selection')
model_type = st.radio(
    "Choose model type:",
    options=['chat', 'pretrained'],
    format_func=lambda x: '💬 Chat Fine-tuned Model (short single-turn questions in English)' if x == 'chat' else '🧠 Pre-trained Model',
    index=0,  # Default to chat model
    help="Chat model is fine-tuned for conversations, pre-trained model is the base model"
)

# Load model and tokenizer once at startup
with st.spinner(f'🔄 Loading {model_type} model (this happens only once per session)...'):
    model, tokenizer, device, latest_model_path = load_model_and_tokenizer(model_type=model_type)
    st.success('✅ Model loaded successfully!')

# Show which model is being used (using cached path)
try:
    st.info(f"🤖 Using model: `{os.path.basename(latest_model_path)}`")
    st.info(f"🔤 Tokenizer type: Subword (BPE)")
except Exception as e:
    st.error(f"❌ Error finding model: {e}")

# Option to show memory status
show_memory_status = st.checkbox('Show memory status', value=False)

if show_memory_status:
    # Memory status
    memory_info = psutil.virtual_memory()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Memory Usage", f"{memory_info.percent:.1f}%")
    with col2:
        st.metric("Available Memory", f"{memory_info.available / (1024**3):.1f}GB")
    with col3:
        # GPU memory status
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_free = gpu_memory - gpu_allocated
            st.metric("GPU Memory", f"{gpu_allocated:.1f}GB / {gpu_memory:.1f}GB")
        elif torch.backends.mps.is_available():
            st.metric("GPU Memory", "Mac GPU")
        else:
            st.metric("GPU Memory", "Not available")

user_input = st.text_area('Enter your prompt:', height=100)

# Limit max tokens to prevent OOM
max_tokens = st.slider('Max new tokens', 1, 100, 5, help="Lower values reduce memory usage")
temperature = st.slider('Temperature', 0.01, 1.0, 0.8, 0.05)

# Option to show attention visualizations (with warning)
show_attention = st.checkbox('Show attention visualizations (⚠️ Uses more memory)', value=False)

if show_attention:
    st.warning("⚠️ Attention visualizations use significant memory. Disable if you are experiencing crashes.")

if st.button('Generate'):
    if user_input.strip():
        # Check memory before generation
        if not check_memory_usage():
            st.error("❌ Cannot generate: Low memory. Try restarting the app or closing other applications.")
            st.stop()
        
        # Clean up before generation
        aggressive_cleanup()
        
        with st.spinner('Generating...'):
            try:
                # Use the pre-loaded model, tokenizer, and device with limited tokens
                result, all_attentions, tokenizer_obj = generate_text(
                    prompt=user_input, max_new_tokens=max_tokens,
                    temperature=temperature, tokenizer_type='subword',
                    model=model, tokenizer=tokenizer, device=device)
                
                # Clean up immediately after generation
                aggressive_cleanup()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    aggressive_cleanup()
                    st.error("❌ Out of memory! Try:")
                    st.write("- Reducing max tokens")
                    st.write("- Disabling attention visualizations")
                    st.write("- Restarting the app")
                    st.stop()
                else:
                    raise e
        
        st.subheader('Generated Output:')
        st.write(result)
        
        if show_attention and all_attentions:
            # Check memory before visualizations
            memory_info = psutil.virtual_memory()
            if memory_info.percent > 80:
                st.warning("⚠️ Skipping attention visualizations due to high memory usage")
            else:
                try:
                    from language_model.serving import (visualize_combined_input_attention, visualize_input_attention,
                                        visualize_combined_word_attention, visualize_word_to_word_attention)
                    
                    st.subheader('Attention Analysis')
                    
                    # Create tabs for different types of analysis
                    tab1, tab2 = st.tabs(["🔗 Word Relationships", "🎯 Input Focus"])
                    
                    with tab1:
                        st.info("🔗 Shows which words refer to or relate to other words (e.g., 'it' → 'bear' in the sentence \"The bear ate the cake because it was hungry\")")
                        
                        # Show word-to-word attention
                        combined_word_fig = visualize_combined_word_attention(result, all_attentions, tokenizer_obj, aggregation='mean')
                        if combined_word_fig:
                            st.pyplot(combined_word_fig)
                            del combined_word_fig
                            aggressive_cleanup()
                        
                        # Only show layer-specific if memory allows
                        memory_check = psutil.virtual_memory()
                        if memory_check.percent < 75:
                            layer_word_fig = visualize_word_to_word_attention(result, all_attentions, tokenizer_obj, layer_idx=0, head_idx=0)
                            if layer_word_fig:
                                st.subheader('Layer 0 Word Relationships')
                                st.pyplot(layer_word_fig)
                                del layer_word_fig
                                aggressive_cleanup()
                        else:
                            st.info("Skipped layer-specific visualization to conserve memory")

                    with tab2:
                        st.info("📊 Shows what parts of your input prompt the model focused on")
                        
                        # Show combined input attention
                        combined_input_fig = visualize_combined_input_attention(user_input, result, all_attentions, tokenizer_obj, aggregation='mean')
                        if combined_input_fig:
                            st.pyplot(combined_input_fig)
                            del combined_input_fig
                            aggressive_cleanup()

                        
                except Exception as e:
                    st.error(f"❌ Error generating visualizations: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    aggressive_cleanup()
        
        # Final cleanup
        aggressive_cleanup()
        
        # Show memory status after generation
        memory_info = psutil.virtual_memory()
        st.info(f"Memory after generation: {memory_info.percent:.1f}% used")
        
    else:
        st.warning('Please enter a prompt.')

# Add a cleanup button for manual memory management
if st.button('🧹 Clean Memory'):
    aggressive_cleanup()
    st.success('Memory cleaned!')
