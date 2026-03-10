import streamlit as st
import joblib
import pandas as pd
import io
import re
import os
import html

# --- 1. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_models():
    # Ensure these paths match your GitHub repository structure
    main_model = joblib.load('Models/main_model.pkl')
    mlb_main = joblib.load('Models/mlb_main.pkl')
    sub_model = joblib.load('Models/sub_model.pkl')
    mlb_sub = joblib.load('Models/mlb_sub.pkl')
    
    return main_model, mlb_main, sub_model, mlb_sub

main_model, mlb_main, sub_model, mlb_sub = load_models()

# --- 2. PROCESSING FUNCTIONS ---

def vtt_to_df_streamlit(uploaded_file):
    """Parses the uploaded VTT file into a DataFrame."""
    timestamp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}[\.,]\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}[\.,]\d{3})')
    data = []
    current_start, current_end, current_text = None, None, []
    
    # Decode the uploaded file stream
    content = uploaded_file.getvalue().decode("utf-8-sig", errors="ignore")
    lines = content.splitlines()

    for line in lines:
        line = line.strip()
        if line == "WEBVTT" or line.startswith("Kind:") or line.startswith("Language:"):
            continue
            
        match = timestamp_pattern.match(line)
        if match:
            if current_start and current_text:
                data.append({
                    'start_time': current_start, 
                    'end_time': current_end, 
                    'caption': " ".join(current_text).strip()
                })
            current_start, current_end, current_text = match.group(1), match.group(2), []
        elif current_start and line:
            current_text.append(line)
            
    if current_start and current_text:
        data.append({
            'start_time': current_start, 
            'end_time': current_end, 
            'caption': " ".join(current_text).strip()
        })
    return pd.DataFrame(data)

def remove_duplicate_captions(df, duration_threshold=0.3):
    if df.empty: return df
    
    def time_to_seconds(t_str):
        try:
            parts = t_str.replace(',', '.').split(':')
            return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        except: return 0.0

    df = df.copy()
    df['duration_seconds'] = df['end_time'].apply(time_to_seconds) - df['start_time'].apply(time_to_seconds)
    df['group_id'] = (df['caption'] != df['caption'].shift()).cumsum()

    indices_to_drop = []
    for _, group in df.groupby('group_id'):
        if len(group) > 1:
            keep_idx = group['duration_seconds'].idxmax()
            potential_drop = group.index.difference([keep_idx])
            for idx in potential_drop:
                if df.loc[idx, 'duration_seconds'] < duration_threshold:
                    indices_to_drop.append(idx)
                    
    return df.drop(indices_to_drop).drop(columns=['duration_seconds', 'group_id']).reset_index(drop=True)

def NSI_filter(df):
    special_chars = ["<", ">", "|", "(", ")", "[", "]", "♩", "♪", "♫", "♬", "♭", "♮", "♯", "#", ":"]
    if df.empty: return df
    pattern = '|'.join(map(re.escape, special_chars))
    return df[df['caption'].str.contains(pattern, regex=True, na=False)].reset_index(drop=True)

def remove_time_bold_italics(df):
    if df.empty: return df
    tag_pattern = re.compile(r'</?[ib]>', re.IGNORECASE)
    time_pattern = re.compile(r'\b\d{1,2}\s*:\s*\d{1,2}(?:\s*[:]\d+)*(?:\s*(?:AM|PM|am|pm))?\b')
    weird_emoji_pattern = re.compile(r'\[\s*Â\s*__\s*Â\s*\]|\[\s*__\s*\]', re.IGNORECASE)
    special_chars_pattern = r'[<>|()\[\]♩♪♫♬♭♮♯#:]'

    indices_to_drop = []
    for idx, row in df.iterrows():
        text = row['caption']
        if not re.search(special_chars_pattern, text): continue
        
        clean_text = tag_pattern.sub('', text)
        clean_text = time_pattern.sub('', clean_text)
        clean_text = weird_emoji_pattern.sub('', clean_text)
        
        if not re.search(special_chars_pattern, clean_text):
            indices_to_drop.append(idx)
            
    return df.drop(indices_to_drop).reset_index(drop=True)

def predict_caption(caption):
    main_pred = main_model.predict([caption])
    main_labels = mlb_main.inverse_transform(main_pred)[0]
    sub_pred = sub_model.predict([caption])
    sub_labels = mlb_sub.inverse_transform(sub_pred)[0]
    return ";".join(main_labels) if main_labels else "nan", ";".join(sub_labels) if sub_labels else "nan"

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="VTT Classifier", layout="wide")
st.title("VTT Processing Pipeline")

uploaded_file = st.file_uploader("Upload a VTT file", type=["vtt"])

if uploaded_file:
    with st.status("Processing data...", expanded=True) as status:
        st.write("Parsing VTT...")
        df = vtt_to_df_streamlit(uploaded_file)
        
        st.write("Cleaning captions...")
        df['caption'] = df['caption'].apply(html.unescape)
        df = remove_duplicate_captions(df)
        df = NSI_filter(df)
        df = remove_time_bold_italics(df)
        
        if not df.empty:
            st.write("Classifying with ML models...")
            # Unpack the tuple into two columns
            df[['Main_Labels', 'Sub_Labels']] = df['caption'].apply(lambda x: pd.Series(predict_caption(x)))
            status.update(label="Processing Complete!", state="complete", expanded=False)
            
            st.subheader("Processed Results")
            st.dataframe(df, use_container_width=True)
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"processed_{uploaded_file.name}.csv",
                mime="text/csv"
            )
        else:
            status.update(label="No NSI captions found.", state="error")
            st.warning("The processing pipeline removed all captions based on your filter criteria.")