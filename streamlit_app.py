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
        if line.isdigit():  # This skips the "1", "2", "3" sequence numbers
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

def NSI_filter(df, special_chars):
    if df.empty: return df
    pattern = '|'.join(map(re.escape, special_chars))
    return df[df['caption'].str.contains(pattern, regex=True, na=False)].reset_index(drop=True)

def remove_time_bold_italics(df, special_chars):
    if df.empty: return df
    tag_pattern = re.compile(r'</?[ib]>', re.IGNORECASE)
    time_pattern = re.compile(r'\b\d{1,2}\s*:\s*\d{1,2}(?:\s*[:]\d+)*(?:\s*(?:AM|PM|am|pm))?\b')
    weird_emoji_pattern = re.compile(r'\[\s*Â\s*__\s*Â\s*\]|\[\s*__\s*\]', re.IGNORECASE)
    bg_transparent_pattern = re.compile(r'</?c\.bg_transparent>', re.IGNORECASE)
    
    # Generate the pattern dynamically from the user's list
    special_chars_pattern = '|'.join(map(re.escape, special_chars))

    indices_to_drop = []
    for idx, row in df.iterrows():
        text = row['caption']
        # If the row doesn't even have the target chars, skip cleaning logic
        if not re.search(special_chars_pattern, text): continue
        
        clean_text = tag_pattern.sub('', text)
        clean_text = time_pattern.sub('', clean_text)
        clean_text = weird_emoji_pattern.sub('', clean_text)
        clean_text = bg_transparent_pattern.sub('', clean_text)
        
        # If NO special characters remain after stripping time/tags, it was a false positive
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

with st.expander("⚙️ Parsing Settings", expanded=False):
    # NSI Filter Toggle and Character List
    do_nsi_filter = st.toggle("NSI Filter", value=True, help='Filters for captions that contain <, >, |, (, ), [, ], ♩, ♪, ♫, ♬, ♭, ♮, ♯, #, :')
    
    # Only show character list if NSI Filter is ON
    special_chars_input = st.text_input(
        "Special Characters to filter for:", 
        value="<, >, |, (, ), [, ], ♩, ♪, ♫, ♬, ♭, ♮, ♯, #, :",
        disabled=not do_nsi_filter
    )
    # Convert input string back to a list of characters
    active_chars = [c.strip() for c in special_chars_input.split(',')]

    # False Positive Filter Toggle
    do_false_positive_filter = st.toggle(
        "Remove Common Falsely Flagged NSI", 
        value=True, 
        help="Removes lines of captions that only contain time (E.g., 11:52AM), italics (e.g., <i> hi! </i>), or bold (e.g., <b> bye! </b>) as their only possible NSI flag"
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        use_main = st.checkbox("Tag Primary NSI Class?", value=True)
    with col2:
        use_sub = st.checkbox("Tag Secondary NSI Class?", value=True)

uploaded_files = st.file_uploader("Upload VTT files (Max 100 files)", type=["vtt"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 100:
        st.error("Please upload a maximum of 100 files at a time.")
    else:
        all_dfs = [] # To hold data from every file
        
        with st.status("Processing all files...", expanded=True) as status:
            for uploaded_file in uploaded_files:
                st.write(f"Processing: {uploaded_file.name}")
                
                # 1. Parse & Tag with filename
                temp_df = vtt_to_df_streamlit(uploaded_file)
                if temp_df.empty: continue
                
                temp_df['file_name'] = uploaded_file.name
                
                # 2. Run Cleaning Pipeline
                temp_df['caption'] = temp_df['caption'].apply(html.unescape)
                temp_df = remove_duplicate_captions(temp_df)

                if do_nsi_filter:
                    temp_df = NSI_filter(temp_df, active_chars)
                
                if do_false_positive_filter:
                    temp_df = remove_time_bold_italics(temp_df, active_chars)
                
                if not temp_df.empty:
                    # Only run classification if at least one checkbox is checked
                    if use_main or use_sub:
                        st.write(f"Classifying: {uploaded_file.name}...")
                        
                        def get_predictions(text):
                            # Call ML models ONLY for the requested columns
                            main_lab, sub_lab = "not_classified", "not_classified"
                            
                            # Real prediction logic
                            full_main, full_sub = predict_caption(text)
                            
                            if use_main: main_lab = full_main
                            if use_sub: sub_lab = full_sub
                            
                            return main_lab, sub_lab

                        temp_df[['Main_Labels', 'Sub_Labels']] = temp_df['caption'].apply(
                            lambda x: pd.Series(get_predictions(x))
                        )
                    else:
                        # User turned off all classification
                        temp_df['Main_Labels'] = "not_classified"
                        temp_df['Sub_Labels'] = "not_classified"
                        
                    all_dfs.append(temp_df)
            
            if all_dfs:
                final_df = pd.concat(all_dfs, ignore_index=True)
                
                # Reorder columns
                cols = ['file_name'] + [c for c in final_df.columns if c != 'file_name']
                final_df = final_df[cols]

                # --- CHANGE START: Moved Download Button above Status/Preview ---
                csv = final_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                
                # Naming logic: use the first filename if single, or "batch" if multiple
                base_name = uploaded_files[0].name.replace('.vtt', '') if len(uploaded_files) == 1 else "batch"
                
                st.download_button(
                    label=f"📥 Download {len(all_dfs)} Processed Files as CSV",
                    data=csv,
                    file_name=f"parsed_NSI_{base_name}.csv",
                    mime="text/csv"
                )

                status.update(label="All files processed successfully!", state="complete", expanded=True)
                st.subheader("Combined Results Preview")
                st.dataframe(final_df, use_container_width=True)
            else:
                status.update(label="No NSI captions found.", state="error")
                st.warning("No captions matched the filter criteria.")
        
# --- 4. FOOTER ---
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: grey; font-size: 0.8em;">
        Having issues? Please contact Lloyd May at lloyd [dot] may [at] monash [dot] edu
    </div>
    """, 
    unsafe_allow_html=True
)