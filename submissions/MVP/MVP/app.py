import streamlit as st
import pandas as pd
import json
import tempfile
import os

# ==============================
# App Configuration
# ==============================
st.set_page_config(
    page_title="MVP",
    page_icon="",
    layout="centered"
)

# initialize session state
if 'example_mgf' not in st.session_state:
    st.session_state['example_mgf'] = None
if 'example_json' not in st.session_state:
    st.session_state['example_json'] = None

# ==============================
# Introductory Section
# ==============================
st.title("MVP Playground")

st.markdown("""
This web app lets you test our trained model on your own data.
            
### 📚 References
🔗 **Paper:** [Read the publication here](https://github.com/HassounLab/MVP)  
📦 **Source Code:** [GitHub Repository](https://github.com/HassounLab/MVP)

---

### 🧠 Available Models
We have two models trained on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) training dataset:
- **binnedSpec** – trained on binned spectra and does not require formula information.
- **formSpec** – our main model trained on spectra with subformula annotations. Requires formula and adduct information.

---

### ⚙️ Instructions

1. **Prepare two input files:**
   - **Spectra file (.mgf)** – your experimental spectra data.
   - **Candidates file (.json)** – candidate molecules for each spectrum.

2. **Select a model** from the dropdown.

3. **Click “Run Prediction”** to start processing.  
   ⚠️ **Note:** For fair usage, the web app limits computation to **1,000 pairs**. Each pair consists of one spectrum and one candidate molecule.

4. After processing, you’ll receive a downloadable **CSV file** with your results.

---

### 📁 Example Input Files

You can download example files to understand the required format:
- [Download sample spectra (MGF)](data/app/data.mgf)
- [Download sample candidates (JSON)](data/app/identifier_to_candidates.json)

Here's an example of the spectra file format (.mgf):
```
BEGIN IONS
TITLE=example_spectrum
PEPMASS=100.0
CHARGE=1+
FORMULA=C10H12O2 # optional, required for formSpec model
ADDUCT=[M+H]+ # optional, required for formSpec model
100.0 1000
101.0 1500
102.0 2000
END IONS
```
---

### 💡 Tip
If you want to process **more than 1,000 pairs**,  
please **clone the repository** and run it locally with GPU support for faster computation.
""")

# ==============================
# File Upload Section
# ==============================
st.subheader("📤 Upload Your Files")


# --- File uploaders ---
mgf_file = st.file_uploader("Upload spectra file (.mgf)", type=["mgf"])
json_file = st.file_uploader("Upload candidates file (.json)", type=["json"])

# --- Example files button ---
if st.button("Use Example Files"):
    with open("data/app/data.mgf", "rb") as f:
        st.session_state["example_mgf"] = f.read()
    with open("data/app/identifier_to_candidates.json", "rb") as f:
        st.session_state["example_json"] = f.read()
    st.success("✅ Example files loaded!")

# --- Determine which files to use ---
if mgf_file is not None:
    mgf_bytes = mgf_file.read()
elif "example_mgf" in st.session_state:
    mgf_bytes = st.session_state["example_mgf"]
else:
    mgf_bytes = None

if json_file is not None:
    json_bytes = json_file.read()
elif "example_json" in st.session_state:
    json_bytes = st.session_state["example_json"]
else:
    json_bytes = None

# --- Display results ---
if mgf_bytes and json_bytes:
    st.success("Files are ready to use!")
else:
    st.info("Please upload your files or 'Use Example Files'.")


# ==============================
# Model Selection and Run Button
# ==============================
model_choice = st.selectbox(
    "Select model to use:",
    options=["binnedSpec", "formSpec"]
)

run_button = st.button("🚀 Run Prediction")

# ==============================
# Run Prediction
# ==============================
if run_button:
    if not mgf_bytes or not json_bytes:
        st.error("Please upload both a spectra (.mgf) and candidates (.json) file.")
    else:
        with st.spinner("Running predictions... please wait ⏳", show_time=True):
            # Save uploaded files to temporary paths
            st.write("Saving files to temporary paths...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mgf") as tmp_mgf:
                tmp_mgf.write(mgf_bytes)
                mgf_path = tmp_mgf.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json:
                tmp_json.write(json_bytes)
                candidates_pth = tmp_json.name

            # Check number of pairs in candidates file
            st.write("Checking number of pairs in candidates file...")
            with open(candidates_pth, 'r') as f:
                candidates_data = json.load(f)
            total_pairs = sum(len(cands) for cands in candidates_data.values())
            if total_pairs > 1000:
                st.error(f"⚠️ Too many pairs ({total_pairs})! Please limit to 1,000 pairs for the web app.")
                st.stop()

            # preprocess spectra
            st.write("Preprocessing spectra...")
            from utils_app import preprocess_spectra, setup_config, run_inference
            dataset_pth, subformula_dir = preprocess_spectra(mgf_path, model_choice, mass_diff_thresh=20)

            if dataset_pth is None:
                st.error("Error in preprocessing spectra. Please check your input files.")
                if model_choice == "formSpec":
                    st.info("Make sure that for 'formSpec' model, each spectrum has 'formula' and 'adduct' metadata.")
                st.stop()

            # Prepare model config paths
            st.write("Preparing model config paths...")
            params = setup_config(model_choice, dataset_pth, candidates_pth, subformula_dir)

            try:
                st.write("Running inference...")
                run_inference(params)
            except Exception as e:
                st.error(f"Error running model inference: {e}")
                st.stop()

            # Convert to CSV
            st.write("Converting to CSV...")
            df = pd.read_pickle(params['df_test_path'])
            csv_path = params['df_test_path'].replace(".pkl", ".csv")
            df.to_csv(csv_path, index=False)

            st.success(f"✅ Done! Model: {model_choice}")
            st.download_button(
                label="📥 Download Results CSV",
                data=open(csv_path, "rb").read(),
                file_name=os.path.basename(csv_path),
                mime="text/csv"
            )

        st.info("To run larger datasets or enable GPU acceleration, please clone the repo and run locally.")

# ==============================
# Footer
# ==============================
st.markdown("---")

