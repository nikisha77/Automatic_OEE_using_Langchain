# Packaging OEE Dashboard & Conversational AI

This project provides an interactive dashboard and AI chatbot for analyzing Overall Equipment Effectiveness (OEE) data for packaging devices. The solution includes:

- **Synthetic Data Generation** (`gen_data.py`)
- **Streamlit Dashboard & Conversational AI** (`app.py`)

---

## 1. Setup Instructions

### Prerequisites

- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/)
- [Streamlit](https://streamlit.io/)
- Required Python packages (see below)

### Install Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## 2. Generate Synthetic Data

Before running the dashboard, you must generate the synthetic OEE dataset.

Run the following command in your terminal:

```bash
python gen_data.py
```

This will create a file named `oee_data.xlsx` in your project directory.

---

## 3. Set Up Environment Variables

The Streamlit app requires a `GROQ_API_KEY` for the conversational AI features.

- Create a `.env` file in the project root.
- Add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 4. Launch the Streamlit App

Start the dashboard and chatbot with:

```bash
streamlit run app.py
```

- Use the sidebar to filter and explore OEE data visually.
- Switch to the "Conversational AI" tab to ask questions about OEE using natural language.

---

## Project Structure

```
.
├── app.py           # Streamlit dashboard & chatbot
├── gen_data.py      # Synthetic data generator
├── oee_data.xlsx    # Generated data file (after running gen_data.py)
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## Notes

- Always run `gen_data.py` first if you want to regenerate or update the data.
- The dashboard will not work unless `oee_data.xlsx` is present.
- For best results, ensure your `GROQ_API_KEY` is valid and has sufficient quota.
