# 🦅 Resume Classification using Falcon-7B

This project demonstrates how to use the **Falcon-7B** large language model (LLM) for resume classification and analysis using **in-context prompting**. Instead of traditional model training, we leverage Falcon's powerful text generation abilities to classify and evaluate resumes via direct inference.

## 🔍 What is Falcon-7B?

Falcon-7B is a **decoder-only** transformer model developed by the **Technology Innovation Institute (TII)**. It is designed for tasks like:

- ✅ Text generation  
- 🗃️ Classification  
- 📄 Summarization  

| Feature         | Description                          |
|----------------|--------------------------------------|
| Type            | Decoder-only (like GPT)              |
| Parameters      | 7 Billion                            |
| Best for        | Long, complex input text             |
| Provider        | Hugging Face (tiiuae/falcon-7b)      |
| Format          | Python Transformers API              |

---

## 🧠 Why Not BERT?

| BERT                | Falcon-7B                          |
|---------------------|------------------------------------|
| Encoder-only        | Decoder-only                       |
| Best for classification | Best for generation & classification |
| Lightweight         | Heavy, GPU recommended             |
| Short input limit   | Supports longer input sequences    |

---

## 🚀 What You Can Do

Using Falcon-7B, we implement **zero-shot and few-shot prompting** to:

- 🔍 Ask: _"Is this a good resume for a Data Science role?"_
- 🧾 Generate feedback for a candidate's resume
- 🗂️ Classify resumes into job categories


## ⚙️ Setup in Google Colab

### 1. Clone this repository or open your own notebook

### 2. Install dependencies

```python
!pip install transformers accelerate
```

### 3. Load the Falcon-7B model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

---

### 4. Run Resume Inference

```python
resume_text = """
[Paste resume text here]
"""

prompt = f"Given this resume: {resume_text} \n\nWhich job category is it most suitable for?"

result = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
print(result)
```

---

## 💡 Notes

* This is an **inference-only** setup. Falcon-7B is not fine-tuned on your data.
* It is ideal for **quick prototyping** and qualitative feedback.
* You can enhance predictions using **few-shot examples** in the prompt.

---

## 📁 Dataset

The resumes used for testing are sourced from a publicly available dataset (e.g., Kaggle). Each record contains:

* `resume` (text)
* `category` (label)

---

## 📌 Future Improvements

* Implement batch inference
* Experiment with prompt engineering
* Deploy as a web-based app using Gradio or Streamlit

---

## 🧑‍💻 Author

Waseem Zahid
Final Year CS Student | AI Enthusiast | Resume Quality Prediction Research

