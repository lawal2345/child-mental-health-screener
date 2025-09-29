import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
from datetime import datetime

class MentalHealthScreener:
    def __init__(self):
        print("Initializing Mental Health Screener...")
        
        # Use CPU for Spaces (GPU available on paid tiers)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading Phi-3-mini model")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            torch_dtype="auto" if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
            
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        print("Model loaded successfully!")

    def screen_text(self, user_input, age_group="child"):
        """Analyzes text for mental health indicators using structured prompting"""
        prompt = f"""You are a clinical assessment tool designed to identify potential anxiety and depression indicators in children's expressions. Analyze the following text from a {age_group} and provide a structured assessment:

Text: "{user_input}"

Provide your assessment in this EXACT JSON format:
{{
    "risk_level": "low/moderate/high",
    "anxiety_indicators": ["list specific indicators found or empty list"],
    "depression_indicators": ["list specific indicators found or empty list"], 
    "key_concerns": ["list main concerns or empty list"],
    "recommended_action": "brief recommendation",
    "explanation": "brief clinical reasoning",
    "confidence": "low/medium/high"
}}

Base your assessment on established clinical indicators for childhood anxiety and depression. Be thorough but cautious."""

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False
            )
            
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Parse JSON from response
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                result = json.loads(json_str)
                return result
            else:
                return {"error": "Could not parse structured response", "raw": response}
        except Exception as e:
            return {"error": str(e), "raw": response}

# Initialize the screener globally (will load once when Space starts)
print("Starting Mental Health Screener")
screener = MentalHealthScreener()

def screen_text_interface(text, age_group):
    """Interface function for Gradio"""
    if not text.strip():
        return "Please enter some text to analyze."
    
    try:
        start_time = time.time()
        
        # Perform screening
        result = screener.screen_text(text, age_group)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if "error" in result:
            return f"Error in analysis: {result['error']}\n\nRaw response: {result.get('raw', 'N/A')}"
        
        # Format the output nicely
        risk_level = result.get('risk_level', 'unknown').upper()
        
        # Color coding
        risk_colors = {
            "LOW": "🟢",
            "MODERATE": "🟡", 
            "HIGH": "🔴"
        }
        risk_emoji = risk_colors.get(risk_level, "⚪")
        
        output = f"""# {risk_emoji} Risk Assessment: {risk_level}

## 🧠 Anxiety Indicators
{format_list(result.get('anxiety_indicators', []))}

## 💭 Depression Indicators  
{format_list(result.get('depression_indicators', []))}

## Key Concerns
{format_list(result.get('key_concerns', []))}

## Recommended Action
{result.get('recommended_action', 'N/A')}

## Clinical Reasoning
{result.get('explanation', 'N/A')}

## Model Confidence
**{result.get('confidence', 'unknown').upper()}**

---
*Response time: {response_time:.2f} seconds*

**IMPORTANT DISCLAIMER:** This is a screening tool prototype for research purposes only. It is NOT a diagnostic instrument and should not be used for clinical decision-making without professional oversight."""

        return output
        
    except Exception as e:
        return f"Error: {str(e)}"

def format_list(items):
    """Format list items as markdown"""
    if not items:
        return "*None identified*"
    return "\n".join([f"- {item}" for item in items])

# Create Gradio interface
with gr.Blocks(title="Child Mental Health Screener", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Child Mental Health Screening Tool
    ### Early Identification System for Research
    
    This tool uses generative language models to analyze text and identify potential indicators 
    of anxiety and depression in children, aligned with the THRIVE framework for mental health support.
    
    **How to use:** Enter text describing a child's feelings, behaviors, or concerns, select the age group, 
    and click "Analyze" to receive a structured assessment.
    
    **First analysis may take 30-60 seconds as the model loads. Subsequent analyses will be faster.**
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text to Analyze",
                placeholder="Example: 'I feel worried all the time and can't sleep at night. My tummy hurts before school.'",
                lines=6
            )
            
            age_group = gr.Dropdown(
                choices=["child (5-11)", "young person (12-18)", "parent observation"],
                value="child (5-11)",
                label="Age Group / Context"
            )
            
            submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
        
        with gr.Column(scale=3):
            output = gr.Markdown(label="Assessment Results")
    
    # Examples
    gr.Markdown("### Example Inputs (click to try)")
    gr.Examples(
        examples=[
            ["I feel worried all the time and can't sleep at night. My tummy hurts before school.", "child (5-11)"],
            ["Nobody likes me. I don't want to go to school anymore. I just want to stay in bed.", "young person (12-18)"],
            ["I'm having a great day and looking forward to playing with my friends!", "child (5-11)"],
            ["My child has become very withdrawn, refuses to eat, and cries frequently without clear reason.", "parent observation"],
        ],
        inputs=[text_input, age_group]
    )
    
    submit_btn.click(
        fn=screen_text_interface,
        inputs=[text_input, age_group],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### About This Tool
    
    **Purpose:** Demonstrate responsible AI for early identification in child mental health pathways
    
    **Technical Approach:**
    - **Model:** Microsoft Phi-3-mini-4k-instruct
    - **Method:** Structured prompt engineering with clinical reasoning
    - **Features:** Explainable outputs, confidence scoring, age-appropriate analysis
    
    **Responsible AI Principles:**
    - **Transparency:** All reasoning is explained
    - **Interpretability:** Clear indicator identification  
    - **Confidence scoring:** Model expresses uncertainty
    - **Human oversight:** Designed to support, not replace, clinical judgment
    
    **Research Alignment:** This prototype aligns with the NIHR HealthTech Research Centre's focus on 
    developing AI-informed tools for child mental health, particularly the THRIVE framework approach 
    to early identification and needs-based signposting.
    
    **Cambridge PhD Application:** Developed by Lawal Jesutofunmi as part of application for PhD in Psychiatry, 
    University of Cambridge, under supervision of Dr. Anna Moore (Timely Group).
    
    ---
    **RESEARCH PROTOTYPE ONLY** - Not for clinical use
    """)

if __name__ == "__main__":
    demo.launch()
