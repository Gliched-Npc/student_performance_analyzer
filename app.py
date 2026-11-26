import os
from pathlib import Path
import streamlit as st
import pandas as pd
import requests


# --------------------------
# Load Dataset
# --------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"c:\Data Science\week 32\project_1\students_performance.csv")


df = load_data()

# --------------------------
# Compute Averages
# --------------------------
avg_scores = df.groupby("Subject")["Score"].mean().to_dict()

# --------------------------
# Title
# --------------------------
st.title("📊 Student Performance Analyzer")


# --------------------------
# Select Student
# --------------------------
student_id = st.selectbox("Select Student ID", df["StudentID"].unique())
student_data = df[df["StudentID"] == student_id]

st.subheader(f"Performance for {student_data.iloc[0]['Name']}")
st.dataframe(student_data)


# --------------------------
# Identify Weak Subjects
# --------------------------


weak_subjects = student_data[student_data["Score"] < student_data["Subject"].map(avg_scores)]["Subject"].tolist()
failed_subjects = student_data[student_data["Score"] < 40]["Subject"].tolist()

if weak_subjects:
    st.warning(f"Weak subjects (below class average): {', '.join(weak_subjects)}")
if failed_subjects:
    st.error(f"Failed subjects: {', '.join(failed_subjects)}")
if not weak_subjects and not failed_subjects:
    st.success("This student is doing well in all subjects!")


# --------------------------
# Groq API Call Function
# --------------------------


def get_groq_api_key():
    """Return GROQ_API_KEY from environment, local .env file, or Streamlit secrets."""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip().removeprefix("export ").strip() == "GROQ_API_KEY":
                return value.strip().strip('"').strip("'")

    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def _build_groq_payload(model, prompt):
    """Build chat completion payload with reasoning settings suited to the model."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful tutor creating interactive quizzes."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_completion_tokens": 2048,
    }
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"
    return payload


def _pick_fallback_groq_model(headers, excluded_model):
    """Fetch available Groq models and return the first active chat model."""
    resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=15)
    if resp.status_code != 200:
        return None
    skip_terms = ("whisper", "guard", "tts", "orpheus", "embed")
    candidates = [
        m.get("id")
        for m in resp.json().get("data", [])
        if m.get("id")
        and m.get("id") != excluded_model
        and not any(term in m["id"].lower() for term in skip_terms)
    ]
    return candidates[0] if candidates else None


def groq_generate(prompt):
    """Generate a chat completion using Groq's OpenAI-compatible API."""
    try:
        api_key = get_groq_api_key()
        if not api_key:
            st.error("GROQ_API_KEY not found in .env, environment variables, or Streamlit secrets.")
            return "API key missing"

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        model = "openai/gpt-oss-20b"

        response = requests.post(url, headers=headers, json=_build_groq_payload(model, prompt), timeout=30)

        if response.status_code == 404 and "model_not_found" in response.text:
            fallback_model = _pick_fallback_groq_model(headers, model)
            if fallback_model:
                response = requests.post(
                    url,
                    headers=headers,
                    json=_build_groq_payload(fallback_model, prompt),
                    timeout=30,
                )

        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return "API request failed"

        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]

    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        return "Unable to generate response"


# --------------------------
# Buttons for AI Actions
# --------------------------


if st.button("📚 Generate Study Tips"):
    if weak_subjects or failed_subjects:
        subject = weak_subjects[0] if weak_subjects else failed_subjects[0]
        student_name = student_data.iloc[0]['Name']
        score = student_data[student_data['Subject'] == subject]['Score'].iloc[0]
        class_avg = avg_scores[subject]
        
        prompt = f"""A student named {student_name} scored {score}/100 in {subject}. 
        The class average for {subject} is {class_avg:.1f}. 
        Give 3 practical and specific study tips to help them improve in {subject}. 
        Keep tips actionable and encouraging."""
        
        with st.spinner("Generating personalized study tips..."):
            tips = groq_generate(prompt)
            st.success("✅ Study Tips Generated!")
            st.write(tips)
    else:
        st.info("🎉 Great! This student is performing well in all subjects.")

# -----------------
# Interactive Quiz 
# -----------------

if st.button("🎮 Start Interactive Quiz", help="Answer all questions then see results"):
    if weak_subjects or failed_subjects:
        subject = weak_subjects[0] if weak_subjects else failed_subjects[0]
        st.session_state.interactive_mode = True
        st.session_state.quiz_subject = subject
        st.session_state.quiz_completed = False
        st.rerun()
    else:
        st.info("🎉 No practice questions needed - student is doing well!")


# --------------------------
# Interactive Quiz Implementation
# --------------------------

if hasattr(st.session_state, 'interactive_mode') and st.session_state.interactive_mode:
    st.markdown("---")
    st.subheader("🎮 Interactive Quiz Mode")
    
    subject = st.session_state.quiz_subject
    
    # Generate structured quiz data only once
    if 'quiz_data' not in st.session_state:
        prompt = f"""Create 3 multiple choice questions for {subject} at high school level.
        Format each question EXACTLY as shown below, using || as separators:
        
        Q1||[Question text]||A)[Option A]||B)[Option B]||C)[Option C]||D)[Option D]||ANSWER:[A/B/C/D]
        Q2||[Question text]||A)[Option A]||B)[Option B]||C)[Option C]||D)[Option D]||ANSWER:[A/B/C/D]
        Q3||[Question text]||A)[Option A]||B)[Option B]||C)[Option C]||D)[Option D]||ANSWER:[A/B/C/D]
        
        Make questions challenging but fair for high school level."""
        
        with st.spinner("Creating interactive quiz..."):
            quiz_data = groq_generate(prompt)
            st.session_state.quiz_data = quiz_data
    
    quiz_data = st.session_state.quiz_data
    
    if quiz_data and "||" in quiz_data:
        questions = [q.strip() for q in quiz_data.split('\n') if q.strip() and '||' in q]
        
        if questions:
            parsed_questions = []
            
            # Parse all questions
            for i, question_line in enumerate(questions[:3], 1):
                parts = question_line.split('||')
                if len(parts) >= 7:
                    question_data = {
                        'number': i,
                        'text': parts[1],
                        'options': [parts[2], parts[3], parts[4], parts[5]],
                        'correct_answer': parts[6].replace('ANSWER:', '').strip()
                    }   
                    parsed_questions.append(question_data)
            
            # Display all questions
            user_answers = {}
            for question in parsed_questions:
                st.markdown(f"### Question {question['number']}")
                st.write(question['text'])
                
                # Display options
                for option in question['options']:
                    st.write(option)
                
                # User answer selection
                user_answer = st.radio(
                    f"Your answer for Question {question['number']}:",
                    ['A', 'B', 'C', 'D'],
                    key=f"q{question['number']}_interactive",
                    horizontal=True
                )
                user_answers[question['number']] = user_answer
                st.markdown("---")
            
            # Submit Quiz Button
            if st.button("📊 Submit Quiz & Show Results", key="submit_quiz"):
                st.session_state.quiz_completed = True
                st.session_state.user_answers = user_answers
                st.session_state.parsed_questions = parsed_questions
                st.rerun()
            
            # Show results if quiz is completed
            if hasattr(st.session_state, 'quiz_completed') and st.session_state.quiz_completed:
                st.markdown("---")
                st.subheader("📋 Quiz Results")
                
                correct_count = 0
                total_questions = len(st.session_state.parsed_questions)
                
                # Show all answers together
                for question in st.session_state.parsed_questions:
                    user_answer = st.session_state.user_answers[question['number']]
                    correct_answer = question['correct_answer']
                    is_correct = user_answer == correct_answer
                    
                    if is_correct:
                        correct_count += 1
                        st.success(f"**Q{question['number']}:** Your answer: **{user_answer}** ✅ Correct!")
                    else:
                        st.error(f"**Q{question['number']}:** Your answer: **{user_answer}** ❌ Correct answer: **{correct_answer}**")
                
                # Overall score
                score_percentage = (correct_count / total_questions) * 100
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Score", f"{correct_count}/{total_questions}")
                with col2:
                    st.metric("Percentage", f"{score_percentage:.1f}%")
                with col3:
                    if score_percentage >= 70:
                        st.metric("Grade", "🎉 Great!")
                        st.balloons()
                    elif score_percentage >= 50:
                        st.metric("Grade", "👍 Good")
                    else:
                        st.metric("Grade", "📚 Keep Studying")
                
                # Performance message
                if score_percentage >= 80:
                    st.success("🌟 Excellent work! You have a strong understanding of this subject.")
                elif score_percentage >= 60:
                    st.info("👍 Good job! Review the questions you missed to improve further.")
                else:
                    st.warning("📖 Keep practicing! Focus on the areas where you got questions wrong.")
            
            # Quiz control buttons
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Take New Quiz"):
                    # Clear all quiz session state
                    for key in ['interactive_mode', 'quiz_data', 'quiz_completed', 'user_answers', 'parsed_questions']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            
            with col2:
                if st.button("❌ Exit Quiz"):
                    # Clear all quiz session state
                    for key in ['interactive_mode', 'quiz_data', 'quiz_completed', 'user_answers', 'parsed_questions']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                    
    else:
        st.error("Error generating interactive quiz. Please try again.")
        if st.button("🔄 Try Again"):
            if 'quiz_data' in st.session_state:
                del st.session_state.quiz_data
            st.rerun()

# --------------------------
# Show Class Average Comparison
# --------------------------

st.subheader("📈 Class Average vs Student Score")
comparison_df = student_data.copy()
comparison_df["Class_Average"] = comparison_df["Subject"].map(avg_scores)
st.dataframe(comparison_df)
