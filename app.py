import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- Constants ---
CATEGORIES = {
    "🎨 Creative Arts": [
        ('drawing', 'Drawing'), ('dancing', 'Dancing'), ('singing', 'Singing'),
        ('acting', 'Acting'), ('photography', 'Photography'), ('designing', 'Designing'),
        ('creative_writing', 'Content writing'), ('crafting', 'Crafting'),
        ('cartooning', 'Cartooning'), ('makeup', 'Makeup'), ('listening_music', 'Listening Music')
    ],
    "🔬 STEM & Technology": [
        ('coding', 'Coding'), ('mathematics', 'Mathematics'), ('physics', 'Physics'),
        ('chemistry', 'Chemistry'), ('biology', 'Biology'), ('electricity_components', 'Electricity Components'),
        ('mechanic_parts', 'Mechanic Parts'), ('computer_parts', 'Computer Parts'),
        ('researching', 'Researching'), ('science', 'Science'), ('engineering', 'Engeeniering'),
        ('solving_puzzles', 'Solving Puzzles')
    ],
    "🏥 Medical & Life Sciences": [
        ('doctor', 'Doctor'), ('botany', 'Botany'), ('zoology', 'Zoology'),
        ('exercise', 'Exercise'), ('pharmacist', 'Pharmisist'), ('animals', 'Animals'),
        ('gardening', 'Gardening'), ('yoga', 'Yoga'), ('gymnastics', 'Gymnastics')
    ],
    "💼 Business & Social Sciences": [
        ('teaching', 'Teaching'), ('accounting', 'Accounting'), ('economics', 'Economics'),
        ('business', 'Bussiness'), ('business_education', 'Bussiness Education'),
        ('journalism', 'Journalism'), ('sociology', 'Sociology'), ('psychology', 'Psycology'),
        ('history', 'History'), ('geography', 'Geography'), ('debating', 'Debating')
    ],
    "🌍 Languages & Literature": [
        ('hindi', 'Hindi'), ('french', 'French'), ('english', 'English'),
        ('urdu', 'Urdu'), ('other_language', 'Other Language'), ('literature', 'Literature'),
        ('reading', 'Reading')
    ],
    "🎭 Other Activities": [
        ('sports', 'Sports'), ('video_game', 'Video Game'), ('travelling', 'Travelling'),
        ('cycling', 'Cycling'), ('knitting', 'Knitting'), ('astrology', 'Asrtology'),
        ('historic_collection', 'Historic Collection'), ('architecture', 'Architecture'),
        ('director', 'Director')
    ]
}

# --- Load and Train Model ---
@st.cache_data
def load_and_train():
    df = pd.read_csv("mlproject.csv")
    feature_cols = [col for col in df.columns if col not in [
        'Courses', 'Top Careers', 'Highest Position', 'Avg Salary', 'Social Respect'
    ]]
    target_col = 'Courses'
    
    X = df[feature_cols]
    y = df[target_col]
    course_info = df[['Courses', 'Top Careers', 'Highest Position', 'Avg Salary', 'Social Respect']].drop_duplicates()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)
    return model, feature_cols, course_info

# --- Initialize Session State ---
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 0
    if 'selected_interests' not in st.session_state:
        st.session_state.selected_interests = set()

# --- Pages ---
def welcome_page():
    st.title("🎓 Smart Career Recommender")
    st.write("Find the perfect course and career based on your interests and hobbies.")
    if st.button("Start Now →"):
        st.session_state.page += 1

def interest_selection_page():
    st.header("✨ Select Your Interests & Hobbies")
    
    for category, interests in CATEGORIES.items():
        st.subheader(category)
        cols = st.columns(3)
        
        for i, (key, label) in enumerate(interests):
            with cols[i % 3]:
                if st.checkbox(label, key=f"interest_{key}"):
                    st.session_state.selected_interests.add(label)  # Store the actual feature name
                else:
                    if label in st.session_state.selected_interests:
                        st.session_state.selected_interests.remove(label)

    if st.button("Show Recommendations →"):
        if len(st.session_state.selected_interests) < 2:
            st.warning("Please select at least 2 interests.")
        else:
            st.session_state.page += 1

def results_page():
    st.success("🎯 Your Ideal Course & Career Based on Interests")
    
    model, feature_cols, course_info = load_and_train()
    
    # Create proper input vector matching the model's expected features
    input_vector = [1 if feature in st.session_state.selected_interests else 0 for feature in feature_cols]
    
    # Make prediction
    best_course = model.predict([input_vector])[0]
    info = course_info[course_info['Courses'] == best_course].iloc[0]
    
    # Display results
    st.subheader(f"🌟 Best Match: {best_course}")
    st.markdown(f"**Top Careers:** {info['Top Careers']}")
    st.markdown(f"**Highest Position:** {info['Highest Position']}")
    st.markdown(f"**Avg Salary:** {info['Avg Salary']}")
    st.markdown(f"**Social Respect:** {info['Social Respect']}")

    st.divider()

    # Individual interest suggestions
    st.subheader("📌 Career Suggestions by Each Interest")
    for interest in st.session_state.selected_interests:
        one_hot = [1 if feature == interest else 0 for feature in feature_cols]
        predicted = model.predict([one_hot])[0]
        details = course_info[course_info['Courses'] == predicted].iloc[0]

        with st.expander(f"💡 {interest} → {predicted}"):
            st.markdown(f"**Top Careers:** {details['Top Careers']}")
            st.markdown(f"**Highest Position:** {details['Highest Position']}")
            st.markdown(f"**Avg Salary:** {details['Avg Salary']}")
            st.markdown(f"**Social Respect:** {details['Social Respect']}")

    if st.button("🔄 Restart"):
        st.session_state.page = 0
        st.session_state.selected_interests = set()

# --- Main App ---
def main():
    init_session_state()
    pages = [welcome_page, interest_selection_page, results_page]
    pages[st.session_state.page]()

if __name__ == "__main__":
    main()