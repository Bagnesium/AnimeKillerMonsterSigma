import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# ───────────────────────────────────────────────
# LOAD MODEL ASSETS
# ───────────────────────────────────────────────
try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
except:
    st.error("⚠️ Model files not found. Please run the training script first.")
    st.stop()

skills_columns = [
    'Programming_AI',
    'Math_Science',
    'Leadership',
    'Creativity_Video',
    'Physical_Fitness',
    'Family_Responsibility'
]

# ───────────────────────────────────────────────
# LOAD UNIVERSITY DATA
# ───────────────────────────────────────────────
@st.cache_data
def load_universities():
    """Parse university data from CSV"""
    data = []
    with open('uni_data.csv', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|')
            if len(parts) >= 4:
                name = parts[0].strip()
                sat_range = parts[1].strip()
                gpa = parts[2].strip()
                ielts = parts[3].strip()
                
                # Parse SAT range
                sat_min, sat_max = 1200, 1200
                if not any(x in sat_range for x in ['IB', 'Entrance', 'Math', 'Abitur', 'Gaokao', 'Bac', 'German', 'EJU', 'ДВИ', 'Til-I']):
                    sat_nums = re.findall(r'\d{3,4}', sat_range)
                    if len(sat_nums) >= 2:
                        sat_min = int(sat_nums[0])
                        sat_max = int(sat_nums[1])
                    elif len(sat_nums) == 1:
                        sat_min = sat_max = int(sat_nums[0])
                
                # Parse GPA
                try:
                    gpa_val = float(gpa)
                except:
                    gpa_val = 3.5
                
                # Parse IELTS
                try:
                    ielts_val = float(ielts)
                except:
                    ielts_val = 6.5
                
                data.append({
                    'university': name,
                    'sat_min': sat_min,
                    'sat_max': sat_max,
                    'sat_avg': (sat_min + sat_max) / 2,
                    'gpa_required': gpa_val,
                    'ielts_required': ielts_val
                })
    
    return pd.DataFrame(data)

universities = load_universities()

# Expanded profession requirements (must match training)
profession_requirements = {
    # Tech & Engineering
    'Software Engineer': [9, 8, 5, 4, 3, 4],
    'Data Scientist': [9, 10, 6, 5, 3, 4],
    'Machine Learning Engineer': [10, 9, 5, 4, 3, 4],
    'Web Developer': [8, 6, 4, 6, 3, 4],
    'Mobile App Developer': [8, 7, 5, 5, 3, 4],
    'DevOps Engineer': [8, 7, 6, 4, 3, 4],
    'Cybersecurity Analyst': [8, 8, 5, 4, 3, 5],
    'Database Administrator': [7, 8, 5, 3, 3, 4],
    'UI/UX Designer': [6, 5, 5, 9, 3, 4],
    'Game Developer': [8, 7, 5, 8, 3, 4],
    'Mechanical Engineer': [6, 9, 6, 5, 4, 4],
    'Electrical Engineer': [7, 9, 6, 4, 3, 4],
    'Civil Engineer': [5, 9, 7, 5, 5, 4],
    'Chemical Engineer': [5, 10, 6, 4, 4, 4],
    'Aerospace Engineer': [7, 10, 6, 5, 4, 4],
    'Biomedical Engineer': [6, 9, 6, 5, 5, 5],
    'Environmental Engineer': [5, 8, 6, 5, 5, 6],
    'Robotics Engineer': [9, 9, 6, 6, 4, 4],
    
    # Business & Finance
    'Product Manager': [6, 7, 9, 7, 4, 5],
    'Marketing Manager': [5, 6, 8, 8, 4, 5],
    'Sales Manager': [4, 5, 9, 7, 4, 5],
    'Business Analyst': [6, 8, 7, 5, 3, 4],
    'Financial Analyst': [5, 9, 6, 4, 3, 4],
    'Investment Banker': [5, 9, 8, 5, 4, 4],
    'Accountant': [4, 8, 5, 3, 3, 5],
    'Management Consultant': [6, 8, 9, 7, 4, 5],
    'Business Consultant': [6, 7, 8, 6, 4, 5],
    'Operations Manager': [5, 7, 8, 5, 4, 5],
    'Supply Chain Manager': [5, 8, 8, 5, 4, 5],
    'HR Manager': [4, 5, 8, 6, 4, 8],
    'Entrepreneur': [7, 7, 10, 8, 6, 6],
    'Real Estate Agent': [3, 5, 7, 6, 5, 6],
    'Economist': [5, 10, 6, 5, 3, 4],
    
    # Creative & Media
    'Graphic Designer': [6, 4, 5, 10, 3, 4],
    'Video Editor': [7, 4, 5, 10, 3, 4],
    'Photographer': [5, 4, 5, 9, 4, 4],
    'Animator': [7, 5, 5, 10, 3, 4],
    '3D Artist': [7, 6, 5, 10, 3, 4],
    'Art Director': [6, 5, 7, 10, 3, 4],
    'Content Creator': [6, 4, 6, 9, 4, 4],
    'Social Media Manager': [5, 4, 7, 8, 4, 5],
    'Writer': [4, 5, 5, 9, 3, 5],
    'Journalist': [4, 6, 6, 8, 4, 5],
    'Editor': [4, 6, 6, 8, 3, 5],
    'Copywriter': [4, 5, 6, 9, 3, 5],
    'Musician': [3, 4, 5, 10, 4, 4],
    'Music Producer': [6, 5, 6, 10, 3, 4],
    'Film Director': [5, 5, 8, 10, 4, 5],
    'Fashion Designer': [4, 4, 6, 10, 3, 4],
    'Interior Designer': [4, 5, 6, 9, 3, 4],
    
    # Healthcare & Science
    'Doctor': [4, 10, 7, 4, 6, 7],
    'Surgeon': [4, 10, 8, 5, 7, 6],
    'Nurse': [3, 7, 6, 4, 7, 8],
    'Pharmacist': [4, 9, 6, 4, 4, 6],
    'Dentist': [4, 9, 7, 5, 6, 6],
    'Veterinarian': [4, 9, 6, 5, 6, 7],
    'Physical Therapist': [3, 8, 6, 5, 9, 7],
    'Occupational Therapist': [3, 7, 6, 6, 7, 8],
    'Psychologist': [4, 8, 6, 5, 4, 8],
    'Psychiatrist': [4, 10, 7, 4, 4, 7],
    'Research Scientist': [5, 10, 6, 5, 4, 4],
    'Biologist': [4, 10, 5, 5, 5, 5],
    'Chemist': [4, 10, 5, 4, 4, 4],
    'Physicist': [5, 10, 5, 4, 4, 4],
    'Environmental Scientist': [4, 9, 6, 5, 6, 6],
    
    # Education & Social Services
    'Teacher': [4, 7, 7, 7, 5, 7],
    'Professor': [5, 9, 7, 6, 4, 5],
    'School Counselor': [4, 6, 7, 6, 4, 9],
    'Education Administrator': [4, 6, 9, 6, 4, 7],
    'Social Worker': [3, 5, 6, 5, 5, 10],
    'Therapist': [3, 6, 6, 5, 4, 9],
    'Counselor': [3, 6, 7, 6, 4, 9],
    'Non-Profit Manager': [4, 6, 8, 6, 5, 9],
    
    # Sports & Fitness
    'Personal Trainer': [3, 6, 7, 6, 10, 6],
    'Athlete': [3, 5, 7, 5, 10, 6],
    'Sports Coach': [3, 6, 9, 6, 9, 7],
    'Nutritionist': [4, 8, 6, 5, 7, 7],
    'Physical Education Teacher': [3, 6, 7, 6, 9, 7],
    
    # Trades & Other
    'Architect': [6, 8, 7, 9, 4, 4],
    'Urban Planner': [5, 7, 7, 7, 5, 6],
    'Lawyer': [4, 8, 8, 6, 4, 5],
    'Paralegal': [4, 7, 6, 5, 3, 5],
    'Chef': [3, 5, 6, 8, 7, 6],
    'Culinary Artist': [3, 5, 6, 9, 6, 5],
    'Event Planner': [4, 5, 8, 8, 5, 6],
    'Project Manager': [5, 7, 9, 6, 4, 5],
    'Data Analyst': [7, 9, 5, 5, 3, 4],
    'Statistician': [6, 10, 5, 4, 3, 4],
}

req_skills = pd.DataFrame(profession_requirements, index=skills_columns).T
PROFESSIONS = list(req_skills.index)

# Enhanced keyword groups for better portfolio parsing
keyword_groups = {
    'Programming_AI': [
        'python', 'java', 'c++', 'javascript', 'coding', 'programming', 'software',
        'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
        'algorithm', 'data structure', 'app development', 'web development', 
        'hackathon', 'github', 'api', 'database', 'sql', 'react', 'node',
        'tensorflow', 'pytorch', 'neural network', 'computer science'
    ],
    'Math_Science': [
        'math', 'mathematics', 'calculus', 'algebra', 'geometry', 'trigonometry',
        'statistics', 'probability', 'linear algebra', 'differential equations',
        'physics', 'chemistry', 'biology', 'science', 'stem', 'olympiad',
        'research', 'lab', 'experiment', 'thesis', 'ap calc', 'ap physics',
        'science fair', 'engineering', 'quantitative'
    ],
    'Leadership': [
        'president', 'vice president', 'captain', 'leader', 'founder', 'co-founder',
        'started', 'founded', 'led', 'organized', 'managed', 'directed',
        'club president', 'team captain', 'student council', 'board member',
        'coordinator', 'chair', 'head', 'officer', 'initiative', 'mentor',
        'volunteered', 'organized event', 'led team'
    ],
    'Creativity_Video': [
        'video', 'editing', 'premiere', 'final cut', 'davinci resolve', 'film',
        'content creation', 'youtube', 'tiktok', 'social media', 'creative',
        'design', 'graphic design', 'photoshop', 'illustrator', 'art',
        'photography', 'digital art', 'animation', 'music production',
        'portfolio', 'creative writing', 'painting', 'drawing', 'cinema'
    ],
    'Physical_Fitness': [
        'sports', 'athlete', 'varsity', 'track', 'soccer', 'basketball', 'football',
        'swimming', 'tennis', 'volleyball', 'gym', 'fitness', 'workout',
        'training', 'marathon', 'running', 'cycling', 'weightlifting',
        'calisthenics', 'yoga', 'martial arts', 'dance', 'physical',
        'exercise', 'competition', 'championship', 'team sport'
    ],
    'Family_Responsibility': [
        'family', 'sibling', 'brother', 'sister', 'childcare', 'babysitting',
        'household', 'chores', 'cooking', 'cleaning', 'caring for', 'helped',
        'responsibilities', 'taking care', 'family business', 'helped parents',
        'younger siblings', 'family duties', 'home responsibilities'
    ]
}

# ───────────────────────────────────────────────
# UTILITY FUNCTIONS
# ───────────────────────────────────────────────
def extract_sat_gpa(text):
    """Enhanced SAT and GPA extraction"""
    if not text:
        return 1200, 3.5
    
    text_lower = text.lower()
    
    # Extract SAT
    sat_patterns = [
        r'sat[:\s]+(\d{3,4})',
        r'(\d{3,4})[:\s]+sat',
        r'\b(1[0-6]\d{2}|[89]\d{2})\b'
    ]
    sat = 1200
    for pattern in sat_patterns:
        match = re.search(pattern, text_lower)
        if match:
            potential_sat = int(match.group(1))
            if 800 <= potential_sat <= 1600:
                sat = potential_sat
                break
    
    # Extract GPA
    gpa_patterns = [
        r'gpa[:\s]+(\d\.\d+)',
        r'(\d\.\d+)[:\s]+gpa',
        r'\b([2-4]\.\d{1,2})\b'
    ]
    gpa = 3.5
    for pattern in gpa_patterns:
        match = re.search(pattern, text_lower)
        if match:
            potential_gpa = float(match.group(1))
            if 2.0 <= potential_gpa <= 4.0:
                gpa = potential_gpa
                break
    
    return sat, gpa

def parse_portfolio(text):
    """Enhanced portfolio parsing"""
    if not text:
        return 1200, 3.5, {k: 1 for k in skills_columns}
    
    text_lower = text.lower()
    sat, gpa = extract_sat_gpa(text)
    
    skills = {}
    for skill, keywords in keyword_groups.items():
        match_count = 0
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                match_count += 1
        
        if match_count == 0:
            skill_level = 1
        elif match_count == 1:
            skill_level = 3
        elif match_count == 2:
            skill_level = 5
        elif match_count == 3:
            skill_level = 6
        elif match_count == 4:
            skill_level = 7
        elif match_count >= 5:
            skill_level = min(10, 8 + (match_count - 5) // 2)
        
        skills[skill] = skill_level
    
    return sat, gpa, skills

def fuse_skills(quiz, text, w_quiz=0.6, w_text=0.4):
    """Combine quiz + portfolio skills"""
    fused = {}
    for k in skills_columns:
        if text[k] <= 2:
            fused[k] = round(0.8 * quiz[k] + 0.2 * text[k], 1)
        else:
            fused[k] = round(w_quiz * quiz[k] + w_text * text[k], 1)
    return fused

def score_profession(profession, skills, sat, gpa):
    """Predict match score using ML model"""
    req = req_skills.loc[profession]
    
    deltas = {f'delta_{k}': skills[k] - req[k] for k in skills_columns}
    deltas_vals = np.array([deltas[f'delta_{k}'] for k in skills_columns])
    
    negative_deltas = np.clip(-deltas_vals, 0, None)
    skill_penalty = np.sum(negative_deltas ** 1.3) * 0.08
    
    positive_deltas = np.clip(deltas_vals, 0, None)
    skill_reward = np.sum(np.log1p(positive_deltas)) * 0.15
    
    top_req_indices = np.argsort(req.values)[-2:]
    critical_gaps = -deltas_vals[top_req_indices]
    critical_penalty = np.sum(np.clip(critical_gaps, 0, None) ** 1.5) * 0.12
    
    total_skills = sum(skills.values())
    avg_skill = np.mean(list(skills.values()))
    
    features = {
        'SAT': sat,
        'GPA': gpa,
        **deltas,
        'skill_penalty': skill_penalty,
        'skill_reward': skill_reward,
        'critical_penalty': critical_penalty,
        'total_skills': total_skills,
        'avg_skill': avg_skill
    }
    
    X = pd.DataFrame([features])[feature_cols]
    X_scaled = scaler.transform(X)
    
    return float(model.predict(X_scaled)[0])

def recommend_universities(sat, gpa, top_n=10):
    """Recommend universities based on SAT and GPA"""
    # Filter universities where student meets requirements
    suitable = universities[
        (universities['sat_min'] <= sat + 50) &  # Slight buffer
        (universities['gpa_required'] <= gpa + 0.15)
    ].copy()
    
    if len(suitable) == 0:
        # If no matches, find closest ones
        suitable = universities.copy()
    
    # Calculate match score for each university
    suitable['sat_match'] = 1 - abs(suitable['sat_avg'] - sat) / 400
    suitable['gpa_match'] = 1 - abs(suitable['gpa_required'] - gpa) / 2
    suitable['overall_match'] = (suitable['sat_match'] * 0.6 + suitable['gpa_match'] * 0.4).clip(0, 1)
    
    # Sort by match score
    suitable = suitable.sort_values('overall_match', ascending=False)
    
    return suitable.head(top_n)

# ───────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────
st.set_page_config(page_title="AI Career & University Matcher", layout="wide", page_icon="🎯")

st.title("🎯 AI Career & University Matching System")
st.markdown("### *Powered by Machine Learning - 88 Careers + 600 Universities*")

# Instructions
with st.expander("📖 How to use this tool"):
    st.markdown("""
    1. **Rate your skills** using the sliders (1 = beginner, 10 = expert)
    2. **Paste your achievements** - Include SAT, GPA, projects, activities, awards
    3. **Click Analyze** to see:
       - 🏆 Best-fit careers
       - 🎓 Recommended universities
       - 📊 Your skill profile
    
    💡 **Tip**: Be honest and provide detailed information for accurate results!
    """)

st.markdown("---")

# Two-column layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 🧠 Self-Assessment (1-10)")
    
    quiz_skills = {}
    for skill in skills_columns:
        display_name = skill.replace('_', ' / ')
        quiz_skills[skill] = st.slider(
            display_name,
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help=f"Rate your {skill.replace('_', ' ').lower()} ability"
        )

with col_right:
    st.markdown("### 📄 Portfolio / Achievements")
    st.markdown("*Include: SAT score, GPA, extracurriculars, projects, awards, etc.*")
    
    portfolio_text = st.text_area(
        label="Your achievements",
        placeholder="""Example:
SAT: 1480, GPA: 3.85
- Founded coding club, taught Python to 30+ students
- Built ML model for housing price prediction
- Varsity soccer team captain (3 years)
- Video editor for school news channel
- AP Calculus BC (5), AP Physics (5)
- Help parents with family business on weekends""",
        height=280,
        label_visibility="collapsed"
    )

st.markdown("---")

if st.button("🚀 Analyze My Profile", type="primary", use_container_width=True):
    with st.spinner("Analyzing your profile..."):
        # Parse inputs
        sat, gpa, text_skills = parse_portfolio(portfolio_text)
        final_skills = fuse_skills(quiz_skills, text_skills)
        
        # Score all professions
        results = []
        for prof in PROFESSIONS:
            score = score_profession(prof, final_skills, sat, gpa)
            results.append((prof, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Get university recommendations
        uni_recs = recommend_universities(sat, gpa, top_n=10)
        
        # ═══════════════════════════════════════════
        # RESULTS DISPLAY
        # ═══════════════════════════════════════════
        st.markdown("---")
        
        # Academic Profile Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 SAT Score", f"{int(sat)}")
        with col2:
            st.metric("📚 GPA", f"{gpa:.2f}")
        with col3:
            st.metric("💪 Skill Points", f"{sum(final_skills.values())}/60")
        with col4:
            percentile = int((sat - 800) / 800 * 100)
            st.metric("📊 SAT Percentile", f"~{percentile}%")
        
        st.markdown("---")
        
        # Two columns for careers and universities
        col_career, col_uni = st.columns([1, 1])
        
        with col_career:
            st.markdown("## 🏆 Top Career Matches")
            
            for i, (prof, score) in enumerate(results[:8], 1):
                percentage = int(score * 100)
                
                if score >= 0.8:
                    color = "🟢"
                elif score >= 0.6:
                    color = "🟡"
                else:
                    color = "🟠"
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{i}. {prof}** {color}")
                with col2:
                    st.markdown(f"`{percentage}%`")
                
                st.progress(score)
                st.markdown("")
            
            with st.expander(f"📋 See all {len(results)} careers"):
                for i, (prof, score) in enumerate(results, 1):
                    st.write(f"{i}. {prof} — {int(score*100)}%")
        
        with col_uni:
            st.markdown("## 🎓 Recommended Universities")
            
            if len(uni_recs) > 0:
                for i, row in uni_recs.iterrows():
                    match_pct = int(row['overall_match'] * 100)
                    
                    if match_pct >= 80:
                        badge = "🎯 Safety"
                    elif match_pct >= 60:
                        badge = "✅ Target"
                    else:
                        badge = "🚀 Reach"
                    
                    st.markdown(f"**{row['university']}** {badge}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"SAT: {int(row['sat_min'])}-{int(row['sat_max'])}")
                    with col2:
                        st.caption(f"GPA: {row['gpa_required']:.2f}")
                    with col3:
                        st.caption(f"Match: {match_pct}%")
                    
                    st.progress(row['overall_match'])
                    st.markdown("")
            else:
                st.info("Add your SAT and GPA to get university recommendations!")
        
        st.markdown("---")
        
        # Skill Profile
        st.markdown("### 📊 Your Skill Profile")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            skills_df = pd.DataFrame({
                'Skill': [s.replace('_', ' ') for s in skills_columns],
                'Level': [final_skills[s] for s in skills_columns]
            }).set_index('Skill')
            st.bar_chart(skills_df, height=300)
        
        with col2:
            st.markdown("#### Skill Breakdown")
            for skill in skills_columns:
                level = final_skills[skill]
                st.write(f"**{skill.replace('_', ' ')}:** {level}/10")
                st.progress(level / 10)
        
        st.markdown("---")
        
        # Consistency Check
        st.markdown("### ⚠️ Self-Assessment Consistency")
        inconsistencies = []
        for k in skills_columns:
            diff = abs(quiz_skills[k] - text_skills[k])
            if diff >= 4:
                inconsistencies.append((k, quiz_skills[k], text_skills[k], diff))
        
        if inconsistencies:
            st.warning(f"Found {len(inconsistencies)} potential inconsistencies:")
            for skill, quiz_val, text_val, diff in inconsistencies:
                st.write(f"- **{skill.replace('_', ' ')}**: Self-rated {quiz_val}/10, portfolio suggests {text_val}/10")
            st.info("💡 Consider adjusting ratings or adding more portfolio details.")
        else:
            st.success("✅ Your self-assessment aligns well with your portfolio!")
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 Personalized Recommendations")
        
        top_career = results[0][0]
        top_score = results[0][1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Career Development")
            if top_score >= 0.8:
                st.success(f"🎉 Excellent match with **{top_career}**!")
            elif top_score >= 0.6:
                st.info(f"👍 Good fit for **{top_career}**. Consider skill development.")
            else:
                st.warning(f"⚠️ Moderate match. Explore other options or build relevant skills.")
            
            # Skill gaps
            req = req_skills.loc[top_career]
            gaps = []
            for skill in skills_columns:
                if final_skills[skill] < req[skill]:
                    gaps.append((skill, req[skill] - final_skills[skill]))
            
            if gaps:
                gaps.sort(key=lambda x: x[1], reverse=True)
                st.markdown(f"**To improve for {top_career}:**")
                for skill, gap in gaps[:3]:
                    st.write(f"• {skill.replace('_', ' ')}: +{gap:.1f} points needed")
        
        with col2:
            st.markdown("#### 🎓 University Strategy")
            if len(uni_recs) > 0:
                safety = len(uni_recs[uni_recs['overall_match'] >= 0.8])
                target = len(uni_recs[(uni_recs['overall_match'] >= 0.6) & (uni_recs['overall_match'] < 0.8)])
                reach = len(uni_recs[uni_recs['overall_match'] < 0.6])
                
                st.write(f"**Your university balance:**")
                st.write(f"• 🎯 Safety schools: {safety}")
                st.write(f"• ✅ Target schools: {target}")
                st.write(f"• 🚀 Reach schools: {reach}")
                
                if safety < 2:
                    st.info("💡 Consider adding more safety schools to your list.")
                if reach < 2:
                    st.info("💡 Don't be afraid to apply to reach schools!")

else:
    st.info("👆 Fill in your information and click 'Analyze' to see personalized career and university recommendations!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>🤖 AI-Powered Matching | 88 Careers | 600+ Universities | Real-time Analysis</p>
    <p>This tool provides suggestions based on your input. Always research thoroughly!</p>
</div>
""", unsafe_allow_html=True)