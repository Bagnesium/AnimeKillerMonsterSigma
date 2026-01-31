import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import time

print("🚀 Starting optimized training...")
start_time = time.time()

# ───────────────────────────────────────────────
# CONFIG - EXPANDED PROFESSIONS
# ───────────────────────────────────────────────
skills_columns = [
    'Programming_AI',
    'Math_Science',
    'Leadership',
    'Creativity_Video',
    'Physical_Fitness',
    'Family_Responsibility'
]

# Expanded to 50+ professions with more nuanced skill requirements
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

print(f"Total professions: {len(PROFESSIONS)}")

# ───────────────────────────────────────────────
# SYNTHETIC STUDENTS - REDUCED BUT STILL EFFECTIVE
# ───────────────────────────────────────────────
np.random.seed(42)
N = 3000  # Reduced from 10000 - still gives 264,000 samples

students = pd.DataFrame({
    'SAT': np.random.normal(1200, 150, N).clip(800, 1600),
    'GPA': np.random.beta(8, 2, N) * 1.5 + 2.5,
})
students['GPA'] = students['GPA'].clip(2.0, 4.0)

# Generate skills efficiently
for s in skills_columns:
    students[s] = np.random.normal(5.5, 2.5, N).clip(1, 10).round().astype(int)

# Add skill correlations vectorized
high_prog = students['Programming_AI'] >= 8
students.loc[high_prog, 'Math_Science'] = np.minimum(10, 
    students.loc[high_prog, 'Math_Science'] + np.random.randint(0, 3, high_prog.sum()))

high_lead = students['Leadership'] >= 8
students.loc[high_lead, 'Family_Responsibility'] = np.minimum(10, 
    students.loc[high_lead, 'Family_Responsibility'] + 1)

print(f"Generated {N} student profiles")

# ───────────────────────────────────────────────
# BUILD TRAINING DATA - SIMPLIFIED LOOP (RELIABLE)
# ───────────────────────────────────────────────
print("Building training data...")

rows = []
count = 0
total = N * len(PROFESSIONS)

for idx, stu in students.iterrows():
    for prof in PROFESSIONS:
        req = req_skills.loc[prof]
        deltas = stu[skills_columns] - req
        deltas_vals = deltas.values
        
        # Enhanced scoring system
        negative_deltas = np.clip(-deltas_vals, 0, None)
        skill_penalty = np.sum(negative_deltas ** 1.3) * 0.08
        
        positive_deltas = np.clip(deltas_vals, 0, None)
        skill_reward = np.sum(np.log1p(positive_deltas)) * 0.15
        
        # Critical skills (top 2 required skills for profession)
        top_req_indices = np.argsort(req.values)[-2:]
        critical_gaps = -deltas_vals[top_req_indices]
        critical_penalty = np.sum(np.clip(critical_gaps, 0, None) ** 1.5) * 0.12
        
        skill_variance = np.var(stu[skills_columns].values) * 0.02
        skill_score = np.clip(0.65 + skill_reward - skill_penalty - critical_penalty + skill_variance, 0, 1)
        
        # Academic factors
        gpa_score = np.clip((stu['GPA'] - 2.0) / 2.0, 0, 1)
        sat_score = np.clip((stu['SAT'] - 800) / 800, 0, 1)
        
        # Different professions weight academics differently
        if 'Engineer' in prof or 'Scientist' in prof or 'Doctor' in prof:
            academic_weight = 0.25
        elif 'Artist' in prof or 'Musician' in prof or 'Athlete' in prof:
            academic_weight = 0.10
        else:
            academic_weight = 0.20
        
        # Final match score
        match = (
            (1 - academic_weight) * skill_score +
            academic_weight * 0.6 * gpa_score +
            academic_weight * 0.4 * sat_score
        )
        
        row = {
            'profession': prof,
            'SAT': stu['SAT'],
            'GPA': stu['GPA'],
            'match_score': match
        }
        
        # Add delta features
        for i, s in enumerate(skills_columns):
            row[f'delta_{s}'] = deltas_vals[i]
        
        # Add additional features
        row['skill_penalty'] = skill_penalty
        row['skill_reward'] = skill_reward
        row['critical_penalty'] = critical_penalty
        row['total_skills'] = stu[skills_columns].sum()
        row['avg_skill'] = stu[skills_columns].mean()
        
        rows.append(row)
        
        count += 1
        if count % 50000 == 0:
            print(f"  Progress: {count:,}/{total:,} ({count/total*100:.1f}%)")

df = pd.DataFrame(rows)

print(f"Training samples: {len(df):,}")
print(f"Match score stats: mean={df['match_score'].mean():.3f}, std={df['match_score'].std():.3f}")

# ───────────────────────────────────────────────
# TRAIN MODEL
# ───────────────────────────────────────────────
feature_cols = ['SAT', 'GPA'] + \
               [f'delta_{s}' for s in skills_columns] + \
               ['skill_penalty', 'skill_reward', 'critical_penalty', 'total_skills', 'avg_skill']

X = df[feature_cols]
y = df['match_score']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=pd.qcut(y, q=5, duplicates='drop')
)

print("Scaling features...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Faster model with fewer estimators but still accurate
print("\n🔄 Training model...")
model = GradientBoostingRegressor(
    n_estimators=200,  # Reduced from 500
    max_depth=4,       # Reduced from 5
    learning_rate=0.05,  # Increased for faster convergence
    min_samples_split=30,
    min_samples_leaf=15,
    subsample=0.8,
    random_state=42,
    verbose=0
)

model.fit(X_train_s, y_train)

# Evaluate
train_score = model.score(X_train_s, y_train)
test_score = model.score(X_test_s, y_test)
y_pred = model.predict(X_test_s)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"Train R²: {train_score:.4f}")
print(f"Test R²: {test_score:.4f}")
print(f"MAE: {mae:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🎯 Top 10 Most Important Features:")
print(feature_importance.head(10))

# Save artifacts
print("\nSaving model...")
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(feature_cols, "feature_cols.joblib")

elapsed = time.time() - start_time
print(f"\n✅ Model trained and saved successfully!")
print(f"⏱️  Total time: {elapsed:.1f} seconds")
print(f"📊 Professions: {len(PROFESSIONS)}")
print(f"📊 Features: {len(feature_cols)}")