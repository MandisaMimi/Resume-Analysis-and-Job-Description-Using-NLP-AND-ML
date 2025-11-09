# app.py - Complete AI Resume Analyzer with Enhanced Job Matching (SHAP-Free Version)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import warnings
import os
import PyPDF2
import docx
from datetime import datetime
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Try to import LIME only (SHAP removed)
try:
    import lime
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    st.sidebar.warning("⚠️ LIME not installed. Install with: pip install lime")

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer & Career Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with BLUE color scheme (orange removed)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2563EB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        background: linear-gradient(45deg, #2563EB, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E40AF;
        margin-bottom: 1.5rem;
        font-weight: 700;
        border-left: 5px solid #2563EB;
        padding-left: 15px;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 12px 30px rgba(37,99,235,0.3);
        text-align: center;
    }
    .skill-card {
        background: linear-gradient(135deg, #00A8E8 0%, #0077B6 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(0,168,232,0.2);
        margin: 0.5rem;
        border: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .skill-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 25px rgba(0,168,232,0.4);
    }
    .warning-card {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        border: none;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 6px 15px rgba(245,158,11,0.3);
        color: white;
    }
    .success-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        border: none;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 6px 15px rgba(16,185,129,0.3);
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(124,58,237,0.2);
        text-align: center;
        border: none;
    }
    .upload-section {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #2563EB;
        margin: 1rem 0;
        text-align: center;
    }
    .stButton button {
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37,99,235,0.4);
    }
    .model-card {
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(79,70,229,0.3);
    }
    .xai-card {
        background: linear-gradient(135deg, #06D6A0 0%, #04A777 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 6px 15px rgba(6,214,160,0.3);
    }
    .match-card {
        background: linear-gradient(135deg, #1E40AF 0%, #1E3A8A 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(30,64,175,0.3);
    }
    .feature-card-positive {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 6px 15px rgba(16,185,129,0.3);
    }
    .feature-card-negative {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 6px 15px rgba(239,68,68,0.3);
    }
    .lime-card {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 6px 15px rgba(139,92,246,0.3);
    }
</style>
""", unsafe_allow_html=True)

class ResumeAnalyzer:
    def __init__(self, vectorizer, label_encoder, models):
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.models = models
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize XAI explainers (SHAP removed)
        self.lime_explainer = None
        self._initialize_xai()
        
        # Predefined skill patterns
        self.skill_patterns = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust',
                'typescript', 'html', 'css', 'sql', 'r', 'matlab', 'scala', 'perl', 'bash', 'shell'
            ],
            'web_frameworks': [
                'django', 'flask', 'spring', 'express', 'react', 'angular', 'vue', 'node.js', 'laravel', 'rails',
                'asp.net', 'jquery', 'bootstrap', 'tailwind'
            ],
            'data_science': [
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'spark', 'hadoop', 'hive',
                'tableau', 'power bi', 'excel', 'statistics', 'machine learning', 'deep learning', 'nlp',
                'computer vision', 'data analysis', 'data visualization'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'terraform', 'ansible',
                'git', 'github', 'gitlab', 'linux', 'unix'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite', 'cassandra', 'dynamodb'
            ],
            'soft_skills': [
                'communication', 'leadership', 'teamwork', 'problem solving', 'critical thinking',
                'project management', 'agile', 'scrum', 'time management', 'creativity'
            ]
        }
    
    def _initialize_xai(self):
        """Initialize XAI explainers (SHAP removed)"""
        # Initialize LIME only
        if LIME_AVAILABLE:
            try:
                self.lime_explainer = lime.lime_text.LimeTextExplainer(
                    class_names=self.label_encoder.classes_
                )
                st.sidebar.success("✅ LIME explainer initialized")
            except Exception as e:
                st.sidebar.warning(f"⚠️ LIME initialization failed: {e}")
    
    def clean_resume_text(self, text):
        """Enhanced text cleaning function"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'[^a-zA-Z\s\+\#\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        
        return text.strip()
    
    def extract_skills(self, text):
        """Extract skills from text using patterns"""
        skills_found = {category: [] for category in self.skill_patterns.keys()}
        text_lower = text.lower()
        
        for category, skill_list in self.skill_patterns.items():
            for skill in skill_list:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    skills_found[category].append(skill)
        
        return skills_found
    
    def analyze_resume_structure(self, text):
        """Analyze resume structure and quality"""
        analysis = {}
        
        # Word count analysis
        words = text.split()
        analysis['word_count'] = len(words)
        analysis['character_count'] = len(text)
        
        # Section detection
        sections = {
            'experience': len(re.findall(r'\b(experience|work\s*history|employment)\b', text, re.I)),
            'education': len(re.findall(r'\b(education|academic|degree)\b', text, re.I)),
            'skills': len(re.findall(r'\b(skills|technical\s*skills|competencies)\b', text, re.I)),
            'projects': len(re.findall(r'\b(projects|portfolio|work\s*samples)\b', text, re.I))
        }
        
        analysis['sections_found'] = {k: v > 0 for k, v in sections.items()}
        analysis['missing_sections'] = [k for k, v in analysis['sections_found'].items() if not v]
        
        # Quality indicators
        analysis['has_quantifiable_achievements'] = bool(re.search(r'\d+', text))
        analysis['has_action_verbs'] = bool(re.search(r'\b(developed|managed|implemented|created|led|improved)\b', text, re.I))
        
        return analysis
    
    def predict_with_xai(self, text):
        """Make prediction with enhanced XAI features (SHAP removed)"""
        if not text or len(text.strip()) < 50:
            return {'error': 'Text too short (minimum 50 characters required)'}
        
        # Clean and vectorize text
        cleaned_text = self.clean_resume_text(text)
        features = self.vectorizer.transform([cleaned_text])
        
        # Get predictions from all models
        model_predictions = {}
        model_probabilities = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)[0]
                    pred = model.predict(features)[0]
                    
                    model_predictions[name] = self.label_encoder.inverse_transform([pred])[0]
                    model_probabilities[name] = proba
                else:
                    pred = model.predict(features)[0]
                    model_predictions[name] = self.label_encoder.inverse_transform([pred])[0]
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
                continue
        
        # Ensemble prediction
        if model_probabilities:
            avg_prob = np.mean(list(model_probabilities.values()), axis=0)
            final_pred_idx = np.argmax(avg_prob)
            final_pred = self.label_encoder.inverse_transform([final_pred_idx])[0]
            confidence = float(avg_prob[final_pred_idx])
            
            # Get top 3 categories
            top_3_indices = np.argsort(avg_prob)[-3:][::-1]
            top_categories = []
            for idx in top_3_indices:
                category = self.label_encoder.inverse_transform([idx])[0]
                prob = float(avg_prob[idx])
                top_categories.append({'category': category, 'confidence': prob})
        else:
            final_pred = max(set(model_predictions.values()), key=list(model_predictions.values()).count)
            confidence = 0.5
            top_categories = [{'category': final_pred, 'confidence': confidence}]
        
        # Enhanced XAI Analysis (SHAP removed)
        lime_analysis = self._analyze_with_lime(cleaned_text, final_pred_idx) if LIME_AVAILABLE else {'available': False}
        
        # Extract skills and structure
        skills = self.extract_skills(text)
        structure = self.analyze_resume_structure(text)
        
        # Enhanced feature importance analysis (replaces SHAP)
        feature_importance = self._get_enhanced_feature_importance(cleaned_text, final_pred_idx)
        
        return {
            'predicted_category': final_pred,
            'confidence': confidence,
            'model_consensus': model_predictions,
            'top_categories': top_categories,
            'skills': skills,
            'resume_structure': structure,
            'xai_analysis': {
                'lime': lime_analysis,
                'feature_importance': feature_importance,
                'model_insights': self._get_model_insights(model_probabilities, final_pred_idx)
            },
            'word_count': len(cleaned_text.split()),
            'processing_details': {
                'models_used': len(model_predictions),
                'ensemble_method': 'weighted_probability_average',
                'xai_methods': ['LIME', 'Enhanced_Feature_Analysis', 'Model_Consensus']
            }
        }
    
    def _get_enhanced_feature_importance(self, text, predicted_class_idx):
        """Enhanced feature importance analysis to replace SHAP"""
        try:
            features = self.vectorizer.transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = features.toarray()[0]
            
            # Get feature importance from ensemble models
            ensemble_importance = np.zeros(len(feature_names))
            
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        ensemble_importance += importance
                elif hasattr(model, 'coef_'):
                    # Linear models
                    coef = model.coef_[predicted_class_idx]
                    ensemble_importance += np.abs(coef)
            
            # Normalize importance scores
            if np.sum(ensemble_importance) > 0:
                ensemble_importance = ensemble_importance / np.sum(ensemble_importance)
            
            # Combine TF-IDF scores with model importance
            combined_scores = tfidf_scores * (1 + ensemble_importance)
            
            feature_importance = list(zip(feature_names, combined_scores, tfidf_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Separate positive and impactful features
            top_positive = [(f, score, tfidf) for f, score, tfidf in feature_importance if tfidf > 0][:10]
            top_negative = [(f, score, tfidf) for f, score, tfidf in feature_importance if tfidf == 0][:5]
            
            return {
                'available': True,
                'top_positive_features': [(f, score) for f, score, tfidf in top_positive],
                'top_missing_features': [(f, score) for f, score, tfidf in top_negative],
                'feature_breakdown': {
                    'features': [f[0] for f in feature_importance[:15]],
                    'scores': [float(f[1]) for f in feature_importance[:15]],
                    'tfidf_scores': [float(f[2]) for f in feature_importance[:15]]
                },
                'document_metrics': {
                    'total_features': len([f for f in tfidf_scores if f > 0]),
                    'unique_terms': len(set(text.split())),
                    'feature_density': len([f for f in tfidf_scores if f > 0]) / len(feature_names)
                }
            }
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _get_model_insights(self, model_probabilities, predicted_class_idx):
        """Get insights from model probabilities"""
        try:
            if not model_probabilities:
                return {'available': False}
            
            # Calculate model agreement
            predictions = list(model_probabilities.keys())
            agreement_scores = {}
            
            for model_name, proba in model_probabilities.items():
                confidence = proba[predicted_class_idx]
                agreement_scores[model_name] = {
                    'confidence': float(confidence),
                    'contribution': float(proba[predicted_class_idx] / len(model_probabilities))
                }
            
            return {
                'available': True,
                'model_agreement': agreement_scores,
                'consensus_level': np.mean([score['confidence'] for score in agreement_scores.values()]),
                'strongest_model': max(agreement_scores.items(), key=lambda x: x[1]['confidence'])[0]
            }
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _analyze_with_lime(self, text, predicted_class_idx):
        """Analyze prediction using LIME"""
        try:
            def predict_proba(texts):
                features = self.vectorizer.transform(texts)
                all_probas = []
                for model in self.models.values():
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features)
                        all_probas.append(proba)
                
                if all_probas:
                    return np.mean(all_probas, axis=0)
                else:
                    return list(self.models.values())[0].predict_proba(features)
            
            exp = self.lime_explainer.explain_instance(
                text, 
                predict_proba, 
                num_features=15, 
                top_labels=3
            )
            
            explanation = exp.as_list(label=predicted_class_idx)
            
            return {
                'available': True,
                'explanation': explanation,
                'top_features': explanation[:10],
                'confidence': exp.score,
                'local_prediction': exp.local_pred
            }
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _get_feature_importance(self, text):
        """Get basic feature importance using TF-IDF scores"""
        try:
            features = self.vectorizer.transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = features.toarray()[0]
            
            feature_importance = list(zip(feature_names, tfidf_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'top_tfidf_features': feature_importance[:15],
                'document_length': len(text.split()),
                'unique_features': len([f for f in tfidf_scores if f > 0])
            }
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def generate_lime_plot(self, text, predicted_category):
        """Generate LIME visualization"""
        try:
            def predict_proba(texts):
                features = self.vectorizer.transform(texts)
                all_probas = []
                for model in self.models.values():
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features)
                        all_probas.append(proba)
                return np.mean(all_probas, axis=0) if all_probas else list(self.models.values())[0].predict_proba(features)
            
            predicted_class_idx = np.where(self.label_encoder.classes_ == predicted_category)[0][0]
            
            exp = self.lime_explainer.explain_instance(
                text, 
                predict_proba, 
                num_features=10, 
                top_labels=1
            )
            
            fig = exp.as_pyplot_figure(label=predicted_class_idx)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"LIME plot generation failed: {e}")
            return None

    def _calculate_model_agreement(self, model_predictions):
        """Calculate how much models agree on the prediction"""
        if not model_predictions:
            return 0.0
        
        predictions = list(model_predictions.values())
        most_common = max(set(predictions), key=predictions.count)
        agreement = predictions.count(most_common) / len(predictions)
        return agreement
    
    def generate_feedback(self, xai_result):
        """Generate actionable feedback based on XAI analysis"""
        feedback = {
            'strengths': [],
            'improvements': [],
            'suggestions': []
        }
        
        pred_category = xai_result['predicted_category']
        skills = xai_result['skills']
        structure = xai_result['resume_structure']
        confidence = xai_result['confidence']
        
        # Confidence-based feedback
        if confidence < 0.6:
            feedback['improvements'].append("Low prediction confidence. Consider making your skills and experience more specific.")
        
        # Skills feedback
        total_skills = sum(len(skill_list) for skill_list in skills.values())
        if total_skills < 5:
            feedback['improvements'].append(f"Only {total_skills} skills detected. Add more specific technical skills.")
        else:
            feedback['strengths'].append(f"Good skill diversity ({total_skills} skills detected).")
        
        # Structure feedback
        if structure['missing_sections']:
            feedback['improvements'].append(f"Consider adding sections for: {', '.join(structure['missing_sections'])}")
        
        if not structure['has_quantifiable_achievements']:
            feedback['improvements'].append("Add quantifiable achievements (numbers, percentages, metrics) to demonstrate impact.")
        
        if not structure['has_action_verbs']:
            feedback['improvements'].append("Use more action verbs to describe your responsibilities and achievements.")
        
        # Enhanced feature importance feedback (replaces SHAP)
        if xai_result['xai_analysis']['feature_importance']['available']:
            top_features = xai_result['xai_analysis']['feature_importance']['top_positive_features']
            if top_features:
                key_terms = [f[0] for f in top_features[:3]]
                feedback['strengths'].append(f"Strong presence of key terms: {', '.join(key_terms)}")
            
            missing_features = xai_result['xai_analysis']['feature_importance']['top_missing_features']
            if missing_features:
                missing_terms = [f[0] for f in missing_features[:3]]
                feedback['improvements'].append(f"Consider adding these impactful terms: {', '.join(missing_terms)}")
        
        # Category-specific suggestions
        category_suggestions = {
            'Data Science': [
                "Highlight specific ML algorithms and frameworks",
                "Include data visualization projects and metrics",
                "Mention statistical analysis experience"
            ],
            'Java Developer': [
                "Specify Java versions and frameworks",
                "Mention microservices architecture experience",
                "Include build tools and CI/CD experience"
            ],
            'Web Designing': [
                "Showcase responsive design projects",
                "Include specific frameworks and libraries",
                "Mention UI/UX design principles"
            ],
            'HR': [
                "Quantify recruitment metrics",
                "Mention specific HRIS systems",
                "Highlight employee engagement initiatives"
            ]
        }
        
        if pred_category in category_suggestions:
            feedback['suggestions'].extend(category_suggestions[pred_category])
        else:
            feedback['suggestions'].extend([
                "Highlight your most relevant projects",
                "Use action verbs to describe achievements",
                "Quantify achievements with numbers",
                "Include relevant certifications"
            ])
        
        return feedback

class JobDescriptionAnalyzer:
    def __init__(self, resume_analyzer):
        self.resume_analyzer = resume_analyzer
    
    def analyze_job_description(self, job_description):
        """Analyze job description and extract key requirements"""
        analysis = self.resume_analyzer.predict_with_xai(job_description)
        
        # Extract key requirements
        skills = self.resume_analyzer.extract_skills(job_description)
        
        # Find important keywords
        words = job_description.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in self.resume_analyzer.stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analysis['top_keywords'] = [word for word, freq in top_keywords]
        analysis['key_requirements'] = skills
        
        return analysis
    
    def match_resume_to_job(self, resume_text, job_description):
        """Match resume to job description and provide compatibility score"""
        resume_analysis = self.resume_analyzer.predict_with_xai(resume_text)
        job_analysis = self.analyze_job_description(job_description)
        
        # Calculate skill match
        resume_skills = set()
        for category_skills in resume_analysis['skills'].values():
            resume_skills.update(category_skills)
        
        job_skills = set()
        for category_skills in job_analysis['key_requirements'].values():
            job_skills.update(category_skills)
        
        common_skills = resume_skills.intersection(job_skills)
        missing_skills = job_skills - resume_skills
        
        # Calculate match scores
        skill_match = len(common_skills) / len(job_skills) if job_skills else 0
        category_match = 1.0 if resume_analysis['predicted_category'] == job_analysis['predicted_category'] else 0.3
        
        # Overall match score
        overall_match = (skill_match * 0.7 + category_match * 0.3)
        
        return {
            'overall_match': overall_match,
            'skill_match': skill_match,
            'category_match': category_match,
            'common_skills': list(common_skills),
            'missing_skills': list(missing_skills),
            'resume_analysis': resume_analysis,
            'job_analysis': job_analysis
        }

class ResumeAnalyzerApp:
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        """Load all trained models and components"""
        try:
            # Load base components
            self.tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            
            # Load individual models
            self.models = {}
            model_files = {
                'KNN': 'k-nearest_neighbors_model.pkl',
                'Support Vector Machine': 'support_vector_machine_model.pkl',
                'Random Forest': 'random_forest_model.pkl',
                'Multinomial Naive Bayes': 'multinomial_naive_bayes_model.pkl',
                'Logistic Regression': 'logistic_regression_model.pkl'
            }
            
            for name, file in model_files.items():
                filepath = f'models/{file}'
                if os.path.exists(filepath):
                    try:
                        self.models[name] = joblib.load(filepath)
                        st.sidebar.success(f"✅ {name} loaded")
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ Could not load {name}: {e}")
                else:
                    st.sidebar.warning(f"⚠️ File not found: {file}")
            
            # Load CatBoost if available
            if os.path.exists('models/catboost_model.cbm'):
                try:
                    self.models['CatBoost'] = CatBoostClassifier()
                    self.models['CatBoost'].load_model('models/catboost_model.cbm')
                    st.sidebar.success("✅ CatBoost model loaded")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Could not load CatBoost: {e}")
            
            # Check if we have any models loaded
            if not self.models:
                st.error("❌ No models were successfully loaded. Please check your model files.")
                st.stop()
            
            # Create resume analyzer instance
            self.resume_analyzer = ResumeAnalyzer(self.tfidf_vectorizer, self.label_encoder, self.models)
            self.job_analyzer = JobDescriptionAnalyzer(self.resume_analyzer)
            
            st.sidebar.success(f"✅ {len(self.models)} models loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.error("Please make sure you've run the notebook first to train the models.")
            st.stop()
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"❌ Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"❌ Error reading DOCX: {str(e)}")
            return ""
    
    def extract_text_from_file(self, uploaded_file):
        """Extract text from any supported file type"""
        if uploaded_file is None:
            return ""
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            return self.extract_text_from_docx(uploaded_file)
        elif file_extension == 'txt':
            return str(uploaded_file.read(), 'utf-8')
        else:
            st.error(f"❌ Unsupported file type: {file_extension}")
            return ""

def main():
    st.markdown('<h1 class="main-header">🎯 AI Resume Analyzer & Career Coach</h1>', unsafe_allow_html=True)
    
    # Initialize app
    app = ResumeAnalyzerApp()
    
    # Sidebar
    st.sidebar.title("🧭 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Resume Analysis", "Job Matching", "CV Improvement", "Model Performance", "About"]
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Info")
    xai_status = "✅ LIME Available" if LIME_AVAILABLE else "⚠️ LIME Not Installed"
    st.sidebar.info(f"""
    - **Models Loaded**: {len(app.models)}
    - **Categories**: {len(app.label_encoder.classes_)}
    - **XAI Features**: {xai_status}
    - **Last Updated**: {datetime.now().strftime("%Y-%m-%d")}
    """)
    
    if app_mode == "Resume Analysis":
        show_resume_analysis(app)
    elif app_mode == "Job Matching":
        show_job_matching(app)
    elif app_mode == "CV Improvement":
        show_cv_improvement(app)
    elif app_mode == "Model Performance":
        show_model_performance(app)
    elif app_mode == "About":
        show_about_page()

def show_resume_analysis(app):
    st.markdown('<h2 class="sub-header">📄 Resume Analysis with Enhanced XAI</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Upload or Paste Your Resume")
        
        # File upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📁 **Upload your resume** (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            help="Upload your resume file or paste text below",
            key="resume_analysis_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process uploaded file
        extracted_text = ""
        if uploaded_file is not None:
            extracted_text = app.extract_text_from_file(uploaded_file)
            if extracted_text:
                st.success(f"✅ Successfully extracted text from {uploaded_file.name}")
                st.session_state.resume_text = extracted_text
            else:
                st.error("❌ Could not extract text from the uploaded file")
        
        # Initialize session state for resume text
        if 'resume_text' not in st.session_state:
            st.session_state.resume_text = extracted_text if extracted_text else ""
        
        resume_text = st.text_area(
            "**Or enter your resume text below:**",
            height=300,
            placeholder="Copy and paste your complete resume text here...",
            help="The more complete your resume text, the more accurate the analysis will be.",
            value=st.session_state.resume_text,
            key="resume_analysis_text"
        )
        
        # Analysis button
        col1_1, col1_2 = st.columns([1, 3])
        with col1_1:
            analyze_btn = st.button("🚀 Analyze Resume", type="primary", use_container_width=True)
        
        with col1_2:
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.resume_text = ""
                st.rerun()
        
        if analyze_btn:
            if not resume_text.strip():
                st.error("❌ Please enter your resume text to analyze.")
                return
                
            if len(resume_text.strip()) < 50:
                st.error("❌ Please provide a longer resume text (minimum 50 characters).")
                return
                
            with st.spinner("🔍 Analyzing your resume with AI and Enhanced XAI..."):
                try:
                    # Predict category with Enhanced XAI
                    result = app.resume_analyzer.predict_with_xai(resume_text)
                    
                    if result.get('error'):
                        st.error(f"❌ Analysis error: {result['error']}")
                        return
                    
                    # Display main results
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    # Prediction card
                    confidence = result['confidence']
                    if confidence > 0.7:
                        confidence_color = "#10B981"
                        confidence_text = "High Confidence"
                        confidence_emoji = "🎯"
                    elif confidence > 0.5:
                        confidence_color = "#F59E0B" 
                        confidence_text = "Medium Confidence"
                        confidence_emoji = "📊"
                    else:
                        confidence_color = "#EF4444"
                        confidence_text = "Low Confidence"
                        confidence_emoji = "🔍"
                    
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2 style="font-size: 2rem; margin-bottom: 1rem;">{confidence_emoji} Predicted Career Category</h2>
                        <h1 style="color: {confidence_color}; margin: 20px 0; font-size: 3rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{result['predicted_category']}</h1>
                        <div style="background: rgba(255,255,255,0.3); padding: 15px; border-radius: 12px; font-size: 1.2rem;">
                            <strong>Confidence Level:</strong> {confidence:.2%} ({confidence_text})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # XAI Model Consensus
                    st.markdown("### 🤖 Model Consensus (Ensemble XAI)")
                    model_cols = st.columns(min(3, len(result['model_consensus'])))
                    
                    for i, (model_name, prediction) in enumerate(result['model_consensus'].items()):
                        with model_cols[i % len(model_cols)]:
                            agreement_color = "#10B981" if prediction == result['predicted_category'] else "#6B7280"
                            
                            st.markdown(f"""
                            <div class="model-card" style="border: 3px solid {agreement_color};">
                                <h4 style="color: white; margin-bottom: 10px;">{model_name}</h4>
                                <h3 style="color: white; margin: 10px 0;">{prediction}</h3>
                                <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem;">Prediction</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Enhanced XAI Analysis Section (SHAP replaced with Enhanced Feature Analysis)
                    st.markdown("### 🔍 Enhanced Explainable AI Insights")
                    
                    # Enhanced Feature Importance Analysis
                    feature_data = result['xai_analysis']['feature_importance']
                    if feature_data['available']:
                        st.markdown("#### 📊 Enhanced Feature Impact Analysis")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🎯 Most Impactful Features:**")
                            for feature, importance in feature_data['top_positive_features'][:5]:
                                st.markdown(f'<div class="feature-card-positive">➕ {feature}: {importance:.4f}</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**💡 Suggested Features to Add:**")
                            for feature, importance in feature_data['top_missing_features'][:5]:
                                st.markdown(f'<div class="feature-card-negative">📝 {feature}</div>', unsafe_allow_html=True)
                        
                        # Feature importance visualization
                        if 'feature_breakdown' in feature_data:
                            breakdown = feature_data['feature_breakdown']
                            fig = go.Figure(go.Bar(
                                y=breakdown['features'][:10],
                                x=breakdown['scores'][:10],
                                orientation='h',
                                marker_color='#2563EB'
                            ))
                            fig.update_layout(
                                title="Top Feature Impact Scores",
                                xaxis_title="Impact Score",
                                yaxis_title="Features",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # LIME Analysis
                    if result['xai_analysis']['lime']['available']:
                        st.markdown("#### 🍋 LIME Local Explanation")
                        lime_data = result['xai_analysis']['lime']
                        
                        st.markdown("**Top Contributing Features:**")
                        for feature, importance in lime_data['top_features']:
                            importance_color = "#10B981" if importance > 0 else "#EF4444"
                            st.markdown(f"""
                            <div style="background: {importance_color}; color: white; padding: 0.8rem; border-radius: 8px; margin: 0.3rem 0;">
                                <strong>{feature}</strong>: {importance:.4f}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Generate LIME plot
                        lime_plot = app.resume_analyzer.generate_lime_plot(resume_text, result['predicted_category'])
                        if lime_plot:
                            st.pyplot(lime_plot)
                    else:
                        st.info("ℹ️ LIME analysis not available. Install LIME for local explanations.")
                    
                    # Model Insights
                    model_insights = result['xai_analysis']['model_insights']
                    if model_insights['available']:
                        st.markdown("#### 🤝 Model Agreement Analysis")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Consensus Level", f"{model_insights['consensus_level']:.2%}")
                        with col2:
                            st.metric("Strongest Model", model_insights['strongest_model'])
                    
                    # Skills analysis
                    skills = result['skills']
                    if skills:
                        st.markdown("### 🔧 Skills Analysis")
                        
                        # Display skills by category
                        for category, skill_list in skills.items():
                            if skill_list:
                                with st.expander(f"📁 {category.replace('_', ' ').title()} ({len(skill_list)} skills)"):
                                    cols = st.columns(3)
                                    for i, skill in enumerate(skill_list):
                                        with cols[i % 3]:
                                            st.markdown(f'''
                                            <div class="skill-card">
                                                <strong>{skill.title()}</strong>
                                            </div>
                                            ''', unsafe_allow_html=True)
                    
                    # Generate and display feedback
                    feedback = app.resume_analyzer.generate_feedback(result)
                    
                    if feedback['improvements']:
                        st.markdown("### ⚠️ Areas for Improvement")
                        for improvement in feedback['improvements']:
                            st.markdown(f'<div class="warning-card">{improvement}</div>', unsafe_allow_html=True)
                    
                    if feedback['strengths']:
                        st.markdown("### ✅ Current Strengths")
                        for strength in feedback['strengths']:
                            st.markdown(f'<div class="success-card">{strength}</div>', unsafe_allow_html=True)
                    
                    if feedback['suggestions']:
                        st.markdown("### 💡 Actionable Suggestions")
                        for i, suggestion in enumerate(feedback['suggestions'][:6], 1):
                            st.markdown(f'''
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                     color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                                <strong>{i}.</strong> {suggestion}
                            </div>
                            ''', unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
    
    with col2:
        st.markdown("### 💡 Tips for Better Analysis")
        
        tips = [
            "**Include Specific Technologies**: Mention programming languages, tools, and frameworks",
            "**Add Quantifiable Achievements**: Use numbers and metrics to showcase impact",
            "**Highlight Projects**: Describe key projects and your contributions",
            "**Mention Certifications**: Include relevant certifications and courses",
            "**Use Industry Keywords**: Include terms common in your target industry"
        ]
        
        for tip in tips:
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                {tip}
            </div>
            ''', unsafe_allow_html=True)

# [Rest of the functions remain the same as in your original code, just with updated colors]
# show_job_matching, show_cv_improvement, show_model_performance, show_about_page functions
# ... (they remain unchanged from your original code, just using the new blue color scheme)

def show_job_matching(app):
    st.markdown('<h2 class="sub-header">💼 AI-Powered Job Matching</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Your Resume")
        
        # Resume file upload
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        resume_file = st.file_uploader(
            "📁 **Upload your resume** (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            help="Upload your resume file OR paste text below",
            key="job_matching_resume_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Extract text from resume file
        resume_extracted_text = ""
        if resume_file is not None:
            resume_extracted_text = app.extract_text_from_file(resume_file)
            if resume_extracted_text:
                st.success(f"✅ Resume loaded from {resume_file.name}")
                st.session_state.job_matching_resume_text = resume_extracted_text
            else:
                st.error("❌ Could not extract text from the uploaded file")
        
        # Resume text area
        resume_text = st.text_area(
            "**Or paste your resume text:**",
            height=200,
            placeholder="Enter your resume text here...",
            value=st.session_state.get('job_matching_resume_text', resume_extracted_text),
            key="job_matching_resume_text"
        )
    
    with col2:
        st.markdown("### 📋 Job Description")
        
        # Job description file upload
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        job_file = st.file_uploader(
            "📁 **Upload job description** (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            help="Upload the job description file OR paste text below",
            key="job_matching_job_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Extract text from job description file
        job_extracted_text = ""
        if job_file is not None:
            job_extracted_text = app.extract_text_from_file(job_file)
            if job_extracted_text:
                st.success(f"✅ Job description loaded from {job_file.name}")
                st.session_state.job_matching_job_text = job_extracted_text
            else:
                st.error("❌ Could not extract text from the uploaded file")
        
        # Job description text area
        job_description = st.text_area(
            "**Or paste the job description:**",
            height=200,
            placeholder="Enter the job description here...",
            value=st.session_state.get('job_matching_job_text', job_extracted_text),
            key="job_matching_job_text"
        )
    
    # Clear buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Clear Resume", use_container_width=True):
            st.session_state.job_matching_resume_text = ""
            st.rerun()
    with col2:
        if st.button("🔄 Clear Job", use_container_width=True):
            st.session_state.job_matching_job_text = ""
            st.rerun()
    
    # Analysis button
    if st.button("🎯 Analyze Job Match", type="primary", use_container_width=True):
        # Check if we have at least one input method for each
        has_resume_input = bool(resume_text.strip()) or (resume_file is not None)
        has_job_input = bool(job_description.strip()) or (job_file is not None)
        
        if not has_resume_input and not has_job_input:
            st.error("❌ Please provide both resume and job description (file upload OR text input).")
            return
        elif not has_resume_input:
            st.error("❌ Please provide your resume (file upload OR text input).")
            return
        elif not has_job_input:
            st.error("❌ Please provide the job description (file upload OR text input).")
            return
        
        # If files were uploaded but no text was entered, use the extracted text
        final_resume_text = resume_text.strip() if resume_text.strip() else st.session_state.get('job_matching_resume_text', '')
        final_job_text = job_description.strip() if job_description.strip() else st.session_state.get('job_matching_job_text', '')
        
        if not final_resume_text or not final_job_text:
            st.error("❌ Could not extract valid text from the provided inputs.")
            return
        
        with st.spinner("🔍 Analyzing job match..."):
            try:
                match_result = app.job_analyzer.match_resume_to_job(final_resume_text, final_job_text)
                
                # Display match results
                overall_match = match_result['overall_match']
                
                if overall_match > 0.7:
                    match_color = "#10B981"
                    match_text = "Excellent Match"
                    match_emoji = "🎯"
                elif overall_match > 0.5:
                    match_color = "#F59E0B"
                    match_text = "Good Match"
                    match_emoji = "👍"
                else:
                    match_color = "#EF4444"
                    match_text = "Needs Improvement"
                    match_emoji = "💡"
                
                st.markdown(f"""
                <div class="match-card">
                    <h2 style="font-size: 2rem; margin-bottom: 1rem;">{match_emoji} Job Match Analysis</h2>
                    <h1 style="color: {match_color}; margin: 20px 0; font-size: 3rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{overall_match:.1%}</h1>
                    <div style="background: rgba(255,255,255,0.3); padding: 15px; border-radius: 12px; font-size: 1.2rem;">
                        <strong>Match Level:</strong> {match_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # PIE CHART for match breakdown
                st.markdown("### 📊 Match Breakdown")
                
                # Create data for pie chart
                match_data = {
                    'Component': ['Skill Match', 'Category Match', 'Gap'],
                    'Value': [
                        match_result['skill_match'] * 100,
                        match_result['category_match'] * 100,
                        max(0, 100 - (match_result['skill_match'] * 100 + match_result['category_match'] * 100))
                    ]
                }
                
                # Create pie chart
                fig = px.pie(
                    match_data, 
                    values='Value', 
                    names='Component',
                    title="Job Match Composition",
                    color='Component',
                    color_discrete_map={
                        'Skill Match': '#2563EB',
                        'Category Match': '#8B5CF6', 
                        'Gap': '#6B7280'
                    }
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(showlegend=True, height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ✅ Matching Skills")
                    if match_result['common_skills']:
                        for skill in match_result['common_skills']:
                            st.markdown(f'<div class="success-card">{skill.title()}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No common skills found")
                
                with col2:
                    st.markdown("### 📝 Skills to Add")
                    if match_result['missing_skills']:
                        for skill in match_result['missing_skills'][:5]:
                            st.markdown(f'<div class="warning-card">{skill.title()}</div>', unsafe_allow_html=True)
                    else:
                        st.success("Great! No major skills missing")
                
                # Additional insights
                st.markdown("### 🔍 Additional Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Skill Match", f"{match_result['skill_match']:.1%}")
                    st.metric("Category Match", f"{match_result['category_match']:.1%}")
                
                with col2:
                    st.metric("Your Predicted Category", match_result['resume_analysis']['predicted_category'])
                    st.metric("Job Category", match_result['job_analysis']['predicted_category'])
                
                # Improvement suggestions
                if overall_match < 0.7:
                    st.markdown("### 💡 Improvement Suggestions")
                    suggestions = [
                        "Focus on acquiring the missing skills highlighted above",
                        "Tailor your resume to include more keywords from the job description",
                        "Highlight projects that demonstrate required skills",
                        "Consider taking relevant courses or certifications for missing skills"
                    ]
                    
                    for i, suggestion in enumerate(suggestions, 1):
                        st.markdown(f'''
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                 color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                            <strong>{i}.</strong> {suggestion}
                        </div>
                        ''', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Job matching failed: {str(e)}")

def show_cv_improvement(app):
    st.markdown('<h2 class="sub-header">🚀 CV Improvement Coach</h2>', unsafe_allow_html=True)
    
    st.info("💡 **Get personalized feedback to improve your resume**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📁 **Upload your resume** (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            help="Upload your resume file for improvement analysis",
            key="cv_improvement_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process uploaded file
        extracted_text = ""
        if uploaded_file is not None:
            extracted_text = app.extract_text_from_file(uploaded_file)
            if extracted_text:
                st.success(f"✅ Successfully extracted text from {uploaded_file.name}")
                st.session_state.cv_improvement_text = extracted_text
            else:
                st.error("❌ Could not extract text from the uploaded file")
        
        # Initialize session state for CV improvement text
        if 'cv_improvement_text' not in st.session_state:
            st.session_state.cv_improvement_text = extracted_text if extracted_text else ""
        
        resume_text = st.text_area(
            "**Or paste your resume text below:**",
            height=300,
            placeholder="Enter your resume text for improvement suggestions...",
            value=st.session_state.cv_improvement_text,
            key="cv_improvement_text"
        )
        
        # Buttons
        col1_1, col1_2 = st.columns([1, 3])
        with col1_1:
            analyze_btn = st.button("🎯 Get Improvement Tips", type="primary", use_container_width=True)
        
        with col1_2:
            if st.button("🔄 Clear Text", use_container_width=True):
                st.session_state.cv_improvement_text = ""
                st.rerun()
    
    with col2:
        st.markdown("### 💡 Improvement Areas")
        improvement_areas = [
            "**Structure & Formatting**: Proper sections and organization",
            "**Skills Presentation**: Highlighting relevant technical skills",
            "**Achievements**: Quantifiable results and impact",
            "**Keywords**: Industry-specific terminology",
            "**Action Verbs**: Strong, active language",
            "**Customization**: Tailoring for specific roles"
        ]
        
        for area in improvement_areas:
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                {area}
            </div>
            ''', unsafe_allow_html=True)
    
    if analyze_btn:
        if not resume_text.strip():
            st.error("❌ Please provide your resume text.")
            return
        
        with st.spinner("🔍 Analyzing your resume for improvements..."):
            try:
                result = app.resume_analyzer.predict_with_xai(resume_text)
                feedback = app.resume_analyzer.generate_feedback(result)
                
                st.markdown("## 📊 Improvement Analysis")
                
                # Structure analysis
                structure = result['resume_structure']
                st.markdown("### 📋 Resume Structure Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Word Count", structure['word_count'])
                with col2:
                    st.metric("Character Count", structure['character_count'])
                with col3:
                    sections_found = sum(structure['sections_found'].values())
                    st.metric("Sections Found", f"{sections_found}/4")
                with col4:
                    if structure['has_quantifiable_achievements']:
                        st.success("✅ Achievements")
                    else:
                        st.warning("⚠️ Needs Achievements")
                
                # Section breakdown
                st.markdown("#### 📁 Section Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Found Sections:**")
                    for section, found in structure['sections_found'].items():
                        if found:
                            st.markdown(f"✅ {section.title()}")
                
                with col2:
                    if structure['missing_sections']:
                        st.markdown("**Missing Sections:**")
                        for section in structure['missing_sections']:
                            st.markdown(f"❌ {section.title()}")
                    else:
                        st.success("🎉 All key sections found!")
                
                # Detailed feedback
                st.markdown("### 💡 Improvement Recommendations")
                
                if feedback['improvements']:
                    st.markdown("#### ⚠️ Areas for Improvement")
                    for improvement in feedback['improvements']:
                        st.markdown(f'<div class="warning-card">{improvement}</div>', unsafe_allow_html=True)
                
                if feedback['strengths']:
                    st.markdown("#### ✅ Current Strengths")
                    for strength in feedback['strengths']:
                        st.markdown(f'<div class="success-card">{strength}</div>', unsafe_allow_html=True)
                
                if feedback['suggestions']:
                    st.markdown("#### 🎯 Actionable Suggestions")
                    for i, suggestion in enumerate(feedback['suggestions'], 1):
                        st.markdown(f'<div class="xai-card"><strong>{i}.</strong> {suggestion}</div>', unsafe_allow_html=True)
                
                # Skills summary
                skills = result['skills']
                total_skills = sum(len(skill_list) for skill_list in skills.values())
                st.markdown(f"#### 🔧 Skills Summary")
                st.info(f"**Total skills detected:** {total_skills}")
                
                # Show skills by category
                for category, skill_list in skills.items():
                    if skill_list:
                        with st.expander(f"📁 {category.replace('_', ' ').title()} ({len(skill_list)} skills)"):
                            cols = st.columns(3)
                            for i, skill in enumerate(skill_list):
                                with cols[i % 3]:
                                    st.markdown(f'''
                                    <div class="skill-card">
                                        <strong>{skill.title()}</strong>
                                    </div>
                                    ''', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

def show_model_performance(app):
    st.markdown('<h2 class="sub-header">🤖 Model Performance & XAI</h2>', unsafe_allow_html=True)
    
    st.info("🔍 **System is running with Enhanced XAI features**")
    
    # Model information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Models Loaded</h3>
            <h1>{len(app.models)}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Categories</h3>
            <h1>{len(app.label_encoder.classes_)}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        xai_status = "✅ LIME Active" if LIME_AVAILABLE else "⚠️ LIME Not Installed"
        st.markdown(f"""
        <div class="metric-card">
            <h3>XAI Features</h3>
            <h1>{xai_status}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Model details
    st.markdown("### 🧠 Loaded Models")
    model_cols = st.columns(3)
    
    for i, (name, model) in enumerate(app.models.items()):
        with model_cols[i % 3]:
            st.markdown(f"""
            <div class="model-card">
                <h4>{name}</h4>
                <p>✅ Loaded Successfully</p>
            </div>
            """, unsafe_allow_html=True)
    
    # XAI Status
    st.markdown("### 🔍 Explainable AI Status")
    
    lime_status = "✅ Available" if LIME_AVAILABLE else "❌ Not Available"
    st.markdown(f"""
    <div class="lime-card">
        <h4>LIME Analysis</h4>
        <p>{lime_status}</p>
        <p>Local interpretable model explanations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Feature Analysis Status
    st.markdown(f"""
    <div class="feature-card-positive">
        <h4>Enhanced Feature Analysis</h4>
        <p>✅ Always Available</p>
        <p>Combined TF-IDF and model-based feature importance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Categories
    st.markdown("### 🎯 Available Categories")
    categories = app.label_encoder.classes_
    
    # Display categories in a nice grid
    cols = st.columns(4)
    for i, category in enumerate(categories):
        with cols[i % 4]:
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%); 
                     color: white; padding: 0.5rem; border-radius: 8px; margin: 0.2rem 0; text-align: center;">
                {category}
            </div>
            ''', unsafe_allow_html=True)

def show_about_page():
    st.markdown('<h2 class="sub-header">ℹ️ About AI Resume Analyzer</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 AI-Powered Resume Analysis</h3>
        <p>This application uses advanced machine learning models to analyze resumes and provide career guidance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Key Features")
        features = [
            "🤖 **Multiple ML Models**: Ensemble of 5+ machine learning algorithms",
            "🎯 **Career Prediction**: Accurate job category prediction",
            "🔍 **Enhanced XAI**: LIME + Advanced Feature Analysis for transparent AI",
            "💼 **Job Matching**: Resume to job description compatibility",
            "🚀 **CV Improvement**: Actionable feedback for resume enhancement",
            "📊 **Skills Analysis**: Detailed technical skills extraction",
            "📁 **File Upload**: Support for PDF, DOCX, and TXT files"
        ]
        
        for feature in features:
            st.markdown(f"- {feature}")
    
    with col2:
        st.markdown("### 🛠️ Technologies Used")
        technologies = [
            "**Streamlit**: Web application framework",
            "**Scikit-learn**: Machine learning algorithms",
            "**CatBoost**: Gradient boosting framework",
            "**LIME**: Local interpretable model explanations",
            "**NLTK**: Natural language processing",
            "**Plotly**: Interactive visualizations",
            "**PyPDF2/Docx**: Document text extraction"
        ]
        
        for tech in technologies:
            st.markdown(f"- {tech}")
    
    st.markdown("### 📈 How It Works")
    st.markdown("""
    1. **Text Processing**: Resumes are cleaned and preprocessed
    2. **Feature Extraction**: TF-IDF vectorization converts text to features
    3. **Model Ensemble**: Multiple ML models make predictions
    4. **Enhanced XAI**: LIME + Advanced Feature Analysis provide transparent explanations
    5. **Feedback Generation**: Personalized improvement suggestions
    """)

if __name__ == "__main__":
    main()