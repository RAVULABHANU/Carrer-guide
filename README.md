#  Smart Career Guide System (Machine Learning)

## Project Overview

The Smart Career Guide System is a machine learning–based web application that recommends suitable academic courses and career paths based on a user’s interests and hobbies. The application uses a trained Random Forest model to analyze user-selected interests and predict the most relevant course along with career-related insights.

This project is designed to help students make informed career decisions by combining data analysis, machine learning, and an interactive web interface.

---

##  Features

* Interactive Streamlit-based user interface
* Interest and hobby selection across multiple domains
* Machine learning–based course recommendation
* Displays top career options, highest positions, average salary, and social respect
* Individual career suggestions based on each selected interest

---

##  Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Machine Learning (Random Forest Classifier)

---

##  Project Structure

```
Smart-Career-Guide-System/
│
├── app.py                 # Main Streamlit application
├── mlproject.csv          # Dataset used for training the ML model
├── requirements.txt       # Required Python libraries
├── README.md              # Project documentation
├── .gitattributes         # Git configuration file
```

---

##  How the System Works

1. The user selects interests and hobbies from different categories such as Technology, Arts, Medical, Business, and Languages.
2. The selected interests are converted into a feature vector.
3. A Random Forest machine learning model predicts the most suitable course.
4. Career-related details associated with the predicted course are displayed.
5. Additional career suggestions are generated for each selected interest.

---

##  How to Run the Project

### Step 1: Clone the Repository

```
git clone https://github.com/your-username/smart-career-guide-system.git
cd smart-career-guide-system
```

### Step 2: Install Dependencies

```
pip install -r requirements.txt
```

### Step 3: Run the Application

```
streamlit run app.py
```

---

##  Dataset

The dataset (`mlproject.csv`) contains interest-related features along with course names and career-related attributes such as:

* Top Careers
* Highest Position
* Average Salary
* Social Respect

The model is trained dynamically when the application runs.

---

##  Learning Outcomes

* Gained hands-on experience with data preprocessing and EDA
* Implemented a machine learning model for recommendation systems
* Built an end-to-end ML application using Streamlit
* Improved understanding of feature engineering and model evaluation

---


