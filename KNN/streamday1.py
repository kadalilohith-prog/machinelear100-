import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Dashboard",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "count" not in st.session_state:
    st.session_state.count = 0

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🎓 Student Dashboard")

page = st.sidebar.selectbox(
    "Choose a Page",
    [
        "Home",
        "Student Registration",
        "Student Data",
        "Performance",
        "Upload Data"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Settings")

show_data = st.sidebar.checkbox(
    "Show Student Data",
    value=True
)

age = st.sidebar.slider(
    "Select Age",
    15,
    30,
    20
)

# --------------------------------------------------
# SAMPLE DATA
# --------------------------------------------------

students = pd.DataFrame({
    "Name": [
        "Rahul",
        "Priya",
        "Arjun",
        "Sneha",
        "Kiran",
        "Anjali"
    ],
    "Age": [20, 21, 19, 22, 20, 21],
    "Course": [
        "Python",
        "Data Science",
        "Python",
        "AI",
        "Data Science",
        "AI"
    ],
    "Marks": [85, 92, 78, 88, 95, 81]
})

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.title("🎓 Student Management Dashboard")

    st.write(
        "Welcome to the Streamlit Student Dashboard."
    )

    st.markdown(
        """
        This application demonstrates the major
        features of Streamlit.
        """
    )

    st.markdown("---")

    # METRICS

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Students",
            len(students)
        )

    with col2:
        st.metric(
            "Average Marks",
            f"{students['Marks'].mean():.1f}"
        )

    with col3:
        st.metric(
            "Highest Marks",
            students["Marks"].max()
        )

    with col4:
        st.metric(
            "Average Age",
            f"{students['Age'].mean():.1f}"
        )

    st.markdown("---")

    # TWO COLUMNS

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Student Marks")

        st.bar_chart(
            students.set_index("Name")["Marks"]
        )

    with col2:

        st.subheader("📈 Age Distribution")

        st.line_chart(
            students.set_index("Name")["Age"]
        )

# --------------------------------------------------
# STUDENT REGISTRATION
# --------------------------------------------------

elif page == "Student Registration":

    st.title("📝 Student Registration")

    st.write(
        "Enter student information below."
    )

    name = st.text_input(
        "Student Name"
    )

    email = st.text_input(
        "Email Address"
    )

    student_age = st.number_input(
        "Age",
        min_value=15,
        max_value=50,
        value=20
    )

    gender = st.radio(
        "Gender",
        ["Male", "Female", "Other"]
    )

    course = st.selectbox(
        "Select Course",
        [
            "Python",
            "Data Science",
            "Machine Learning",
            "Artificial Intelligence"
        ]
    )

    skills = st.multiselect(
        "Select Skills",
        [
            "Python",
            "SQL",
            "Excel",
            "Machine Learning",
            "Data Visualization"
        ]
    )

    agree = st.checkbox(
        "I confirm that the above information is correct."
    )

    if st.button("Register Student"):

        if name and email and agree:

            st.success(
                f"Student {name} registered successfully!"
            )

            st.write("### Student Details")

            st.write("Name:", name)
            st.write("Email:", email)
            st.write("Age:", student_age)
            st.write("Gender:", gender)
            st.write("Course:", course)
            st.write("Skills:", skills)

        else:

            st.error(
                "Please enter your name, email and confirm the information."
            )

# --------------------------------------------------
# STUDENT DATA
# --------------------------------------------------

elif page == "Student Data":

    st.title("👨‍🎓 Student Data")

    st.write(
        "Student records stored in a Pandas DataFrame."
    )

    if show_data:

        st.dataframe(
            students,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("🔍 Filter Students")

    selected_course = st.selectbox(
        "Select Course",
        ["All"] + list(students["Course"].unique())
    )

    if selected_course == "All":

        filtered_data = students

    else:

        filtered_data = students[
            students["Course"] == selected_course
        ]

    st.dataframe(
        filtered_data,
        use_container_width=True
    )

# --------------------------------------------------
# PERFORMANCE
# --------------------------------------------------

elif page == "Performance":

    st.title("📊 Student Performance")

    tab1, tab2, tab3 = st.tabs(
        [
            "Marks",
            "Statistics",
            "Charts"
        ]
    )

    # TAB 1

    with tab1:

        st.subheader("Student Marks")

        st.dataframe(
            students[
                ["Name", "Marks"]
            ],
            use_container_width=True
        )

    # TAB 2

    with tab2:

        st.subheader("Statistical Summary")

        st.write(
            students["Marks"].describe()
        )

    # TAB 3

    with tab3:

        st.subheader("Marks Chart")

        st.bar_chart(
            students.set_index("Name")["Marks"]
        )

# --------------------------------------------------
# UPLOAD DATA
# --------------------------------------------------

elif page == "Upload Data":

    st.title("📂 Upload Student Data")

    st.write(
        "Upload a CSV file containing student data."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_data = pd.read_csv(
            uploaded_file
        )

        st.success(
            "File uploaded successfully!"
        )

        st.subheader("Uploaded Data")

        st.dataframe(
            uploaded_data,
            use_container_width=True
        )

        st.subheader("Data Statistics")

        st.write(
            uploaded_data.describe()
        )

# --------------------------------------------------
# SESSION STATE EXAMPLE
# --------------------------------------------------

st.markdown("---")

st.subheader("🔢 Session State Example")

if st.button("Click Counter"):

    st.session_state.count += 1

st.write(
    "Button clicked:",
    st.session_state.count,
    "times"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Built with Python 🐍 and Streamlit 🚀")