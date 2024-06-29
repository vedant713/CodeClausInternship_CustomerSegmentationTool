import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Function to create a user input form
def user_input_features():
    st.sidebar.header('Customer Input Features')
    age = st.sidebar.slider('Age', 18, 100, 30)
    annual_income = st.sidebar.slider('Annual Income (k$)', 15, 150, 50)
    spending_score = st.sidebar.slider('Spending Score (1-100)', 1, 100, 50)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    data = {'Age': age,
            'Annual Income (k$)': annual_income,
            'Spending Score (1-100)': spending_score,
            'Gender': gender}
    features = pd.DataFrame(data, index=[0])
    return features

# Function to load user data file
def load_user_data():
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        return None

# Main function
def main():
    st.title('Customer Segmentation by Vedant')
    st.write('This app segments customers based on their behavior using K-Means clustering.')
    
    input_df = user_input_features()
    user_data = load_user_data()
    
    if user_data is not None:
        st.subheader('User Uploaded Data')
        st.write(user_data)
        df = user_data
    else:
        st.subheader('Sample Data')
        customer_data = {
            'Age': [19, 21, 20, 23, 31, 22, 35, 23, 64, 30, 67, 35, 58, 24, 37, 22, 35, 20, 52, 35],
            'Annual Income (k$)': [15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24],
            'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 14, 99, 15, 77, 13, 79, 35, 66, 29, 98],
            'Gender': ['Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Female', 'Male', 'Male', 'Male', 'Female', 'Male', 'Female']
        }
        df = pd.DataFrame(customer_data)

    df = pd.concat([df, input_df], ignore_index=True)

    # Clustering
    k = st.sidebar.slider('Number of clusters (k)', 1, 10, 3)
    kmeans = KMeans(n_clusters=k)
    df['Cluster'] = kmeans.fit_predict(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    
    # 2D Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Customer Segmentation')
    st.pyplot(plt)

    # 3D Scatter Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_zlabel('Spending Score (1-100)')
    ax.set_title('3D Customer Segmentation')
    st.pyplot(fig)

    # Display the clustered data
    st.subheader('Clustered Data')
    st.write(df)

    # Box Plot
    st.subheader('Box Plot of Features by Cluster')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(x='Cluster', y='Age', data=df, ax=axes[0])
    axes[0].set_title('Age Distribution by Cluster')
    sns.boxplot(x='Cluster', y='Annual Income (k$)', data=df, ax=axes[1])
    axes[1].set_title('Annual Income Distribution by Cluster')
    sns.boxplot(x='Cluster', y='Spending Score (1-100)', data=df, ax=axes[2])
    axes[2].set_title('Spending Score Distribution by Cluster')
    st.pyplot(fig)

        # Heatmap of Correlation Matrix
    st.subheader('Heatmap of Feature Correlation')
    corr_matrix = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
    heatmap_fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(heatmap_fig)



    # Summary statistics for each cluster, excluding non-numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    cluster_summary = df.groupby('Cluster')[numeric_cols].mean()
    st.subheader('Cluster Summary Statistics')
    st.write(cluster_summary)

    # Download clustered data
    csv = df.to_csv(index=False)
    st.download_button(label="Download data as CSV", data=csv, file_name='clustered_customers.csv', mime='text/csv')

if __name__ == '__main__':
    main()
