import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the stock data
def load_data(file_path):
    """Load the stock data from the provided file."""
    column_names = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    df = pd.read_csv(file_path, header=None, names=column_names)
    return df

# Step 2: Clean the data by converting to numeric
def clean_data(df):
    """Clean data by converting relevant columns to numeric and handling missing values."""
    # Convert relevant columns to numeric
    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
    for column in columns_to_convert:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop rows with NaN values
    df.dropna(inplace=True)

    return df

# Step 3: Preprocess the data - Scale the features
def preprocess_data(df):
    """Preprocess the data by scaling numeric features."""
    features = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Select relevant columns
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(features)
    return data_scaled

# Step 4: Set up and train the Self-Organizing Map (SOM)
def train_som(data_scaled):
    """Train the Self-Organizing Map (SOM) on the scaled data."""
    som = MiniSom(x=10, y=10, input_len=data_scaled.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(data_scaled)  # Initialize the SOM with random weights
    som.train_batch(data_scaled, 100)  # Train the SOM for 100 iterations
    return som

# Step 5: Visualize the SOM clusters using heatmap
def visualize_som_clusters(som):
    """Visualize the SOM clusters using a heatmap."""
    plt.figure(figsize=(10, 8))
    plt.title("Self-Organizing Map Clusters")
    sns.heatmap(som.distance_map().T, cmap='coolwarm', cbar=False, square=True)
    plt.show()

# Step 6: Assign clusters to data points based on SOM winning node
def assign_clusters(df, som, data_scaled):
    """Assign clusters to each data point based on the SOM's winning node."""
    labels = np.zeros(len(data_scaled))  # Initialize an array to store the labels
    for i, x in enumerate(data_scaled):
        w = som.winner(x)  # Get the winning node for each data point
        labels[i] = w[0] * 10 + w[1]  # Assign a label based on the SOM grid position

    # Add the cluster labels to the dataframe
    df['SOM_Cluster'] = labels.astype(int)
    return df

# Step 7: Analyze clusters - Calculate the mean of features per cluster
def cluster_analysis(df):
    """Perform analysis of clusters."""
    print("\nCluster Analysis:")

    # Exclude non-numeric columns like 'Date' and 'Symbol' before performing the groupby operation
    numeric_df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Only keep numeric columns

    # Calculate the mean for each cluster
    cluster_means = df.groupby('SOM_Cluster')[['Open', 'High', 'Low', 'Close', 'Volume']].mean()

    # Print the cluster means
    print(cluster_means)
    return cluster_means

# Step 8: Visualize the distribution of the target variable (Close) per cluster
def visualize_cluster_distribution(df):
    """Visualize the distribution of 'Close' price by cluster."""
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='SOM_Cluster', y='Close', data=df)
    plt.title("Distribution of Close Price by Cluster")
    plt.show()

# Step 9: Visualize the correlation matrix of features
def visualize_correlation_matrix(df):
    """Visualize the correlation matrix of features."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[['Open', 'High', 'Low', 'Close', 'Volume']].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix of Features")
    plt.show()

# Main function to execute the program
def main():
    """Main function to execute the program."""
    # Load and clean the data
    file_path = 'stock_data.csv'  # Replace with your actual file path
    df = load_data(file_path)
    df = clean_data(df)

    # Preprocess the data
    data_scaled = preprocess_data(df)

    # Train the SOM
    som = train_som(data_scaled)

    # Visualize the SOM clusters
    visualize_som_clusters(som)

    # Assign clusters to data points
    df = assign_clusters(df, som, data_scaled)

    # Perform cluster analysis
    cluster_means = cluster_analysis(df)

    # Visualize the distribution of 'Close' price by cluster
    visualize_cluster_distribution(df)

    # Visualize the correlation matrix
    visualize_correlation_matrix(df)

# Execute the main function
if __name__ == "__main__":
    main()

