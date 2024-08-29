"""
Attribute Information:

age
sex
chest pain type (4 values)
resting blood pressure
serum cholestoral in mg/dl
fasting blood sugar > 120 mg/dl
resting electrocardiographic results (values 0,1,2)
maximum heart rate achieved
exercise induced angina
oldpeak = ST depression induced by exercise relative to rest
the slope of the peak exercise ST segment
number of major vessels (0-3) colored by flourosopy
thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

# Load data
df = pd.read_csv("heart.csv")

# Preprocess data
x=features = df.drop("target", axis=1)
y=labels = df["target"]

# Normalize numerical data
scaler = StandardScaler()
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

#encode categorical data 
encoder = OneHotEncoder()
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Fit and transform the categorical data
categorical_data_transformed = encoder.fit_transform(features[categorical_cols])
# Convert the sparse matrix to a dense array
categorical_data_transformed_array = categorical_data_transformed.toarray()
# Get the names of the new columns generated by the OneHotEncoder
new_cols = encoder.get_feature_names_out(categorical_cols)
# Create a DataFrame from the dense array
categorical_df = pd.DataFrame(categorical_data_transformed_array, columns=new_cols)
# Drop the original categorical columns from 'features'
features = features.drop(columns=categorical_cols)
# Concatenate the original DataFrame 'features' with the new DataFrame 'categorical_df'
features = pd.concat([features, categorical_df], axis=1)













# Split data into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert data to tensors
features_train = torch.tensor(features_train.values, dtype=torch.float)
labels_train = torch.tensor(labels_train.values, dtype=torch.long)
features_test = torch.tensor(features_test.values, dtype=torch.float)
labels_test = torch.tensor(labels_test.values, dtype=torch.long)

# Create dataloaders
train_data = TensorDataset(features_train, labels_train)
test_data = TensorDataset(features_test, labels_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(features.shape[1], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} - Loss: {loss.item()}")

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy: %d %%" % (100 * correct / total))
