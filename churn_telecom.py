from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(42)

class MLP(nn.Module):
    def __init__(self,input_size,hidden_size1,hidden_size2,output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size1)
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.fc3 = nn.Linear(hidden_size2,output_size)
        self.ReLu = nn.ReLU()
        self.Dropout = nn.Dropout(0.3)
    def forward(self,x):
        x = self.Dropout(self.ReLu(self.fc1(x)))
        x = self.Dropout(self.ReLu(self.fc2(x)))
        x = self.fc3(x)
        return x

data = pd.read_csv(Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.info()
print(data.head())

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

categorical_columns = [
    "gender","Partner","Dependents","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity",
    "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
    "StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]

numerical_columns = [
    "tenure", "MonthlyCharges", "TotalCharges","SeniorCitizen"]

numerical_transformer = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown = "ignore", sparse_output = False))
])

preprocessor = ColumnTransformer(transformers = [
    ("numerical", numerical_transformer, numerical_columns),
    ("categorical", categorical_transformer, categorical_columns)
])

X = data.drop(columns = ["customerID", "Churn"])
y = data["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state = 42,
    test_size = 0.20,
    stratify = y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

X_train = torch.tensor(X_train_processed, dtype=torch.float32)
X_test = torch.tensor(X_test_processed, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

input_size = X_train.shape[1]

model = MLP(input_size = input_size, hidden_size1 = 64, hidden_size2 = 32, output_size = 1)
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)


num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1} / {num_epochs}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    logits = model(X_test)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).float()
    accuracy = (predictions == y_test).float().mean()

print(f"Accuracy in test: {accuracy.item():.4f}")

y_true = y_test.numpy().ravel()
y_pred = predictions.numpy().ravel()
y_prob = probabilities.numpy().ravel()

print(classification_report(y_true, y_pred, target_names=["Fica", "Cancela"]))
print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.4f}")