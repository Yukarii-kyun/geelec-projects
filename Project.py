import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(123)  
n = 4000

# --- Data Generation (Same as before) ---
CaloriesPerDay = np.clip(np.random.normal(2000, 500, n), 800, 4000)
SugarIntakeGrams = np.clip(np.random.normal(50, 20, n), 0, 150)
VegetableServings = np.clip(SugarIntakeGrams * -0.1 + np.random.normal(4, 2, n), 0, 10)
HealthScore = np.clip(CaloriesPerDay * -0.01 + VegetableServings * 3 + np.random.normal(0, 5, n), 50, 100)
PreviousHealthRating = np.clip((HealthScore / 25), 1.0, 4.0)
ExerciseHours = np.clip(np.random.normal(1, 0.5, n), 0, 5)
SedentaryHours = np.clip(np.random.normal(6, 2, n), 0, 16)
JunkFoodCount = np.clip((10 - VegetableServings + np.random.normal(0, 2, n)).astype(int), 0, 10)
MealBalanceScore = np.clip(SugarIntakeGrams * -0.5 + np.random.normal(20, 10, n), 30, 100)

NutritionScore = np.clip(
    CaloriesPerDay * -0.002 + VegetableServings * 2.0 + SugarIntakeGrams * -0.3 + 
    HealthScore * 0.3 + MealBalanceScore * 0.1 - SedentaryHours * 1.0 - JunkFoodCount * 2.0, 0, 100
)

NutritionLevel = pd.cut(NutritionScore, bins=3, labels=["Unhealthy", "Moderate", "Healthy"])

df = pd.DataFrame({
    "CaloriesPerDay": CaloriesPerDay, "SugarIntakeGrams": SugarIntakeGrams,
    "VegetableServings": VegetableServings, "HealthScore": HealthScore,
    "PreviousHealthRating": PreviousHealthRating, "ExerciseHours": ExerciseHours,
    "SedentaryHours": SedentaryHours, "JunkFoodCount": JunkFoodCount,
    "MealBalanceScore": MealBalanceScore, "NutritionScore": NutritionScore,
    "NutritionLevel": NutritionLevel 
})

# --- Preprocessing ---
encoder = LabelEncoder()
df['NutritionLevel_Encoded'] = encoder.fit_transform(df['NutritionLevel'])

X = df.drop(['NutritionLevel', 'NutritionLevel_Encoded', 'NutritionScore'], axis=1)
y = df['NutritionLevel_Encoded']

# FIX 1: Added stratify=y to ensure all classes are present in the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model ---
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test_scaled)

# FIX 2: Explicitly pass the unique labels found in the encoder to match target_names
unique_labels = np.unique(y) 
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", 
      classification_report(y_test, y_pred, labels=unique_labels, target_names=encoder.classes_))

# --- Prediction (New Data) ---
new_food1 = pd.DataFrame([[3500, 90, 3, 70, 2, 5, 2, 8, 30]], columns=X.columns)
new_food2 = pd.DataFrame([[1800, 115, 0, 20, 2, 1, 1, 15, 30]], columns=X.columns)
new_food3 = pd.DataFrame([[2500, 100, 6, 10, 5, 2, 3, 5, 50]], columns=X.columns)

pred1 = model.predict(scaler.transform(new_food1))
pred2 = model.predict(scaler.transform(new_food2))
pred3 = model.predict(scaler.transform(new_food3))

print("Predicted Nutrition Level for Food 1:", encoder.inverse_transform(pred1)[0])
print("Predicted Nutrition Level for Food 2:", encoder.inverse_transform(pred2)[0])
print("Predicted Nutrition Level for Food 3:", encoder.inverse_transform(pred3)[0])