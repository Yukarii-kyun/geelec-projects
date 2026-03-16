import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

np.random.seed(42)  
n = 1000

CaloriesPerDay = np.clip(np.random.normal(2000, 500, n), 800, 4000)
SugarIntakeGrams = np.clip(np.random.normal(50, 20, n), 0, 150)
VegetabbleServings = np.clip(SugarIntakeGrams * -0.1 + np.random.normal(4, 2, n), 0, 10)

HealthScore = np.clip(
    CaloriesPerDay * -0.01 + VegetabbleServings * 3 + np.random.normal(0, 5, n), 50, 100
)

PreviousHealthRating = np.clip((HealthScore / 25), 1.0, 4.0)
ExerciseHours = np.clip(np.random.normal(1, 0.5, n), 0, 5)
SedentaryHours = np.clip(np.random.normal(6, 2, n), 0, 16)

JunkFoodsCount = np.clip(
    (10 - VegetabbleServings + np.random.normal(0, 2, n)).astype(int), 0, 10
)

MealBalancerScore = np.clip(
    SugarIntakeGrams * -0.5 + np.random.normal(20, 10, n), 30, 100
)

NutritionScore = np.clip(
    CaloriesPerDay * -0.002 + VegetabbleServings * 0.5 + SugarIntakeGrams * -0.5 + HealthScore * 0.5 + MealBalancerScore * 0.03 - SedentaryHours * 0.5 - JunkFoodsCount * 1.5, 0, 100
)

low = NutritionScore.min()
high = NutritionScore.max()
mid1 = low + (high - low) / 3
mid2 = low + 2 * (high - low) / 3

NutritionLevel = pd.cut(
    NutritionScore,
    bins=[low - 1, mid1, mid2, high],
    labels=["Unhealthy", "Moderate", "Healthy"]
)

df = pd.DataFrame({
    "CaloriesPerDay": CaloriesPerDay,
    "SugarIntakeGrams": SugarIntakeGrams,
    "VegetabbleServings": VegetabbleServings,
    "HealthScore": HealthScore,
    "PreviousHealthRating": PreviousHealthRating,
    "ExerciseHours": ExerciseHours,
    "SedentaryHours": SedentaryHours,
    "MealBalancerScore": MealBalancerScore,
    "NutritionScore": NutritionScore,
    "NutritionLevel": NutritionLevel 
})
df.head()

encoder = LabelEncoder()
df['NutritionLevel'] = encoder.fit_transform(df['NutritionLevel'])

x = df.drop(['NutritionLevel', 'NutritionScore'], axis=1)
y = df['NutritionLevel']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

importance = pd.DataFrame(
    model.coef_.T, index=x.columns, columns = encoder.classes_
)
importance

new_food1 = pd.DataFrame({
    "CaloriesPerDay": [3500],
    "SugarIntakeGrams": [222],
    "VegetableServings": [0],
    "HealthScore": [50],
    "PreviousHealthRating": [2],
    "ExerciseHours": [1],
    "SedentaryHours": [5],
    "JunkFoodCount": [8],
    "MealBalanceScore": [30],
    "NutritionScore": [20]
})

new_food2 = pd.DataFrame({
    "CaloriesPerDay": [1800],
    "SugarIntakeGrams": [193],
    "VegetableServings": [0],
    "HealthScore": [50],
    "PreviousHealthRating": [2],
    "ExerciseHours": [1],
    "SedentaryHours": [5],
    "JunkFoodCount": [6],
    "MealBalanceScore": [30],
    "NutritionScore": [35]
})