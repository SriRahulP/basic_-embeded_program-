import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import getpass

# Step 1: Load and clean data
df = pd.read_csv("dataset.csv")
df = df.dropna()

# Step 2: Undersample each class to match the smallest class
min_count = df['strength'].value_counts().min()

df_weak = df[df['strength'] == 0].sample(min_count, random_state=1)
df_medium = df[df['strength'] == 1].sample(min_count, random_state=1)
df_strong = df[df['strength'] == 2].sample(min_count, random_state=1)

balanced_df = pd.concat([df_weak, df_medium, df_strong]).sample(frac=1, random_state=1).reset_index(drop=True)

# Optional: Confirm new class distribution
print("Balanced class distribution:\n", balanced_df['strength'].value_counts())

# Step 3: Feature and Label Separation
x = balanced_df["password"]
y = balanced_df["strength"]

# Step 4: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000)
x = tfidf.fit_transform(x)

# Step 5: Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=50)

# Step 6: Train Model
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(x_train, y_train)

# Step 7: Accuracy
print("Model accuracy:", model.score(x_test, y_test))

# Step 8: User Prediction
user_password = getpass.getpass("Enter a password to check its strength: ")
user_vector = tfidf.transform([user_password])
strength = model.predict(user_vector)

# Step 9: Label Mapping
label_map = {
    0: "hmmm...looks WEAK...must try a strong one",
    1: "heyyy...looks MEDIUM...maybe you can try a strong one",
    2: "good...looks STRONG...you should go with this password"
}
print("Password strength:", label_map[int(strength[0])])
