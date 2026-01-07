import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import ipywidgets as widgets
from IPython.display import display, clear_output

# Load dataset
data = pd.read_csv("houses_pakistan_realistic.csv")

# Features and target
X = data[["city", "size_marla", "stories"]]
y = data["price_pkr"]

# Encode city
preprocessor = ColumnTransformer(
    transformers=[
        ("city", OneHotEncoder(handle_unknown="ignore"), ["city"])
    ],
    remainder="passthrough"
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

print("✅ Model trained successfully!\n")

# ---------------- Interactive UI ---------------- #

# Get unique cities from the training data for the dropdown
available_cities = sorted(X_train['city'].unique().tolist())

# Create widgets for user input
city_widget = widgets.Dropdown(
    options=available_cities,
    description='City:',
    disabled=False,
)

size_widget = widgets.FloatSlider(
    value=5,
    min=1,
    max=200,
    step=1,
    description='Size (Marla):',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)

stories_widget = widgets.IntSlider(
    value=1,
    min=1,
    max=10,
    step=1,
    description='Stories:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
)

predict_button = widgets.Button(
    description='Predict Price',
    disabled=False,
    button_style='success',
    tooltip='Click to predict house price'
)

output_widget = widgets.Output()

def on_predict_button_clicked(b):
    with output_widget:
        clear_output()
        current_input_data = pd.DataFrame({
            "city": [city_widget.value],
            "size_marla": [size_widget.value],
            "stories": [stories_widget.value]
        })
        predicted_price = model.predict(current_input_data)[0]
        # Convert to lacs & crores
        price_lac = predicted_price / 100000
        price_crore = predicted_price / 10000000

        print("\n🏠 Predicted House Price:")
        print(f"PKR {predicted_price:,.0f}")
        print(f"≈ {price_lac:.2f} Lacs")
        print(f"≈ {price_crore:.2f} Crore")

predict_button.on_click(on_predict_button_clicked)

# Arrange widgets
input_widgets = widgets.VBox([
    city_widget,
    size_widget,
    stories_widget,
    predict_button
])

display(input_widgets, output_widget)