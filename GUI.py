import mlflow
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


class WaterQualityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Water Quality Prediction")
        self.root.geometry("600x800")
        self.root.configure(bg="white")

        # Load model
        self.load_model()

        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(
            main_frame, text="Water Quality Prediction", font=("Helvetica", 20, "bold")
        )
        title.pack(pady=20)

        # Input fields frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, padx=20)

        # Input fields
        self.inputs = {}
        parameters = [
            ("pH", "ph"),
            ("Hardness (mg/L)", "Hardness"),
            ("Solids (ppm)", "Solids"),
            ("Chloramines (ppm)", "Chloramines"),
            ("Sulfate (mg/L)", "Sulfate"),
            ("Conductivity (μS/cm)", "Conductivity"),
            ("Organic Carbon (ppm)", "Organic_carbon"),
            ("Trihalomethanes (μg/L)", "Trihalomethanes"),
            ("Turbidity (NTU)", "Turbidity"),
        ]

        for i, (label, key) in enumerate(parameters):
            frame = ttk.Frame(input_frame)
            frame.pack(fill=tk.X, pady=5)

            label = ttk.Label(frame, text=label, width=25)
            label.pack(side=tk.LEFT, padx=5)

            entry = ttk.Entry(frame)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.inputs[key] = entry

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)

        predict_btn = ttk.Button(button_frame, text="Predict", command=self.predict)
        predict_btn.pack(side=tk.LEFT, padx=10)

        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_fields)
        clear_btn.pack(side=tk.LEFT, padx=10)

        # Results section
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill=tk.X, pady=20)

        self.result_label = ttk.Label(
            result_frame, text="Enter values and click Predict", font=("Helvetica", 14)
        )
        self.result_label.pack()

    def load_model(self):
        try:
            mlflow.set_tracking_uri("https://dagshub.com/jere-mybao/mlops.mlflow")
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions("Best Model")

            if versions:
                run_id = versions[0].run_id
                logged_model = f"runs:/{run_id}/Best Model"
                self.model = mlflow.pyfunc.load_model(logged_model)
            else:
                messagebox.showerror("Error", "No model found")
                self.model = None
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {e}")
            self.model = None

    def predict(self):
        try:
            # Collect input values
            input_data = {}
            for key, entry in self.inputs.items():
                try:
                    value = float(entry.get())
                    input_data[key] = [value]
                except ValueError:
                    messagebox.showerror(
                        "Error", f"Invalid input for {key}. Please enter a number."
                    )
                    return

            # Create DataFrame
            data = pd.DataFrame(input_data)

            # Make prediction
            if self.model:
                prediction = self.model.predict(data)
                result = "POTABLE" if prediction[0] == 1 else "NOT POTABLE"
                color = "green" if prediction[0] == 1 else "red"

                # Update result label
                self.result_label.configure(
                    text=f"Prediction: Water is {result}",
                    foreground=color,
                    font=("Helvetica", 16, "bold"),
                )
            else:
                messagebox.showerror("Error", "Model not loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {e}")

    def clear_fields(self):
        for entry in self.inputs.values():
            entry.delete(0, tk.END)
        self.result_label.configure(
            text="Enter values and click Predict",
            foreground="black",
            font=("Helvetica", 14),
        )


def main():
    root = tk.Tk()
    app = WaterQualityGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
