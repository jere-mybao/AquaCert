import mlflow
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime

# Set the tracking URI to your DagsHub MLflow instance
mlflow.set_tracking_uri("https://dagshub.com/jere-mybao/mlops.mlflow")
model_name = "Best Model"


class PredictionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set up the window with larger dimensions
        self.title("Water Quality Analysis System")
        self.geometry("800x1000")  # Increased window size
        self.configure(bg="white")  # Changed to white background
        self.resizable(True, True)  # Allow resizing

        # Configure styles
        self.setup_styles()

        # Create main container
        self.main_container = ttk.Frame(self, style="Main.TFrame")
        self.main_container.pack(
            fill="both", expand=True, padx=30, pady=20
        )  # Increased padding

        # Create header
        self.create_header()

        # Create input section
        self.create_input_section()

        # Create results section
        self.create_results_section()

        # Load the model
        self.loaded_model = self.load_model()

        # Initialize status
        self.update_status("Ready for prediction")

    def setup_styles(self):
        style = ttk.Style()

        # Configure frame styles with white background
        style.configure("Main.TFrame", background="white")
        style.configure("Input.TFrame", background="white")
        style.configure("Header.TFrame", background="white")

        # Configure label styles with improved contrast
        style.configure(
            "Header.TLabel",
            font=("Helvetica", 28, "bold"),  # Increased font size
            background="white",
            foreground="#1a1a1a",
        )  # Darker text for better contrast

        style.configure(
            "Status.TLabel",
            font=("Helvetica", 12),  # Increased font size
            background="white",
            foreground="#333333",
        )  # Darker text

        style.configure(
            "Input.TLabel",
            font=("Helvetica", 13),  # Increased font size
            background="white",
            foreground="#1a1a1a",
        )  # Darker text

        # Configure button styles
        style.configure(
            "Predict.TButton",
            font=("Helvetica", 14, "bold"),  # Increased font size
            padding=10,
        )  # Increased padding

        style.configure(
            "Reset.TButton", font=("Helvetica", 13), padding=10  # Increased font size
        )  # Increased padding

    def create_header(self):
        header_frame = ttk.Frame(self.main_container, style="Header.TFrame")
        header_frame.pack(fill="x", pady=(0, 20))  # Increased padding

        title_label = ttk.Label(
            header_frame, text="Water Quality Analysis", style="Header.TLabel"
        )
        title_label.pack(pady=15)  # Increased padding

        self.status_label = ttk.Label(header_frame, text="", style="Status.TLabel")
        self.status_label.pack(pady=(0, 15))

    def create_input_section(self):
        # Create scrollable input frame
        input_canvas = tk.Canvas(self.main_container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            self.main_container, orient="vertical", command=input_canvas.yview
        )
        self.scrollable_frame = ttk.Frame(input_canvas, style="Input.TFrame")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: input_canvas.configure(scrollregion=input_canvas.bbox("all")),
        )

        input_canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw", width=740
        )  # Increased width
        input_canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the canvas and scrollbar
        input_canvas.pack(side="left", fill="both", expand=True, pady=(0, 20))
        scrollbar.pack(side="right", fill="y", pady=(0, 20))

        # Create input fields
        self.inputs = {}
        labels = [
            ("pH", "Enter pH value (0-14)"),
            ("Hardness", "Enter hardness value (mg/L)"),
            ("Solids", "Enter total dissolved solids (ppm)"),
            ("Chloramines", "Enter chloramines level (ppm)"),
            ("Sulfate", "Enter sulfate content (mg/L)"),
            ("Conductivity", "Enter conductivity (μS/cm)"),
            ("Organic Carbon", "Enter organic carbon content (ppm)"),
            ("Trihalomethanes", "Enter trihalomethanes level (μg/L)"),
            ("Turbidity", "Enter turbidity (NTU)"),
        ]

        for idx, (label, placeholder) in enumerate(labels):
            frame = ttk.Frame(self.scrollable_frame, style="Input.TFrame")
            frame.pack(fill="x", padx=30, pady=10)  # Increased padding

            lbl = ttk.Label(
                frame, text=label, style="Input.TLabel", width=25
            )  # Increased width
            lbl.pack(side="left", padx=(0, 15))  # Increased padding

            entry = ttk.Entry(
                frame, width=40, font=("Helvetica", 13)
            )  # Increased width and font size
            entry.insert(0, placeholder)
            entry.configure(foreground="gray")
            entry.bind(
                "<FocusIn>",
                lambda e, entry=entry, placeholder=placeholder: self.on_entry_click(
                    e, entry, placeholder
                ),
            )
            entry.bind(
                "<FocusOut>",
                lambda e, entry=entry, placeholder=placeholder: self.on_focus_out(
                    e, entry, placeholder
                ),
            )
            entry.pack(side="left", padx=5)

            self.inputs[label] = entry

        # Create buttons frame
        button_frame = ttk.Frame(self.scrollable_frame, style="Input.TFrame")
        button_frame.pack(fill="x", padx=30, pady=25)  # Increased padding

        # Add Predict and Reset buttons
        self.predict_button = ttk.Button(
            button_frame,
            text="Analyze Water Quality",
            command=self.run_prediction_thread,
            style="Predict.TButton",
        )
        self.predict_button.pack(side="left", padx=10)  # Increased padding

        self.reset_button = ttk.Button(
            button_frame,
            text="Reset Fields",
            command=self.reset_fields,
            style="Reset.TButton",
        )
        self.reset_button.pack(side="left", padx=10)  # Increased padding

    def create_results_section(self):
        self.results_frame = ttk.Frame(self.main_container, style="Input.TFrame")
        self.results_frame.pack(fill="x", pady=20)  # Increased padding

        # Results title
        results_title = ttk.Label(
            self.results_frame,
            text="Analysis Results",
            style="Header.TLabel",
            font=("Helvetica", 20, "bold"),
        )  # Increased font size
        results_title.pack(pady=15)

        # Create result display
        self.result_label = ttk.Label(
            self.results_frame,
            text="Awaiting analysis...",
            font=("Helvetica", 16),  # Increased font size
            background="white",
        )
        self.result_label.pack(pady=15)

        # Create timestamp label
        self.timestamp_label = ttk.Label(
            self.results_frame,
            text="",
            font=("Helvetica", 12),  # Increased font size
            foreground="#333333",
            background="white",
        )
        self.timestamp_label.pack(pady=(0, 15))

    def on_entry_click(self, event, entry, placeholder):
        if entry.get() == placeholder:
            entry.delete(0, "end")
            entry.configure(foreground="black")

    def on_focus_out(self, event, entry, placeholder):
        if entry.get() == "":
            entry.insert(0, placeholder)
            entry.configure(foreground="gray")

    def update_status(self, message):
        self.status_label.configure(text=message)

    def reset_fields(self):
        for label, entry in self.inputs.items():
            entry.delete(0, "end")
            entry.insert(0, f"Enter {label.lower()} value")
            entry.configure(foreground="gray")
        self.result_label.configure(text="Awaiting analysis...")
        self.timestamp_label.configure(text="")
        self.update_status("Fields reset")

    def load_model(self):
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name)

            if versions:
                latest_version = versions[0].version
                run_id = versions[0].run_id
                logged_model = f"runs:/{run_id}/{model_name}"
                loaded_model = mlflow.pyfunc.load_model(logged_model)
                self.update_status("Model loaded successfully")
                return loaded_model
            else:
                self.update_status("Error: No model found")
                return None
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}")
            return None

    def run_prediction_thread(self):
        self.predict_button.configure(state="disabled")
        self.update_status("Processing...")
        thread = threading.Thread(target=self.make_prediction)
        thread.start()

    def make_prediction(self):
        try:
            input_data = {}
            for key, entry in self.inputs.items():
                value = entry.get().strip()
                if value.startswith("Enter"):
                    self.after(0, lambda: self.show_error(f"{key} cannot be empty"))
                    return
                try:
                    input_data[key] = [float(value)]
                except ValueError:
                    self.after(0, lambda: self.show_error(f"Invalid value for {key}"))
                    return

            data = pd.DataFrame(input_data)
            column_mapping = {
                "pH": "ph",
                "Organic Carbon": "Organic_carbon",
            }
            data.rename(columns=column_mapping, inplace=True)

            if self.loaded_model is not None:
                prediction = self.loaded_model.predict(data)
                self.update_prediction_result(prediction[0])
            else:
                self.after(0, lambda: self.show_error("Model not loaded"))
        except Exception as e:
            self.after(0, lambda: self.show_error(f"Error during prediction: {str(e)}"))
        finally:
            self.after(0, lambda: self.predict_button.configure(state="normal"))
            self.after(0, lambda: self.update_status("Ready for prediction"))

    def update_prediction_result(self, prediction):
        result_text = "POTABLE" if prediction == 1 else "NOT POTABLE"
        result_color = "#4CAF50" if prediction == 1 else "#f44336"

        self.after(
            0,
            lambda: self.result_label.configure(
                text=f"Water Quality: {result_text}",
                foreground=result_color,
                font=("Helvetica", 20, "bold"),  # Increased font size
            ),
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.after(
            0,
            lambda: self.timestamp_label.configure(
                text=f"Analysis completed at: {timestamp}"
            ),
        )

    def show_error(self, message):
        messagebox.showerror("Error", message)
        self.predict_button.configure(state="normal")
        self.update_status("Error occurred")


if __name__ == "__main__":
    app = PredictionApp()
    app.mainloop()
