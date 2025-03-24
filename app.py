import customtkinter as ctk
import numpy as np
import joblib
from tkinter import messagebox

class HousePricePredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("House Price Predictor")
        self.geometry("600x500")
        
        # Load the model
        try:
            self.model = joblib.load("house_price_model.pkl")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.destroy()
            return
        
        # Create input fields
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title = ctk.CTkLabel(self, text="House Price Prediction", font=("Arial", 20, "bold"))
        title.pack(pady=20)
        
        # Input frame
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Input fields
        labels = ["Overall Quality (1-10)", "Living Area (sq ft)", 
                 "Garage Cars Capacity", "Basement Area (sq ft)"]
        self.entries = {}
        
        for i, label in enumerate(labels):
            ctk.CTkLabel(input_frame, text=label).pack(pady=5)
            self.entries[i] = ctk.CTkEntry(input_frame, placeholder_text=f"Enter {label}")
            self.entries[i].pack(pady=5)
        
        # Predict button
        predict_btn = ctk.CTkButton(self, text="Predict Price", command=self.predict_price)
        predict_btn.pack(pady=20)
        
    def predict_price(self):
        try:
            # Get input values
            features = [
                int(self.entries[0].get()),      # Overall Quality
                float(self.entries[1].get()),    # Living Area
                int(self.entries[2].get()),      # Garage Cars
                float(self.entries[3].get())     # Basement Area
            ]
            
            # Make prediction
            input_features = np.array([features])
            prediction = self.model.predict(input_features)
            
            # Show result
            messagebox.showinfo(
                "Prediction Result",
                f"The predicted house price is: ${prediction[0]:,.2f}"
            )
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = HousePricePredictorApp()
    app.mainloop()