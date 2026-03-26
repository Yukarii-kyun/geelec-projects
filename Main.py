import sys
from Prediction import NutritionBrain  # Importing your logic module

def run_nutrition_app():
    # 1. Initialize the system
    try:
        brain = NutritionBrain()
        print("="*40)
        print("   NUTRITION DIAGNOSTIC SYSTEM v1.0")
        print("="*40)
        print("Note: Check 'model_report.png' for model logic.")
    except FileNotFoundError:
        print("Error: Model files not found. Please run train.py first!")
        return

    # 2. Main Input Loop
    while True:
        print("\n--- Enter New Patient Data (or type 'quit' to exit) ---")
        user_inputs = []
        
        try:
            # We loop through the features saved during training
            # This ensures we always ask for data in the correct order
            for feature in brain.features:
                val = input(f"Enter {feature}: ").strip().lower()
                
                if val == 'quit':
                    print("Exiting system...")
                    sys.exit()
                
                user_inputs.append(float(val))

            # 3. Send data to the predictor module
            result = brain.predict_single(user_inputs)
            
            # 4. Display Result
            print("\n" + "*"*30)
            print(f"ANALYSIS COMPLETE: {result}")
            print("*"*30)

        except ValueError:
            print("\n[!] Input Error: Please enter numbers only.")
        except Exception as e:
            print(f"\n[!] An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_nutrition_app()