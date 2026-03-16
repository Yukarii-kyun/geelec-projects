class FoodAnalyzer:
    def __init__(self):
        # Thresholds per 100g (You can adjust these based on guidelines)
        self.thresholds = {
            'sugar_max': 10,       # grams
            'saturated_fat_max': 3, # grams
            'fiber_min': 3,        # grams
            'protein_min': 5,      # grams
            'sodium_max': 400      # mg
        }

    def analyze(self, food_name, nutrition_data):
        """
        nutrition_data: Dictionary with keys like 'sugar', 'fat', 'fiber', etc.
        """
        score = 100
        reasons = []

        # 1. Check Sugar
        sugar = nutrition_data.get('sugar', 0)
        if sugar > self.thresholds['sugar_max']:
            deduction = (sugar - self.thresholds['sugar_max']) * 2
            score -= deduction
            reasons.append(f"High sugar ({sugar}g)")

        # 2. Check Saturated Fat
        sat_fat = nutrition_data.get('saturated_fat', 0)
        if sat_fat > self.thresholds['saturated_fat_max']:
            deduction = (sat_fat - self.thresholds['saturated_fat_max']) * 3
            score -= deduction
            reasons.append(f"High saturated fat ({sat_fat}g)")

        # 3. Check Fiber (Bonus points)
        fiber = nutrition_data.get('fiber', 0)
        if fiber >= self.thresholds['fiber_min']:
            score += 5
            reasons.append(f"Good fiber content ({fiber}g)")

        # 4. Check Sodium
        sodium = nutrition_data.get('sodium', 0)
        if sodium > self.thresholds['sodium_max']:
            deduction = (sodium - self.thresholds['sodium_max']) / 10
            score -= deduction
            reasons.append(f"High sodium ({sodium}mg)")

        # 5. Check Protein (Bonus points)
        protein = nutrition_data.get('protein', 0)
        if protein >= self.thresholds['protein_min']:
            score += 5
            reasons.append(f"Good protein content ({protein}g)")

        # Cap score between 0 and 100
        score = max(0, min(100, score))

        # Determine Label
        if score >= 80:
            label = "Healthy"
        elif score >= 50:
            label = "Moderate"
        else:
            label = "Unhealthy"

        return {
            "food": food_name,
            "score": round(score, 1),
            "label": label,
            "reasons": reasons
        }

# --- Example Usage ---
if __name__ == "__main__":
    analyzer = FoodAnalyzer()

    # Example 1: Apple
    apple_data = {
        "sugar": 10,
        "saturated_fat": 0,
        "fiber": 4,
        "sodium": 1,
        "protein": 0.3
    }
    print(analyzer.analyze("Apple", apple_data))

    # Example 2: Chocolate Bar
    chocolate_data = {
        "sugar": 50,
        "saturated_fat": 15,
        "fiber": 2,
        "sodium": 20,
        "protein": 5
    }
    print(analyzer.analyze("Chocolate Bar", chocolate_data))
    
    # Example 3: Broccoli
    broccoli_data = {
        "sugar": 1,
        "saturated_fat": 0,
        "fiber": 5,
        "sodium": 30,
        "protein": 3
    }
    print(analyzer.analyze("Broccoli", broccoli_data))