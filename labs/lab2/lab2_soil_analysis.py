import pandas as pd

def load_data():
    
    # Load the soil test dataset
    try:
        df = pd.read_csv('../../datasets/soil_test.csv')
        # Read all the rows
        pd.set_option('display.max_rows', None)
    except FileNotFoundError:
        print("Error: The dataset file was not found. Please ensure 'soil_test.csv' is located in the /datasets/ folder.")
        return
        
    print(df)



def clean_data():
    
    # Load the soil test dataset
    try:
        data = pd.read_csv('../../datasets/soil_test.csv')
        # Read all the rows
        pd.set_option('display.max_rows', None)
    except FileNotFoundError:
        print("Error: The dataset file was not found. Please ensure 'soil_test.csv' is located in the /datasets/ folder.")
        return
        

    # Replace non existent numbers with means
    data.fillna(data.mean(numeric_only=True), inplace=True)
    data = data.round(1)
    print(data)


def compute_statistics():
    
    # Load the soil test dataset
    try:
        data = pd.read_csv('../../datasets/soil_test.csv')
        # Read all the rows
        pd.set_option('display.max_rows', None)
    except FileNotFoundError:
        print("Error: The dataset file was not found. Please ensure 'soil_test.csv' is located in the /datasets/ folder.")
        return
    
    # Compute the statistcs
    maximum = data["nitrogen"].max()
    minimum = data["nitrogen"].min()
    mean = data["nitrogen"].mean()
    median = data["nitrogen"].median()
    std = data["nitrogen"].std()
    
    # Print the results
    print("\nNitrogen Data Analysis:")
    print(f"Minimum nitrogen value: {minimum}")
    print(f"Maximum nitrogen value: {maximum}")
    print(f"Mean nitrogen value: {mean:.1f}")
    print(f"Median of the nitrogen values: {median:.1f}")
    print(f"Standart deviation of the nitrogen values: {mean:.1f}")
    

while True:
    print("\n--- Soil Data Analysis Menu ---")
    print("1. Load Soil Data")
    print("2. Clean Data (Replace Missing Values)")
    print("3. Show Soil Statistics")
    print("Type 'exit' to quit")

    choice = input("Enter your choice: ").strip()

    if choice == "1":
        load_data()
    elif choice == "2":
        clean_data()
    elif choice == "3":
        compute_statistics()
    elif choice.lower() == "exit":
        print("\nExiting program. Goodbye!\n")
        break
    else:
        print("\nInvalid choice. Please select a valid option.\n")
    