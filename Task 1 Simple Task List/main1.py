import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Initialize the tasks dataframe
tasks = pd.DataFrame(columns=['description', 'priority'])
csv_path = 'tasks.csv'
model = None

# Function to save tasks to a CSV file
def save_tasks():
    tasks.to_csv(csv_path, index=False)

# Function to load tasks from a CSV file
def load_tasks():
    global tasks
    if os.path.exists(csv_path):
        tasks = pd.read_csv(csv_path)
        print(f"Loaded {len(tasks)} tasks from {csv_path}")
    else:
        save_tasks()

# Function to add a task
def add_task(description, priority):
    global tasks
    
    # Convert priority to lowercase for case-insensitive comparison
    priority_lower = priority.lower()
    
    # Check if the priority input is valid
    if priority_lower in ['low', 'medium', 'high']:
        # Convert priority to title case for consistency
        priority = priority_lower.capitalize()
        
        # Check if the task already exists
        if tasks['description'].str.lower().eq(description.lower()).any():
            print("Task with the same description already exists.")
        else:
            # If the task does not exist, add it
            new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
            tasks = pd.concat([tasks, new_task], ignore_index=True)
            
            # Sort tasks by priority (high to low)
            tasks.sort_values(by='priority', ascending=False, inplace=True)
            
            # Save tasks to CSV
            save_tasks()
            
            tasks.reset_index(drop=True, inplace=True)  # Reset index
            print("Task added successfully.")
    else:
        print("Invalid priority. Please enter 'Low', 'Medium', or 'High'.")


# Function to remove a task
def remove_task(description):
    global tasks
    if description in tasks['description'].values:
        tasks = tasks[tasks['description'] != description]
        save_tasks()
        print("Task removed successfully.")
    else:
        print("Task not found.")
   

# Function to list all tasks
def list_tasks():
    print(tasks)

# Function to train the model
def train_model():
    global model
    try:
        # Ensure there are tasks to train on
        if len(tasks) == 0:
            print("No tasks available to train the model.")
            return

        # Ensure tasks contain non-stopword descriptions
        vectorizer = CountVectorizer(stop_words='english', min_df=1, token_pattern=r'\b\w+\b')
        if not any(vectorizer.build_analyzer()(desc) for desc in tasks['description']):
            print("Error training model: all descriptions are too short or contain only stop words.")
            return

        # Train the task priority classifier
        clf = MultinomialNB()
        model = make_pipeline(vectorizer, clf)
        model.fit(tasks['description'].astype(str), tasks['priority'])
        print("Model trained successfully.")
    except ValueError as e:
        print(f"Error training model: {e}")

# Function to recommend a task
def recommend_task():
    try:
        if model is None:
            print("Model is not trained yet. Please train the model first.")
            return

        # Predict priorities for existing tasks
        predictions = model.predict(tasks['description'])
        high_priority_tasks = tasks[tasks['priority'] == 'High']
        if not high_priority_tasks.empty:
            recommended_task = high_priority_tasks.sample(n=1).iloc[0]
            print(f"Recommended task: {recommended_task['description']} - Priority: {recommended_task['priority']}")
        else:
            print("No high-priority tasks to recommend.")
    except Exception as e:
        print(f"Error recommending task: {e}")

# Main program loop
def main():
    load_tasks()

    while True:
        print("\nTask Management App")
        print("1. Add Task")
        print("2. Remove Task")
        print("3. List Tasks")
        print("4. Recommend Task")
        print("5. Train Model")
        print("6. Exit")
        option = input("Select an option: ")

        if option == '1':
            description = input("Enter task description: ")
            priority = input("Enter task priority (Low/Medium/High): ").capitalize()
            add_task(description, priority)

        elif option == '2':
            description = input("Enter task description to remove: ")
            remove_task(description)
        elif option == '3':
            list_tasks()
        elif option == '4':
            recommend_task()
        elif option == '5':
            train_model()
        elif option == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please select a valid option.")

if __name__ == "__main__":
    main()
