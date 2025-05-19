import os
from mistralai import Mistral
from datetime import datetime

class CodeAssistant:
    def __init__(self):
        self.api_key = "G4M4d2Ju73R4wmV1xsVsAea47cbDlERo"
        self.client = Mistral(api_key=self.api_key)
        self.model = "codestral-latest"
        self.history = []

    def generate_code(self, prompt, suffix=""):
        response = self.client.fim.complete(
            model=self.model,
            prompt=prompt,
            suffix=suffix,
            temperature=0,
            top_p=1,
        )
        
        result = {
            'timestamp': datetime.now(),
            'prompt': prompt,
            'suffix': suffix,
            'response': response.choices[0].message.content
        }
        
        self.history.append(result)
        return result

    def show_history(self):
        for i, entry in enumerate(self.history, 1):
            print(f"\nEntry {i}:")
            print(f"Time: {entry['timestamp']}")
            print(f"Prompt: {entry['prompt']}")
            print(f"Suffix: {entry['suffix']}")
            print("Generated Code:")
            print(entry['response'])
            print("-" * 50)

def main():
    assistant = CodeAssistant()
    
    while True:
        print("\n1. Generate Code")
        print("2. View History")
        print("3. Exit")
        
        choice = input("Choose an option (1-3): ")
        
        if choice == "1":
            prompt = input("Enter your code prompt: ")
            use_suffix = input("Do you want to add a suffix? (y/n): ").lower()
            
            suffix = ""
            if use_suffix == 'y':
                suffix = input("Enter your code suffix: ")
            
            result = assistant.generate_code(prompt, suffix)
            print("\nGenerated Code:")
            print(f"{prompt}")
            print(f"{result['response']}")
            if suffix:
                print(f"{suffix}")
                
        elif choice == "2":
            assistant.show_history()
            
        elif choice == "3":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()