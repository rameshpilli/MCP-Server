import requests
import json
from typing import Dict, Any, Optional
import sys

class MCPClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "test_key"):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }

    def execute_task(self, task_type: str, parameters: Dict[str, Any] = None) -> Dict:
        """Execute a task on the MCP server."""
        if parameters is None:
            parameters = {}

        payload = {
            "task_type": task_type,
            "parameters": parameters
        }

        try:
            response = requests.post(
                f"{self.base_url}/execute-task",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"\nError executing task '{task_type}': {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    print("Error details:", json.dumps(error_details, indent=2))
                except json.JSONDecodeError:
                    print("Error response:", e.response.text)
            return None

def print_response(response: Optional[Dict]):
    """Pretty print the response."""
    if response is None:
        return
    
    print("\nResponse:")
    print("Status:", response.get("status"))
    print("Timestamp:", response.get("timestamp"))
    print("Cached:", response.get("cached", False))
    print("\nData:")
    print(json.dumps(response.get("data", {}), indent=2))
    print("-" * 50)

def show_menu():
    """Show the available tasks menu."""
    print("\nAvailable Tasks:")
    print("\nExternal APIs:")
    print("1. Get a random programming joke")
    print("2. Get cryptocurrency prices")
    print("3. Get a random activity")
    print("4. Get an inspirational quote")
    print("5. Get a random dog image")
    print("6. Get a random cat fact")
    print("7. Get exchange rates")
    
    print("\nFile Operations:")
    print("8. List directory contents")
    print("9. Get file information")
    print("10. Read file contents")
    print("11. Search files")
    print("12. Calculate file hash")
    
    print("\n0. Exit")

def get_task_details(choice: int) -> tuple[str, Dict[str, Any]]:
    """Get task type and parameters based on user choice."""
    if choice == 1:
        return "get_joke", {}
    elif choice == 2:
        return "get_crypto_prices", {}
    elif choice == 3:
        return "get_activity", {}
    elif choice == 4:
        return "get_quote", {}
    elif choice == 5:
        return "get_dog_image", {}
    elif choice == 6:
        return "get_cat_fact", {}
    elif choice == 7:
        base = input("Enter base currency (default: USD): ").strip() or "USD"
        return "get_exchange_rates", {"base": base}
    elif choice == 8:
        path = input("Enter directory path (default: .): ").strip() or "."
        return "list_directory", {"path": path}
    elif choice == 9:
        path = input("Enter file path: ").strip()
        return "file_info", {"file_path": path}
    elif choice == 10:
        path = input("Enter file path: ").strip()
        return "read_file", {"file_path": path}
    elif choice == 11:
        base_dir = input("Enter base directory (default: .): ").strip() or "."
        pattern = input("Enter search pattern (default: *): ").strip() or "*"
        recursive = input("Search recursively? (y/n, default: y): ").strip().lower() != "n"
        return "search_files", {
            "base_dir": base_dir,
            "pattern": pattern,
            "recursive": recursive
        }
    elif choice == 12:
        path = input("Enter file path: ").strip()
        hash_type = input("Enter hash type (md5/sha1/sha256, default: sha256): ").strip() or "sha256"
        return "calculate_hash", {
            "file_path": path,
            "hash_type": hash_type
        }
    else:
        return None, {}

def interactive_mode():
    """Run the client in interactive mode."""
    client = MCPClient()
    
    while True:
        show_menu()
        try:
            choice = int(input("\nEnter your choice (0-12): "))
            if choice == 0:
                print("Goodbye!")
                break
            
            if 1 <= choice <= 12:
                task_type, parameters = get_task_details(choice)
                if task_type:
                    response = client.execute_task(task_type, parameters)
                    print_response(response)
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        
        input("\nPress Enter to continue...")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run test examples
        client = MCPClient()
        
        print("\nRunning test examples...")
        
        examples = [
            ("list_directory", {"path": "."}),
            ("file_info", {"file_path": "client.py"}),
            ("get_joke", {}),
            ("get_crypto_prices", {}),
            ("get_activity", {})
        ]
        
        for task_type, parameters in examples:
            print(f"\nTesting {task_type}...")
            response = client.execute_task(task_type, parameters)
            print_response(response)
    else:
        # Run interactive mode
        interactive_mode()

if __name__ == "__main__":
    main() 