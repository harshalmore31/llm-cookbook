from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
import time


# Initialize console
console = Console()

# Create and display a panel
console.print(Panel("Welcome to My App!", style="bold blue"))
ask = Prompt.ask("Enter the prompt ! ")
console.print("Prompt Initalized",style="bold green")
# Create and display a table
table = Table(show_header=True)
table.add_column("Name")
table.add_column("Value")
table.add_row("Item 1", "100")
table.add_row("Item 2", "200")
console.print(table)

# Show a progress spinner
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
) as progress:
    task = progress.add_task("Processing...", total=None)
    time.sleep(2)  # Simulate work
    progress.update(task, completed=True)

# Print styled text
console.print("✅ Done!", style="bold green")