from pywinauto.application import Application
import time

# Start Notepad
app = Application().start("notepad.exe")
time.sleep(2)  # Wait for Notepad to fully load

# Write content to Notepad
app.UntitledNotepad.Edit.type_keys("{ENTER}", with_spaces=True)
app.UntitledNotepad.Edit.type_keys("pywinauto Works!", with_spaces=True)
app.UntitledNotepad.Edit.type_keys("{ENTER}", with_spaces=True)
app.UntitledNotepad.Edit.type_keys("Here is a C++ code for Fibonacci series:", with_spaces=True)
app.UntitledNotepad.Edit.type_keys("{ENTER}{ENTER}", with_spaces=True)
cpp_code = '''#include <iostream>
using namespace std;

int main() {
    int n, t1 = 0, t2 = 1, nextTerm = 0;

    cout << "Enter the number of terms: ";
    cin >> n;

    cout << "Fibonacci Series: " << t1 << ", " << t2 << ", ";

    nextTerm = t1 + t2;

    for (int i = 3; i <= n; ++i) {
        cout << nextTerm << ", ";
        t1 = t2;
        t2 = nextTerm;
        nextTerm = t1 + t2;
    }
    return 0;
}
'''
app.UntitledNotepad.Edit.type_keys(cpp_code, with_spaces=True)

# Save the file as test.cpp in C drive
app.UntitledNotepad.menu_select("File->Save As")
time.sleep(1)  # Wait for the Save As dialog to open
app.SaveAs.Edit.set_edit_text("C:\\test.cpp")
app.SaveAs.Save.click()
time.sleep(1)  # Wait for the file to save

# Close Notepad
app.UntitledNotepad.menu_select("File->Exit")
time.sleep(1)  # Wait for Notepad to close

# Run the file in cmd
import os
os.system('cmd /c "g++ C:\\test.cpp -o C:\\test && C:\\test"')