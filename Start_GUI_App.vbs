Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "venv\Scripts\pythonw.exe gui_app.py", 0
Set WshShell = Nothing
