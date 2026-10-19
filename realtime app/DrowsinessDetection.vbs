Set oShell = CreateObject("WScript.Shell")
oShell.Run "cmd /c cd /d """ & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & """ && python app.py", 0, False
