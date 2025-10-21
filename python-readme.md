# Python Projekt Setup

## 1. Einmaliges Setup

Stelle sicher, dass du eine unterstützte Python-Version (3.9–3.12) verwendest. Du kannst hierfür z. B. `pyenv` oder `conda` nutzen.

### Beispiel mit pyenv
```bash
pyenv install 3.12.7
pyenv local 3.12.7
```

### Virtuelle Umgebung anlegen
```bash
python -m venv .venv
```

### Umgebung aktivieren
Mac/Linux:
```bash
source .venv/bin/activate
```
Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```
Windows (CMD):
```bat
.venv\Scripts\activate.bat
```

### Abhängigkeiten installieren
```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## 2. Arbeiten im Projekt (tägliche Nutzung)
Jedes Mal, bevor du mit dem Code arbeitest, aktiviere die virtuelle Umgebung:

Mac/Linux:
```bash
source .venv/bin/activate
```
Windows:
```powershell
.venv\Scripts\Activate.ps1
```

Falls du das Projektverzeichnis noch nicht geöffnet hast:
```bash
cd LogLead
```

### Beispiele zum Starten von Skripten
```bash
python demo/HDFS_samples.py
python quickstart.py  # (falls vorhanden)
```

## 3. Deaktivieren der Umgebung
Wenn du fertig bist:
```bash
deactivate
```

## 4. Häufige Probleme
- "command not found": Prüfe, ob die virtuelle Umgebung aktiv ist.
- Falsche Python-Version: `python --version` ausführen und ggf. `pyenv local` neu setzen.
- Änderungen an Abhängigkeiten: Nach Anpassungen an `pyproject.toml` erneut `pip install -e .` ausführen.

## 6. Kurzübersicht
1. Python-Version setzen
2. Virtuelle Umgebung erstellen
3. Aktivieren
4. Abhängigkeiten installieren
5. Skripte ausführen
6. Deaktivieren