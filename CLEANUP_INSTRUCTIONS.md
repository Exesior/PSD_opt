# Aufräumen: Tote Enden löschen

## 🗑️ Zu löschende Dateien/Ordner

### 1. Alte Test-Dateien (Hauptverzeichnis)
```
C:\Users\ericb\Documents\GitHub\PSD_opt\test_liquid_mass_conservation.py  ← LÖSCHEN
```

### 2. Alte Trial-Ordner (granulation)
```
C:\Users\ericb\Documents\GitHub\PSD_opt\scripts\pbe_validation\granulation\Trials\  ← KOMPLETT LÖSCHEN
```

### 3. Backup-Referenzen (falls noch vorhanden)
```
C:\Users\ericb\Documents\GitHub\PSD_opt - backup\  ← Nur wenn nicht mehr benötigt
```

---

## ✅ Behalten (aktuelle Struktur)

### WMCPBE Source (modifiziert)
```
C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src\wmcpbe\
├── mcpbe_base.py           ← ✓ BEHALTEN (mit liquid_volume Arrays)
├── mcpbe_agg.py            ← ✓ BEHALTEN (mit Masseerhaltung)
├── mcpbe_break.py          ← ✓ BEHALTEN (mit Masseerhaltung)
├── mcpbe.py                ← ✓ BEHALTEN
└── Trials/                 ← ✓ NEU, BEHALTEN
    ├── __init__.py
    └── test_liquid_conservation.py
```

---

## 🔧 PowerShell Befehle zum Löschen

### 1. Altes Testfile im Hauptverzeichnis löschen
```powershell
Remove-Item "C:\Users\ericb\Documents\GitHub\PSD_opt\test_liquid_mass_conservation.py" -Force
```

### 2. Alten Trials Ordner löschen
```powershell
Remove-Item "C:\Users\ericb\Documents\GitHub\PSD_opt\scripts\pbe_validation\granulation\Trials" -Recurse -Force
```

### 3. Diese Aufräum-Anleitung löschen (nach dem Aufräumen)
```powershell
Remove-Item "C:\Users\ericb\Documents\GitHub\PSD_opt\CLEANUP_INSTRUCTIONS.md" -Force
```

---

## 📋 Zusammenfassung der aktuellen Struktur

```
PSD_opt/
├── mcpbe/src/wmcpbe/
│   ├── mcpbe_base.py          ← liquid_volume, porosity, saturation hinzugefügt
│   ├── mcpbe_agg.py           ← Liquid mass conservation in Agglomeration
│   ├── mcpbe_break.py         ← Liquid mass conservation in Breakage
│   ├── mcpbe.py               ← Haupt-Solver
│   └── Trials/                ← NEU: Test-Ordner
│       ├── __init__.py
│       └── test_liquid_conservation.py  ← AKTUELLER TEST
│
├── scripts/pbe_validation/granulation/
│   ├── validation.py          ← BEHALTEN (wird anderweitig verwendet)
│   └── simple_validation.py   ← BEHALTEN (wird anderweitig verwendet)
│
└── CLEANUP_INSTRUCTIONS.md    ← Diese Datei (nachher löschen)
```

---

## ▶️ Nächste Schritte

1. **Löschen** (Befehle oben ausführen)
2. **Testen**: 
   ```powershell
   cd C:\Users\ericb\Documents\GitHub\PSD_opt\mcpbe\src
   python -m wmcpbe.Trials.test_liquid_conservation
   ```
3. **Diese Datei löschen**: `CLEANUP_INSTRUCTIONS.md`
