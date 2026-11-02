"""
Spyder: Datei oeffnen, F5.  SCHRITT 8 -- Studie "ohne_bruchzwang_v1" auswerten (geht auch zwischendurch).

Wie start_5_auswerten.py, nur fuer sa_config_ohne_bruchzwang.py.
Ergebnisse in ergebnisse_ohne_bruchzwang_v1/analysis/.
"""

import analyze_sensitivity

if __name__ == "__main__":
    analyze_sensitivity.main(["--config", "sa_config_ohne_bruchzwang.py"])
