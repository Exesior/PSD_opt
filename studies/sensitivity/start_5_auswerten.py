"""
Spyder: Datei oeffnen, F5.  SCHRITT 5 -- auswerten (geht auch zwischendurch).

Sobol-Indizes ueber die fertigen Saltelli-Bloecke ("ZWISCHENSTAND", solange noch Bloecke
offen sind), Effektgroessen, Rauschen, Markierungen, Excel, Grafiken, bericht.md.
Ergebnisse in <Ergebnisordner>/analysis/.
"""

import analyze_sensitivity

if __name__ == "__main__":
    analyze_sensitivity.main(["--config", "sa_config_gesamt.py"])
