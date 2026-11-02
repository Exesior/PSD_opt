"""
Spyder: Datei oeffnen, F5.  SCHRITT 4 -- Studie starten bzw. fortsetzen.

Fertige Laeufe werden erkannt und uebersprungen; nach einem Abbruch einfach wieder F5.
Fortschritt in der Konsole und in <Ergebnisordner>/progress.log, z. B.
    Punkt   283: 5/5 abgeschlossen
"""

import run_sensitivity

if __name__ == "__main__":
    run_sensitivity.main(["--config", "sa_config_gesamt.py"])
