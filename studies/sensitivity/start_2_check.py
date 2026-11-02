"""
Spyder: Datei oeffnen, F5.  SCHRITT 2 -- Umgebung pruefen (Sekunden).

Importe, SALib, Config, Solver bauen + 200 Events, und: derselbe Simulationscode wie auf
dem Laptop (code_referenz.json)? Am Ende muss "CHECK OK" stehen; bei "CHECK NICHT OK"
nennt es die abweichenden Dateien (README, "Auf dem Remote-Rechner", Schritt 0).
"""

import run_sensitivity

if __name__ == "__main__":
    run_sensitivity.main(["--config", "sa_config_gesamt.py", "--check"])
