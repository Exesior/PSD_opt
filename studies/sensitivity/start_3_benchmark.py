"""
Spyder: Datei oeffnen, F5.  SCHRITT 3 -- beste Prozesszahl messen (~10 min auf dem Remote-Rechner).

Rechnet dieselben 36 Laeufe des Basispunkts mit 6, 9 und 12 Prozessen (6 Kerne /
12 logische Prozessoren) und empfiehlt die Prozesszahl. Die Empfehlung danach in
sa_gemeinsam.py bei WORKERS eintragen. Ergebnis auch in <Ergebnisordner>/benchmark/ergebnis.txt;
die Benchmark-Laeufe zaehlen nicht zur Studie.
"""

import run_sensitivity

if __name__ == "__main__":
    run_sensitivity.main(["--config", "sa_config_gesamt.py", "--benchmark-workers", "6", "9", "12"])
