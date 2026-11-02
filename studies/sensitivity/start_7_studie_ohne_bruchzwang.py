"""
Spyder: Datei oeffnen, F5.  SCHRITT 7 -- Studie "ohne_bruchzwang_v1" starten bzw. fortsetzen.

Wie gesamt_v1, nur das Bruch-Niveau S_ref bleibt im Fit-Bereich (log10 -3.5 .. 0) statt in den
Bruch geschoben zu werden (sa_config_ohne_bruchzwang.py).
Ergebnisse in ergebnisse_ohne_bruchzwang_v1/ -- ergebnisse_gesamt_v1/ wird nicht angefasst.

Fertige Laeufe werden erkannt und uebersprungen; nach einem Abbruch einfach wieder F5.
Fortschritt in der Konsole und in ergebnisse_ohne_bruchzwang_v1/progress.log.
"""

import run_sensitivity

if __name__ == "__main__":
    run_sensitivity.main(["--config", "sa_config_ohne_bruchzwang.py"])
