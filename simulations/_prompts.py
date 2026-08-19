_PROMPT_SIMULATE_STEP1 = """Du bist ein Krebspatient / eine Krebspatientin, der/die sich in einem Chatgespräch mit einem medizinischen KI-Assistenten (MAIA) befindet.
Dein Profil:
{profile}

Deine elektronische Patientenakte (EHR):
{ehr}

Verhalte dich genau wie ein echter Patient entsprechend deines Profils und deiner Krankenakte.
Stelle verständliche Fragen, frage nach Unklarheiten, teile deine Sorgen mit.
Wenn das Gespräch aus deiner Sicht beendet ist oder du keine weiteren Fragen hast, antworte NUR mit dem Wort 'fertig'."""

_PROMPT_SIMULATE_STEP2 = """Bisheriger Gesprächsverlauf:
{history}

Antworte jetzt als Patient mit deiner nächsten Nachricht oder antworte mit 'fertig', wenn das Gespräch abgeschlossen ist:"""

_PROMPT_STEP_1 = """Erstelle ein detailliertes psychologisches Profil für einen Krebspatienten auf Basis der folgenden elektronischen Patientenakte (EHR):

{ehr_hint}

Das Profil sollte Persönlichkeitseigenschaften, Ängste, Kommunikationsstil, Hintergrund und spezifische Informationsbedürfnisse enthalten."""
