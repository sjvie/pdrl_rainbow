# pdrl_rainbow

Struktur:
1. Netzwerk bauen (Duelling Struktur)
2. -> ersetzen durch Noisy Net (die Liniearen Layer)
3. Erstellen Experience Replay Buffer /Rauslöschen von alten/unwichtigen daten nicht vergessen
4. Bild Skalieren & preprocessen
5. Agenten erstellen wir model & target_model (+ Hyperparameter):
6. For Loop:
- Sieht die Umwelt: input Bild (state)
- Action: Q Value vom NN bestimmen lassen und mit argmax auswählen
- Action in Step Funktion
-> Reward + next_state
- Dinge abspeicher (für Experience Replay)
- state = next_state
Learnstep: 
- Batch vom Experience Replay erstellen
- predicten von future rewards (Multistep beachten)
- Erwartete Q Values berechnen (Multistep beachten)
- Lossfunktion mit Distributional RL (Gradient Descent)
- Backpropagation (Fit)
- Nach Update Period Target network updaten

Später:
Monitoring Zeug:
KLD, Action Distribution, Reward tracken, Loss mit tracken
