gen_positive.py
===

Skrypt do generowania przykładów pozytywnych z anotowanych korpusów. 
Wybiera pary dla których zachodzi relacja marka - produkt. 
Zwraca indeks pierwszego słowa z nazwy marki lub nazwy produktu oraz kontekst, 
czyli wszystkie tokeny w zdaniu w którym dana marka lub produkt wystepują.


gen_negative.py
===

Skrypt do generowania przykładów negatywnych z anotowanych korpusów.
Wybiera pary dla których zachodzi relacja marka - produkt oraz 
wszystkie rzeczowniki w zdaniu nie będące cześcią nazwy marki lub produktu.
Zwraca indeksy oraz kontekst tak jak _gen_positive.py_ dla każdej marki z wybranymi rzeczownikami, 
dla każdego produktu z wybranymi rzeczownikami, oraz dla wybranych rzeczwoników. 
Warunkiem wyboru jest odległość słów wynosząca nie więcej niż 3.

