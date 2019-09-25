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


gen_negative_substituted.py
===

Skrypt do generowania przykładów negatywnych. Przyjmuje plik w formacie wynikowym 
gen_positive.py (idx1:ctx1  idx2:ctx2). Tworzy słownik marka:produkt dla wszystkich 
marek występujących w podanym pliku. Następnie w zależności od trybu podmienia w zdaniach markę lub produkt
w taki sposób by dla występującej w zdaniu marki podać produkty kórych ta marka nie produkuje oraz analogicznie 
zmienia marki w które nie pasują do podanego w zdaniu produktu.  