import argparse
import sys
from utils.io_utils import genereaza_matrice_aleatorie,citeste_matrice
from utils.backtracking import rezolva_tsp_backtracking


def main():
    parser = argparse.ArgumentParser(description="Proiect TSP - Backtracking cu multiple condiții de oprire.")

    # Argumente obligatorii și opționale
    parser.add_argument("sursa", help="Calea către fișierul .txt cu matricea sau un număr N pentru generare aleatorie")
    parser.add_argument("--mod", choices=['prima', 'toate', 'timp', 'y_solutii'], default='toate',
                        help="Modul de oprire pentru backtracking (implicit: toate)")
    parser.add_argument("--timp", type=int, help="Timpul maxim de execuție în secunde (pentru mod 'timp')")
    parser.add_argument("--y", type=int, help="Numărul de soluții de găsit (pentru mod 'y_solutii')")
    parser.add_argument("--seed", type=int, default=42, help="Seed pentru generarea aleatorie (implicit: 42)")

    args = parser.parse_args()

    # 1. Încărcarea datelor (matricea)
    try:
        # Încercăm să vedem dacă sursa este un număr (pentru generare random)
        if args.sursa.isdigit():
            n = int(args.sursa)
            print(f"[*] Generăm o matrice aleatorie de dimensiune {n}x{n}...")
            matrice = genereaza_matrice_aleatorie(n, args.seed)
        else:
            print(f"[*] Citim matricea din fișierul: {args.sursa}...")
            matrice = citeste_matrice(args.sursa)
            n = len(matrice)
    except Exception as e:
        print(f"[!] Eroare la încărcarea datelor: {e}")
        sys.exit(1)

    # 2. Validare parametri în funcție de mod
    if args.mod == 'timp' and args.timp is None:
        parser.error("Modul 'timp' necesită argumentul --timp (secunde).")
    if args.mod == 'y_solutii' and args.y is None:
        parser.error("Modul 'y_solutii' necesită argumentul --y (număr soluții).")

    # 3. Execuție Backtracking
    print(f"[*] Pornim algoritmul (Mod: {args.mod})...")
    print("-" * 40)

    path, cost, stats = rezolva_tsp_backtracking(
        n,
        matrice,
        mod=args.mod,
        timp_max=args.timp,
        y_max=args.y
    )

    # 4. Afișare rezultate
    if path:
        print(f"S-a găsit o soluție!")
        print(f"-> Drum: {' -> '.join(map(str, path))}")
        print(f"-> Cost total: {cost}")
    else:
        print("Nu a fost găsită nicio soluție validă în parametrii dați.")

    print("-" * 40)
    print(f"Statistici execuție:")
    print(f" - Timp real: {stats['timp_executie']:.4f} secunde")
    print(f" - Soluții explorate: {stats['solutii_gasite']}")


if __name__ == "__main__":
    main()