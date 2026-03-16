import random
from simpleai.search import SearchProblem, hill_climbing_random_restarts

class TSPHillClimbing(SearchProblem):
    """
    Clasa care modelează Problema Comis-Voiajorului (TSP) ca o problemă 
    de căutare locală pentru biblioteca simpleai.
    
    Starea (state) este reprezentată printr-un tuplu de indici de orașe,
    indicând ordinea în care acestea sunt vizitate. Orașul de start (0) 
    este menținut pe prima poziție pentru a evita traseele redundante 
    (rotații ale aceluiași tur).
    """

    def __init__(self, initial_state, matrice):
        """
        Inițializează problema TSP pentru Hill Climbing.

        Args:
            initial_state (tuple): Traseul de pornire (ex: (0, 1, 2, 3)).
            matrice (list of list of int/float): Matricea de distanțe NxN.
        """
        super().__init__(initial_state)
        self.matrice = matrice
        self.n = len(matrice)

    def actions(self, state):
        """
        Generează și returnează lista de acțiuni posibile din starea curentă.
        
        O acțiune constă în interschimbarea (swap) a două orașe distincte din traseu.
        Primul oraș (index 0) nu este interschimbat, fiind păstrat ca punct de start fix.

        Args:
            state (tuple): Traseul curent.

        Returns:
            list: O listă de tupluri (i, j) reprezentând indicii care urmează a fi interschimbați.
        """
        actiuni = []
        # Generăm toate perechile de indici (i, j) pentru orașele de pe pozițiile 1 .. n-1
        for i in range(1, self.n):
            for j in range(i + 1, self.n):
                actiuni.append((i, j))
        return actiuni

    def result(self, state, action):
        """
        Aplică o acțiune asupra stării curente pentru a obține o stare nouă.

        Args:
            state (tuple): Traseul curent.
            action (tuple): Perechea de indici (i, j) care trebuie interschimbați.

        Returns:
            tuple: Un nou traseu (starea vecină) cu cele două orașe interschimbate.
        """
        i, j = action
        # Convertim tuplul în listă pentru a putea face modificarea
        stare_noua = list(state)
        # Efectuăm interschimbarea
        stare_noua[i], stare_noua[j] = stare_noua[j], stare_noua[i]
        # Returnăm noua stare sub formă de tuplu
        return tuple(stare_noua)

    def value(self, state):
        """
        Evaluează starea curentă. 
        
        Deoarece algoritmii din `simpleai` încearcă să MAXIMIZEZE această valoare,
        iar noi dorim să MINIMIZĂM costul traseului TSP, vom returna costul total 
        înmulțit cu -1. Astfel, cel mai mic cost va reprezenta cea mai mare valoare.

        Args:
            state (tuple): Traseul evaluat.

        Returns:
            int/float: Valoarea negativă a costului total al traseului.
        """
        cost = 0
        # Adunăm costurile trecerilor între orașele consecutive
        for i in range(self.n - 1):
            cost += self.matrice[state[i]][state[i + 1]]
        
        # Adăugăm costul de întoarcere la orașul de start pentru a închide turul
        cost += self.matrice[state[-1]][state[0]]
        
        return -cost

    def generate_random_state(self):
        """
        Generează o stare de pornire aleatoare.
        
        Această metodă este obligatorie pentru funcționarea algoritmului 
        `hill_climbing_random_restarts`, deoarece acesta are nevoie să genereze 
        puncte noi de plecare.

        Returns:
            tuple: Un nou traseu generat aleator, care începe întotdeauna cu orașul 0.
        """
        orase = list(range(1, self.n))
        random.shuffle(orase)
        return tuple([0] + orase)


def rezolva_tsp_hc(matrice, restarts_limit=20):
    """
    Rezolvă Problema Comis-Voiajorului folosind algoritmul Hill Climbing 
    cu reporniri aleatorii (Random Restarts).

    Args:
        matrice (list of list of int/float): Matricea de distanțe NxN.
        restarts_limit (int): Numărul maxim de reporniri aleatorii din puncte 
                              diferite pentru a evita blocarea în optimi locali.

    Returns:
        tuple: Un tuplu sub forma (traseu_optim, cost_minim), unde:
            - traseu_optim (list): Lista orașelor în ordinea optimă găsită.
            - cost_minim (int/float): Costul turului complet pentru traseul găsit.
    """
    n = len(matrice)
    if n <= 1:
        return ([0] if n == 1 else [], 0)

    # Starea inițială "de bază" (de ex. 0 -> 1 -> 2 -> ... -> n-1)
    stare_initiala = tuple(range(n))
    
    # Instanțiem problema 
    problema = TSPHillClimbing(stare_initiala, matrice)
    
    # Rulăm Hill Climbing Random Restarts
    # Această funcție va rula de mai multe ori și va apela automat
    # problema.generate_random_state() pentru fiecare nouă încercare.
    rezultat = hill_climbing_random_restarts(problema, restarts_limit=restarts_limit)
    
    # Rezultatul returnat conține 'state' (tuplul traseului optim găsit)
    traseu_optim = list(rezultat.state)
    
    # Valoarea obținută este negativă, o înmulțim cu -1 pentru a obține costul real
    cost_minim = -problema.value(rezultat.state)
    
    return traseu_optim, cost_minim