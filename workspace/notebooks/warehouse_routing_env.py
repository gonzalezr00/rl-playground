
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class WarehouseRoutingEnv(gym.Env):
    """
    Clase que representa l'entorn WarehouseRoutingEnv. 
    Hereda de gym.Env i implementa els metodes necessaris per a la interacció amb l'entorn.
    Conte metodes per a inicialitzar l'entorn, resetear-lo, realitzar accions, i obtenir observacions i informació sobre l'estat actual de l'entorn.
    """

    SAFE_ROUTE = 0
    EXPRESS_ROUTE = 1

    def __init__(self, priority_bonus=False):
        """
        Inicialitza l'entorn, definint l'espaci d'accions i l'espai d'observacions, utilitzant la clase "spaces" de la llibreria gymnasium.
        Especificament, per al action_space s'utilitza spaces.Discrete(2) ja que hi ha dues accions possibles (SAFE_ROUTE i EXPRESS_ROUTE).
        Per a l'observation_space s'utilitza spaces.Tuple amb tres espais Discrete, representant el treball restant, la congestió i la disponibilitat de reserves.
        A més, s'inicialitzen les variables d'estat de l'entorn: remaining_work, congestion i reserve_available, que representen el treball restant, la congestió i la disponibilitat de reserves respectivament.
        Per a les variable d'estat, nomes es defineixen valors inicials que porten el seu tipus de dades, ja que seran actualitzades a valors contextuals en el metode reset() i step().
        També s'accepta un paràmetre priority_bonus que es converteix en un atribut de la classe i per defecte es False. Aquest paràmetre s'utilitza per determinar si es concedeix un bonus de prioritat en cas d'èxit amb l'acció EXPRESS_ROUTE.
        """
        super().__init__()

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Tuple((
            spaces.Discrete(22),
            spaces.Discrete(11),
            spaces.Discrete(2)
        ))

        self.priority_bonus = priority_bonus

        self.remaining_work = 0
        self.congestion = 0
        self.reserve_available = False

    def reset(self, *, seed=None, options=None):
        """
        Reseteja l'entorn a un estat inicial aleatori, que sigui coherent amb les regles de l'entorn, rang de valors permesos representat per [inicial, final].
        Es genera un valor aleatori per a cada variable d'estat: remaining_work [12, 21], congestion [0, 10] i reserve_available [0, 1].
        S'acepta un paràmetre seed per a la inicialització del generador de nombres aleatoris, que es passa al metode reset() de la clase base gym.Env, amb l'objectiu de garantir la reproducibilitat dels resultats.
        Tambe es pot passar un paràmetre options, que no s'utilitza en aquest entorn, però es manté per compatibilitat amb la interfície de gymnasium.
        Es retorna una tupla amb l'observació inicial i un diccionari amb informació addicional sobre l'estat inicial de l'entorn, resultats dels metodes _get_obs() i _get_info() amb outcome=None, respectivament.
        """
        super().reset(seed=seed)

        self.remaining_work = int(self.np_random.integers(12, 22))
        self.congestion = int(self.np_random.integers(0, 11))
        self.reserve_available = bool(self.np_random.integers(0, 2))

        return self._get_obs(), self._get_info()

    def _action_parameters(self, action):
        """
        Defineix els paràmetres de probabilitat i els valors associats a cada acció (SAFE_ROUTE o EXPRESS_ROUTE) en funció de l'estat actual de l'entorn.
        Retorna un diccionari amb els paràmetres de probabilitat i els valors associats a l'acció seleccionada, que seran utilitzats en el metode step() per determinar l'efecte de l'acció sobre l'estat de l'entorn
        Per a cada acció, es defineixen les següents probabilitats i valors:
        - incident_probability: probabilitat d'incident que provoca la finalització de l'episodi amb un resultat negatiu, que depend de la congestió i la disponibilitat de reserves.
        - delay_probability: probabilitat de retard que depend del trevall pendent.
        - progress_values: array amb els possible valors de progres que es restaran del travaill pendent, per episodi, en cas de no pas tenir un incident.
        - progress_probabilities: probabilitats de cadascu dels valors de progres possibles.
        - reserve_loss_probability: probabilitat de perdua de la reserva operativa, que depen de la congestió
        - congestion_changes: possibilitats de canvis en la congestio
        - congestion_probabilities: probabilitats de aquest canvis en la congestio

        """

        r = self.remaining_work
        c = self.congestion
        b = self.reserve_available

        if action == self.SAFE_ROUTE:

            incident_probability = min(
                0.30,
                0.002 + 0.002 * c + (0.010 if not b else 0.0)
            )

            delay_probability = min(
                0.40,
                0.025 + 0.003 * r
            )

            progress_values = np.array([1, 2])
            progress_probabilities = np.array([0.40, 0.60])

            # Min 0.01
            reserve_loss_probability = (
                0.010 + 0.001 * c
                if b else 0.0
            )

            congestion_changes = np.array([-1, 0, 1])
            congestion_probabilities = np.array([0.45, 0.45, 0.10])

        else:

            incident_probability = min(
                0.40,
                0.006 + 0.008 * c + (0.025 if not b else 0.0)
            )

            delay_probability = min(
                0.15,
                0.002 + 0.0003 * r
            )

            progress_values = np.array([2, 3, 4])
            progress_probabilities = np.array([0.30, 0.50, 0.20])

            reserve_loss_probability = (
                0.070 + 0.008 * c
                if b else 0.0
            )

            congestion_changes = np.array([-1, 0, 1])
            congestion_probabilities = np.array([0.10, 0.45, 0.45])

        return {
            "incident_probability": incident_probability,
            "delay_probability": delay_probability,
            "progress_values": progress_values,
            "progress_probabilities": progress_probabilities,
            "reserve_loss_probability": reserve_loss_probability,
            "congestion_changes": congestion_changes,
            "congestion_probabilities": congestion_probabilities,
        }

    def step(self, action):
        """
        Executa una acció en l'entorn i actualitza l'estat de l'entorn en conseqüència.
        Es comprova que l'acció sigui vàlida dins de l'espai d'accions definit, mitjançant el metode contains() de l'espai d'accions, heredat de spaces.Discrete.
        Es calculen els paràmetres de probabilitat (mitjançant els processos estocàstics) i els valors associats a l'acció seleccionada mitjançant el metode _action_parameters().
        Es genera un event estocastic (nombre aleatori) per determinar si es produeix un incident, un retard o un progrés en el treball pendent, segons les probabilitats definides.
        Si aquest nombre es menor que la probabilitat de incident, es considera com si un inciden hages tingut lloc, i es restorna una recompensa de -1 y es considera com terminat.
        Si aquest nombre es mes gran o igual que la probabilitat de incident pero menor que l'adicio de la probabilitat de retard i la de incident, es considera com si fos un retard, i es perd la finestra d'expedició, es considera com terminat.
        Si aquest nombre es mes gran o igual que la probabilitat de incident i la de retard, es considera com si fos un progrés estocastic, donat pels parametres de progress asignats a la accio seleccionada.
        i se ajusta el travaill pendent en conseqüència, si el progress es igual o mes gran que el travaill pendent anterior, es considera que el travaill restant es 0.
        Si hi ha una reserva operativa i el nombre de aquest event estocastic es mes petit que la probabilitat de reserva perduda, es considera com si fos una perdua de reserva actualitzant el valor de reserve_available
        S'actualitza la congestió del sistema basantse en si les probabilitats donades.
        Si el travaill restant es 0, es considera com terminat i es dona una recompensa de 1. Pero, si es cumplaixen certes condicions i es te el bonus de prioritat, es dona un valor de 1.5 de recompensa.
        Si encara queda travaill restant, i no s'ha aturat per ninguna rao, es considera com no terminat i s'actualitzan els valors.
        """
        assert self.action_space.contains(action), \
            "Error, acció = {}".format(action)

        params = self._action_parameters(action)

        random_event = self.np_random.random()

        if random_event < params["incident_probability"]:
            return (
                self._get_obs(),
                -1.0,
                True,
                False,
                self._get_info(outcome="incident")
            )

        if random_event < (
            params["incident_probability"]
            + params["delay_probability"]
        ):
            return (
                self._get_obs(),
                0.0,
                True,
                False,
                self._get_info(outcome="delay")
            )

        progress = int(
            self.np_random.choice(
                params["progress_values"],
                p=params["progress_probabilities"]
            )
        )

        self.remaining_work = max(
            0,
            self.remaining_work - progress
        )

        if (
            self.reserve_available
            and self.np_random.random()
            < params["reserve_loss_probability"]
        ):
            self.reserve_available = False

        change = int(
            self.np_random.choice(
                params["congestion_changes"],
                p=params["congestion_probabilities"]
            )
        )

        self.congestion = int(
            np.clip(self.congestion + change, 0, 10)
        )

        if self.remaining_work == 0:

            reward = 1.0
            outcome = "success"

            if (
                self.priority_bonus
                and action == self.EXPRESS_ROUTE
                and self.reserve_available
                and self.congestion <= 3
            ):
                reward = 1.5
                outcome = "priority_success"

            return (
                self._get_obs(),
                reward,
                True,
                False,
                self._get_info(outcome=outcome)
            )

        return (
            self._get_obs(),
            0.0,
            False,
            False,
            self._get_info(outcome="running")
        )

    def _get_obs(self):
        """
        El metode _get_obs() retorna l'observació actual de l'entorn com una tupla que conté els valors de les variables d'estat: remaining_work, congestion i reserve_available.
        Aquests valors es converteixen a enters abans de ser retornats.
        """
        return (
            int(self.remaining_work),
            int(self.congestion),
            int(self.reserve_available)
        )

    def _get_info(self, outcome=None):
        """
        El metode _get_info() retorna un diccionari amb informació addicional sobre l'estat actual de l'entorn.
        En aquest cas, el diccionari conté nomes una clau "outcome" que indica el resultat de l'última acció realitzada.
        """
        return {
            "outcome": outcome
        }


WarehouseEnv = WarehouseRoutingEnv
