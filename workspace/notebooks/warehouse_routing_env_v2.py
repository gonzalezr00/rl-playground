
from warehouse_routing_env import WarehouseRoutingEnv as BaseWarehouseRoutingEnv


class WarehouseRoutingEnv(BaseWarehouseRoutingEnv):
    """
    Completeu-ho a l'Exercici 1.2.
    """
    def __init__(self, priority_bonus=True):
            """
            Es defineix el parametre priority_bonus com a True per defecte, el que significa que es concedirà un bonus de prioritat en cas d'èxit, i es passa aquest valor al constructor de la clase base mitjançant super().__init__().
            """
            super().__init__(priority_bonus=priority_bonus)



WarehouseEnv = WarehouseRoutingEnv
