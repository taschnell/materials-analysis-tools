import elli

from elli.dispersions import CodyLorentz


film = CodyLorentz()

air = elli.IsotropicMaterial(1.0)

glass = elli.IsotropicMaterial(1.52**2)


print(type(film))

