
class Player(object):
    def __init__(self, name, score=100, wins=0, matches=0):
        self.name = name
        self.score = score
        self.wins = wins
        self.matches = matches
        self.ys = []
        self.index = []
        self.mean=0
        self.std=0
