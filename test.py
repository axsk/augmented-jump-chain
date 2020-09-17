import birthdeath as bd


def test_temporal_gillespie(n=10):
    p = bd.BirthDeath(n_t=1)
    xs, ts = p.gillespie(0, 0, 10)
