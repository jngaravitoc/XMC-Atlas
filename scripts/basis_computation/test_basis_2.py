import pyEXP

if __name__ == "__main__":

    with open("test_basis.yaml", "r") as f:
        config = f.read()

    pyEXP.basis.Basis.factory(config)

