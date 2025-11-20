import numpy as np
from .iforest import ODIForest
from .cere import CERE
from .metrics import pr_auc


class GAOptimizer:
    def __init__(
        self,
        X_train, y_train, X_val, y_val,
        feature_count,
        pop_size=8,
        generations=5,
        mutation_rate=0.2,
        seed=42
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.feature_count = feature_count
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        # unified random generator
        self.rng = np.random.RandomState(seed)

        # Search spaces
        self.depth_choices = [1, 2, 3]
        self.d_out_choices = [64, 128, 256, 512]
        self.activation_choices = ["relu", "gelu"]

        self.t_choices = [50, 100, 150]
        self.psi_choices = [64, 128, 256, 512]
        self.depth_choices_tree = [None, 6, 8, 10]

    # --------------------------------------------------------------------
    # RANDOM CHROMOSOME (full)
    # --------------------------------------------------------------------
    def random_chromosome(self):
        return {
            "feature_mask": self.rng.randint(0, 2, size=self.feature_count),

            # CERE architecture
            "depth": self.rng.choice(self.depth_choices),
            "d_out": self.rng.choice(self.d_out_choices),
            "activation": self.rng.choice(self.activation_choices),

            # ODIF parameters
            "t": self.rng.choice(self.t_choices),
            "psi": self.rng.choice(self.psi_choices),
            "max_depth": self.rng.choice(self.depth_choices_tree)
        }

    # --------------------------------------------------------------------
    # FITNESS FUNCTION
    # --------------------------------------------------------------------
    def fitness(self, chrom):
        mask = chrom["feature_mask"].astype(bool)

        if mask.sum() == 0:
            return 0.0

        Xtr = self.X_train[:, mask]
        Xv = self.X_val[:, mask]

        cere = CERE(
            input_dim=Xtr.shape[1],
            d_out=chrom["d_out"],
            depth=chrom["depth"],
            activation=chrom["activation"],
            seed=0
        )

        forest = ODIForest(
            t=chrom["t"],
            psi=chrom["psi"],
            max_depth=chrom["max_depth"],
            seed=0
        ).fit(Xtr, cere=cere)

        scores = forest.score_samples(Xv, cere=cere)
        return pr_auc(self.y_val, scores)

    # --------------------------------------------------------------------
    # MUTATION
    # --------------------------------------------------------------------
    def mutate(self, chrom):
        new = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in chrom.items()
        }

        # feature mask mutation
        for i in range(self.feature_count):
            if self.rng.rand() < self.mutation_rate:
                new["feature_mask"][i] ^= 1

        # CERE architecture mutation
        if self.rng.rand() < self.mutation_rate:
            new["depth"] = self.rng.choice(self.depth_choices)

        if self.rng.rand() < self.mutation_rate:
            new["d_out"] = self.rng.choice(self.d_out_choices)

        if self.rng.rand() < self.mutation_rate:
            new["activation"] = self.rng.choice(self.activation_choices)

        # ODIF mutation
        if self.rng.rand() < self.mutation_rate:
            new["t"] = self.rng.choice(self.t_choices)

        if self.rng.rand() < self.mutation_rate:
            new["psi"] = self.rng.choice(self.psi_choices)

        if self.rng.rand() < self.mutation_rate:
            new["max_depth"] = self.rng.choice(self.depth_choices_tree)

        return new

    # --------------------------------------------------------------------
    # CROSSOVER
    # --------------------------------------------------------------------
    def crossover(self, p1, p2):
        child = {}

        # feature mask 1-point crossover
        cut = self.rng.randint(1, self.feature_count - 1)
        child["feature_mask"] = np.concatenate([
            p1["feature_mask"][:cut],
            p2["feature_mask"][cut:]
        ])

        # CERE architecture 50-50 mix
        child["depth"] = p1["depth"] if self.rng.rand() < 0.5 else p2["depth"]
        child["d_out"] = p1["d_out"] if self.rng.rand() < 0.5 else p2["d_out"]
        child["activation"] = p1["activation"] if self.rng.rand() < 0.5 else p2["activation"]

        # ODIF 50-50 mix
        child["t"] = p1["t"] if self.rng.rand() < 0.5 else p2["t"]
        child["psi"] = p1["psi"] if self.rng.rand() < 0.5 else p2["psi"]
        child["max_depth"] = p1["max_depth"] if self.rng.rand() < 0.5 else p2["max_depth"]

        return child

    # --------------------------------------------------------------------
    # GA MAIN LOOP
    # --------------------------------------------------------------------
    def run(self):
        population = [self.random_chromosome() for _ in range(self.pop_size)]

        for gen in range(self.generations):
            fitnesses = [self.fitness(ch) for ch in population]

            # select best 2
            parents_idx = np.argsort(fitnesses)[::-1][:2]
            p1, p2 = population[parents_idx[0]], population[parents_idx[1]]

            new_pop = [p1, p2]

            # generate children
            while len(new_pop) < self.pop_size:
                c = self.crossover(p1, p2)
                c = self.mutate(c)
                new_pop.append(c)

            population = new_pop
            print(f"[GEN {gen+1}] Best PR-AUC = {max(fitnesses):.6f}")

        final_fits = [self.fitness(ch) for ch in population]
        best_idx = np.argmax(final_fits)
        best = population[best_idx]

        print("\n=== BEST CHROMOSOME FOUND ===")
        print(best)

        return best
