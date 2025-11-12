import numpy as np
from copy import deepcopy
from .iforest import fit_iforest, evaluate_iforest   # adapt names if needed
from .cere import CERE

class GAOptimizer:
    def __init__(self, X_train, y_val=None, pop_size=6, generations=4, seed=42):
        self.X_train = X_train
        self.y_val = y_val
        self.pop_size = pop_size
        self.generations = generations
        self.rng = np.random.RandomState(seed)

    def init_population(self):
        pop = []
        for _ in range(self.pop_size):
            indiv = {
                "cere_seed": self.rng.randint(0, 10000),
                "d_out": self.rng.choice([64, 128, 256]),
                "depth": self.rng.choice([1, 2]),
            }
            pop.append(indiv)
        return pop

    def mutate(self, indiv):
        child = deepcopy(indiv)
        if self.rng.rand() < 0.5:
            child["cere_seed"] = self.rng.randint(0, 10000)
        if self.rng.rand() < 0.3:
            child["d_out"] = self.rng.choice([64, 128, 256])
        if self.rng.rand() < 0.3:
            child["depth"] = self.rng.choice([1, 2])
        return child

    def evaluate(self, indiv):
        cere = CERE(input_dim=self.X_train.shape[1],
                    d_out=indiv["d_out"],
                    depth=indiv["depth"],
                    seed=indiv["cere_seed"])
        X_repr = cere.transform(self.X_train)
        forest = fit_iforest(X_repr)                 # build ODIF forest
        score = evaluate_iforest(forest, X_repr, self.y_val)
        return score

    def run(self):
        pop = self.init_population()
        scores = [self.evaluate(ind) for ind in pop]
        for gen in range(self.generations):
            print(f"\n🌱 Generation {gen+1}/{self.generations}")
            new_pop, new_scores = [], []
            for ind in pop:
                child = self.mutate(ind)
                score = self.evaluate(child)
                new_pop.append(child)
                new_scores.append(score)
            combined = pop + new_pop
            combined_scores = np.array(scores + new_scores)
            top_idx = combined_scores.argsort()[::-1][:self.pop_size]
            pop = [combined[i] for i in top_idx]
            scores = [combined_scores[i] for i in top_idx]
            print(f"  Best score: {scores[0]:.4f}")
        return pop[0], scores[0]
