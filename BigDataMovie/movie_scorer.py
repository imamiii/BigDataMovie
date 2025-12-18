import numpy as np

class MovieScorer:
    def __init__(self, G, person_stats_df):
        self.G = G
        self.person_stats = person_stats_df.set_index("name")
        self.global_avg = self.person_stats["avg_rating"].mean()

    def get_person_rating(self, name):
        if name in self.person_stats.index:
            rating = self.person_stats.loc[name, "avg_rating"]
            if not np.isnan(rating):
                return rating
        return self.global_avg  # 冷启动

    def calc_actor_score(self, actors):
        if not actors:
            return self.global_avg
        return np.mean([self.get_person_rating(a) for a in actors])

    def calc_director_score(self, directors):
        if not directors:
            return self.global_avg
        return np.mean([self.get_person_rating(d) for d in directors])

    def calc_relation_score(self, directors, actors, alpha=0.5):
        scores = []

        for d in directors:
            for a in actors:
                if self.G.has_edge(d, a):
                    edge = self.G[d][a]
                    coop = min(edge.get("weight", 1), 5) / 5 * 10
                    rating = edge.get("avg_rating", self.global_avg)
                    score = alpha * coop + (1 - alpha) * rating
                else:
                    score = 5.5  # 未合作惩罚
                scores.append(score)

        if not scores:
            return self.global_avg
        return np.mean(scores)

    def predict(self, directors, actors, weights=None):
        if weights is None:
            weights = {
                "actor": 0.4,
                "director": 0.4,
                "relation": 0.2
            }

        s_actor = self.calc_actor_score(actors)
        s_director = self.calc_director_score(directors)
        s_relation = self.calc_relation_score(directors, actors)

        final_score = (
            weights["actor"] * s_actor +
            weights["director"] * s_director +
            weights["relation"] * s_relation
        )

        return {
            "actor_score": round(s_actor, 2),
            "director_score": round(s_director, 2),
            "relation_score": round(s_relation, 2),
            "final_score": round(min(10, max(0, final_score)), 1)
        }
