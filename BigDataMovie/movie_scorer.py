import numpy as np
import math


class MovieScorer:
    """
    电影评分预测器（改进版）
    改进点：
    1. 人物评分使用贝叶斯平均，避免少样本偏差
    2. 合作次数采用对数压缩，符合边际效应递减
    3. 未合作关系评分使用人物能力均值，而非固定常数
    """

    def __init__(self, G, person_stats_df, bayes_k=5):
        self.G = G
        self.person_stats = person_stats_df.set_index("name")
        self.global_avg = self.person_stats["avg_rating"].mean()

        # 贝叶斯平滑参数（经验值 5~10 都合理）
        self.bayes_k = bayes_k

    # =========================
    # 1️⃣ 人物评分（贝叶斯平均）
    # =========================
    def get_person_rating(self, name):
        """
        使用贝叶斯平均的人物评分：
        score = (n * avg_rating + k * global_avg) / (n + k)
        """
        if name in self.person_stats.index:
            row = self.person_stats.loc[name]
            avg_rating = row["avg_rating"]
            movie_count = row["movie_count"]

            if not np.isnan(avg_rating) and movie_count > 0:
                return (
                    movie_count * avg_rating +
                    self.bayes_k * self.global_avg
                ) / (movie_count + self.bayes_k)

        # 冷启动：完全没有数据
        return self.global_avg

    # =========================
    # 2️⃣ 角色评分
    # =========================
    def calc_actor_score(self, actors):
        if not actors:
            return self.global_avg
        return np.mean([self.get_person_rating(a) for a in actors])

    def calc_director_score(self, directors):
        if not directors:
            return self.global_avg
        return np.mean([self.get_person_rating(d) for d in directors])

    # =========================
    # 3️⃣ 关系评分（改进重点）
    # =========================
    def calc_relation_score(self, directors, actors, alpha=0.5):
        """
        关系评分由两部分组成：
        - 合作强度（对数压缩后的合作次数）
        - 合作质量（历史合作电影评分）
        """
        scores = []

        for d in directors:
            d_score = self.get_person_rating(d)

            for a in actors:
                a_score = self.get_person_rating(a)

                # -------- 有合作关系 --------
                if self.G.has_edge(d, a):
                    edge = self.G[d][a]

                    # 合作次数（对数压缩 → 0~10）
                    weight = edge.get("weight", 1)
                    coop_score = (
                        math.log1p(weight) /
                        math.log1p(10)
                    ) * 10

                    # 合作电影平均评分
                    rating = edge.get("avg_rating", self.global_avg)

                    score = alpha * coop_score + (1 - alpha) * rating

                # -------- 无合作关系 --------
                else:
                    # 使用导演 + 演员能力均值作为基线
                    score = (d_score + a_score) / 2

                scores.append(score)

        if not scores:
            return self.global_avg

        return np.mean(scores)

    # =========================
    # 4️⃣ 总评分预测
    # =========================
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
